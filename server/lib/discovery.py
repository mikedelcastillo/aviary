"""Network discovery of Tapo RTSP cameras on the local subnet.

Cameras get DHCP leases that we can't pin in config, so instead of hardcoding
IPs we sweep the subnet and probe each host for a working RTSP stream. The whole
module is pure stdlib (no opencv, no external deps) so it can run in the lightweight
``/discover`` path and be unit-tested against a localhost fake server WITHOUT
pulling in the heavy ML stack.

The sweep fans out one authenticated RTSP probe per candidate host. Earlier
versions did a preliminary throwaway TCP connect to ``:rtsp_port`` and then a
second connection for DESCRIBE; that was fast, but it was also easy to race tiny
WiFi cameras that already had live RTSP consumers. A single DESCRIBE connection
per host is slower by only the connect timeout on dead addresses and is much
less likely to make real cameras disappear intermittently.

We deliberately speak raw RTSP over a socket rather than shelling out to ffmpeg
or opencv: it keeps the probe dependency-free, lets us distinguish "wrong
password" (AUTH_FAILED) from "no such stream" (STREAM_NOT_FOUND) precisely, and
makes the digest computation deterministic so the tests can recompute and verify
the exact response we send.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import logging
import re
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Callable
from urllib.parse import quote

from lib.config import CameraCredentials, DiscoveryConfig


LOGGER = logging.getLogger("lib.discovery")


def build_rtsp_url(credentials: CameraCredentials, host: str, port: int, stream_path: str) -> str:
    """Assemble a credentialed RTSP URL safe to hand to ffmpeg/OpenCV.

    The username and password are percent-encoded: Tapo camera passwords are
    user-chosen and routinely contain characters that are special in a URL
    (``@ : / # ? %`` ...). The DESCRIBE probe authenticates via header digest, so
    it confirms a camera regardless — but an unencoded credential here produces a
    malformed URL that ffmpeg silently fails (or hangs) to open, which shows up
    as a camera stuck on "connecting". Encoding keeps the URL well-formed.
    """
    user = quote(credentials.username, safe="")
    secret = quote(credentials.password, safe="")
    return f"rtsp://{user}:{secret}@{host}:{port}{stream_path}"


def redact_rtsp_url(url: str) -> str:
    """Mask the password in an RTSP URL for safe logging."""
    return re.sub(r"(rtsp://[^:/@]+:)[^@/]*@", r"\1***@", url, count=1)


# Per-host states published live during a sweep for the dashboard's discovery
# grid. Kept as plain strings (not an Enum) so the dashboard can colour-map them
# without importing anything heavier.
HOST_PENDING = "pending"   # queued, not yet probed        (grey)
HOST_TESTING = "testing"   # probe in flight               (yellow)
HOST_FOUND = "found"       # confirmed camera              (green)
HOST_FAILED = "failed"     # port closed / auth / no stream (red)


class DiscoveryProgress:
    """Thread-safe, live view of an in-flight discovery sweep.

    Discovery runs across a pool of worker threads; each publishes its host's
    state transitions here. The dashboard render thread reads :meth:`snapshot`
    every frame to draw the coloured host grid. A single short-held lock keeps
    the two sides decoupled — neither blocks the other for more than a dict poke.

    A fresh sweep calls :meth:`begin` (seeding every host as pending and flipping
    ``active`` on) and :meth:`end` in a ``finally`` (flipping ``active`` off, so
    the dashboard reverts to the normal camera band once the scan settles).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, str] = {}
        self._order: list[str] = []   # original (ascending) host order for the grid
        self._network = ""            # display prefix, e.g. "192.168.1."
        self._active = False

    def begin(self, hosts: list[str], network: str = "") -> None:
        with self._lock:
            self._order = list(hosts)
            self._states = {host: HOST_PENDING for host in hosts}
            self._network = network
            self._active = True

    def mark(self, host: str, state: str) -> None:
        with self._lock:
            if host in self._states:
                self._states[host] = state

    def end(self) -> None:
        with self._lock:
            self._active = False

    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def snapshot(self) -> dict:
        with self._lock:
            states = dict(self._states)
            order = list(self._order)
            network = self._network
            active = self._active
        counts = {HOST_PENDING: 0, HOST_TESTING: 0, HOST_FOUND: 0, HOST_FAILED: 0}
        for state in states.values():
            counts[state] = counts.get(state, 0) + 1
        return {
            "active": active,
            "network": network,
            "order": order,
            "states": states,
            "counts": counts,
        }


@dataclass(frozen=True)
class DiscoveredCamera:
    host: str
    port: int
    stream_path: str
    # Full, credentialed URL ready to hand straight to cv2.VideoCapture. We build
    # it here (rather than in the supervisor) so the credentials live in exactly
    # one place per discovered host.
    rtsp_url: str


@dataclass(frozen=True)
class DiscoveryResult:
    cameras: list[DiscoveredCamera]   # confirmed: auth ok AND stream path ok
    hosts_scanned: int
    ports_open: int                   # hosts with :port reachable
    auth_failures: int                # :port open + RTSP reachable but creds rejected
    elapsed_seconds: float


class _ProbeOutcome(Enum):
    # Internal classification of a single host's RTSP DESCRIBE handshake. Kept
    # private; callers only see the aggregated DiscoveryResult.
    CONFIRMED = "confirmed"            # 200 OK -> a real, authenticated stream
    AUTH_FAILED = "auth_failed"        # 401 even after presenting credentials
    STREAM_NOT_FOUND = "stream_404"    # reachable + authed but no such stream path
    ERROR = "error"                    # port open but RTSP/protocol failed
    PORT_CLOSED = "port_closed"        # socket connect failed/timed out


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# WWW-Authenticate parameters arrive as either key="quoted value" or bare
# key=token. This matches both and yields (key, quoted, bare); exactly one of the
# value groups is populated per match.
_AUTH_PARAM_RE = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|([^\s,]+))')


def _parse_auth_params(header_value: str) -> dict[str, str]:
    """Parse the parameters of a WWW-Authenticate header into a dict.

    Lower-cases keys so callers don't have to worry about header casing. The
    leading scheme token (e.g. ``Digest``) is not a key=value pair and is simply
    skipped by the regex.
    """
    params: dict[str, str] = {}
    for key, quoted, bare in _AUTH_PARAM_RE.findall(header_value):
        params[key.lower()] = quoted if quoted != "" or bare == "" else bare
    return params


def _default_cidr() -> str:
    """Best-effort guess of the host's primary /24 subnet.

    We open a UDP socket "towards" a public address: no packets are actually
    sent (UDP connect just sets the default route's source address), but it makes
    the OS pick the outbound interface so ``getsockname`` reveals which local
    IPv4 the camera traffic would use. Isolated in its own function so tests can
    monkeypatch it instead of touching the real network.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        local_ip = sock.getsockname()[0]
    finally:
        sock.close()
    return f"{local_ip}/24"


def _target_hosts(discovery: DiscoveryConfig) -> list[str]:
    """Resolve the configured discovery scope into a concrete list of host IPs.

    Precedence (most explicit wins): an explicit ``hosts`` list, then an explicit
    ``cidr``, then the auto-detected /24. ``strict=False`` lets a host address
    like ``192.168.1.50/24`` be treated as the network it belongs to.
    """
    if discovery.hosts:
        return list(discovery.hosts)
    cidr = discovery.cidr or _default_cidr()
    network = ipaddress.ip_network(cidr, strict=False)
    return [str(host) for host in network.hosts()]


def _read_headers(sock: socket.socket) -> bytes:
    """Read from the socket until the end of the RTSP/HTTP header block.

    RTSP responses end their headers with a blank line (CRLFCRLF). We don't care
    about the SDP body, so we stop there. Reading is bounded by the socket
    timeout set by the caller; an empty read (peer closed) also stops us.
    """
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
        # Cheap guard against a misbehaving peer streaming forever before the
        # blank line; real RTSP headers are tiny.
        if len(data) > 65536:
            break
    return data


def _status_code(response: bytes) -> int:
    """Extract the numeric status code from an RTSP response's first line.

    Expected form: ``RTSP/1.0 <code> <reason>``. Returns 0 if it can't be parsed
    so the caller classifies the host as ERROR rather than crashing.
    """
    try:
        first_line = response.split(b"\r\n", 1)[0].decode("latin-1")
        return int(first_line.split()[1])
    except (IndexError, ValueError):
        return 0


def _www_authenticate(response: bytes) -> str | None:
    """Return the preferred WWW-Authenticate challenge, case-insensitively.

    A server may advertise several schemes on separate WWW-Authenticate lines
    (RFC 7235 allows it, and some Tapo firmwares send a ``Basic`` line before a
    ``Digest`` one). We collect every challenge and prefer ``Digest`` over
    ``Basic`` so we never downgrade to Basic just because it was listed first.
    """
    challenges: list[str] = []
    for line in response.split(b"\r\n"):
        text = line.decode("latin-1")
        name, sep, value = text.partition(":")
        if sep and name.strip().lower() == "www-authenticate":
            challenges.append(value.strip())
    for challenge in challenges:
        if challenge.lower().startswith("digest"):
            return challenge
    return challenges[0] if challenges else None


def _build_digest_header(
    *,
    username: str,
    password: str,
    uri: str,
    params: dict[str, str],
) -> str:
    """Compute an RFC 2617 Digest Authorization header for a DESCRIBE request.

    We deliberately use a STABLE, derived cnonce (md5 of the server nonce,
    truncated) instead of os.urandom/random. The server only requires the
    response hash to be consistent with the cnonce/nc we send, so a deterministic
    cnonce is fully valid -- and it keeps the probe reproducible so tests can
    recompute the exact expected response.
    """
    realm = params.get("realm", "")
    nonce = params.get("nonce", "")
    qop = params.get("qop", "")
    algorithm = params.get("algorithm", "")

    ha1 = _md5(f"{username}:{realm}:{password}")
    ha2 = _md5(f"DESCRIBE:{uri}")

    fields = [
        f'username="{username}"',
        f'realm="{realm}"',
        f'nonce="{nonce}"',
        f'uri="{uri}"',
    ]

    # qop may be a comma-separated list (e.g. "auth,auth-int"); we only support
    # plain "auth" and pick it if offered.
    if "auth" in [token.strip() for token in qop.split(",") if token.strip()]:
        nc = "00000001"
        cnonce = _md5(nonce)[:16]
        response = _md5(f"{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}")
        fields.append(f'response="{response}"')
        fields.append("qop=auth")
        fields.append(f"nc={nc}")
        fields.append(f'cnonce="{cnonce}"')
    else:
        response = _md5(f"{ha1}:{nonce}:{ha2}")
        fields.append(f'response="{response}"')

    if algorithm:
        fields.append(f"algorithm={algorithm}")

    return "Digest " + ", ".join(fields)


def _describe_request(uri: str, cseq: int, *, authorization: str | None = None) -> bytes:
    """Assemble a raw RTSP DESCRIBE request as bytes."""
    lines = [
        f"DESCRIBE {uri} RTSP/1.0",
        f"CSeq: {cseq}",
        "User-Agent: aviary-discovery",
        "Accept: application/sdp",
    ]
    if authorization is not None:
        lines.append(f"Authorization: {authorization}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


def _probe_rtsp_once(
    host: str,
    discovery: DiscoveryConfig,
    credentials: CameraCredentials,
) -> _ProbeOutcome:
    """Perform an authenticated RTSP DESCRIBE and classify the host.

    The flow mirrors what a real RTSP client does: DESCRIBE once, and if the
    server challenges with 401, recompute and resend with an Authorization
    header. We translate the final status into a single outcome so the aggregate
    counters (confirmed / auth_failures) are unambiguous.
    """
    uri = f"rtsp://{host}:{discovery.rtsp_port}{discovery.stream_path}"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(discovery.connect_timeout_seconds)
        sock.connect((host, discovery.rtsp_port))
    except OSError:
        return _ProbeOutcome.PORT_CLOSED

    sock.settimeout(discovery.rtsp_timeout_seconds)
    try:
        # First, unauthenticated, DESCRIBE.
        sock.sendall(_describe_request(uri, cseq=1))
        response = _read_headers(sock)
        code = _status_code(response)

        if code == 200:
            # Some cameras serve the stream without auth.
            return _ProbeOutcome.CONFIRMED
        if code == 404:
            return _ProbeOutcome.STREAM_NOT_FOUND
        if code != 401:
            return _ProbeOutcome.ERROR

        # 401 -> authenticate. Prefer Digest; fall back to Basic if that's all
        # the camera offers.
        challenge = _www_authenticate(response)
        if challenge is None:
            return _ProbeOutcome.ERROR

        if challenge.lower().startswith("digest"):
            params = _parse_auth_params(challenge)
            authorization = _build_digest_header(
                username=credentials.username,
                password=credentials.password,
                uri=uri,
                params=params,
            )
        elif challenge.lower().startswith("basic"):
            token = base64.b64encode(
                f"{credentials.username}:{credentials.password}".encode()
            ).decode("ascii")
            authorization = f"Basic {token}"
        else:
            return _ProbeOutcome.ERROR

        sock.sendall(_describe_request(uri, cseq=2, authorization=authorization))
        response = _read_headers(sock)
        code = _status_code(response)

        if code == 200:
            return _ProbeOutcome.CONFIRMED
        if code == 401:
            return _ProbeOutcome.AUTH_FAILED
        # 404 (not found), 455 (method not valid in state), 457 (invalid range)
        # all indicate the host is a camera we authed against but the configured
        # stream path isn't usable.
        if code in (404, 455, 457):
            return _ProbeOutcome.STREAM_NOT_FOUND
        return _ProbeOutcome.ERROR
    except OSError:
        return _ProbeOutcome.ERROR
    finally:
        sock.close()


def _probe_rtsp(
    host: str,
    discovery: DiscoveryConfig,
    credentials: CameraCredentials,
) -> _ProbeOutcome:
    """Probe RTSP with bounded retries for transient camera/network stalls."""
    attempts = max(1, discovery.probe_attempts)
    retryable = {_ProbeOutcome.PORT_CLOSED, _ProbeOutcome.ERROR}
    outcome = _ProbeOutcome.ERROR
    for attempt in range(1, attempts + 1):
        outcome = _probe_rtsp_once(host, discovery, credentials)
        if outcome not in retryable or attempt == attempts:
            return outcome
        if discovery.probe_retry_delay_seconds > 0:
            time.sleep(discovery.probe_retry_delay_seconds)
    return outcome


def _network_prefix(hosts: list[str]) -> str:
    """The shared ``a.b.c.`` prefix for a /24-style host list (display only)."""
    if hosts and hosts[0].count(".") == 3:
        return hosts[0].rsplit(".", 1)[0] + "."
    return ""


def discover_cameras(
    discovery: DiscoveryConfig,
    credentials: CameraCredentials,
    *,
    clock: Callable[[], float] = time.monotonic,
    progress: DiscoveryProgress | None = None,
) -> DiscoveryResult:
    """Sweep the configured scope and return the confirmed cameras plus stats.

    ``clock`` is injectable so tests can assert on elapsed time deterministically.
    The sweep runs on a ThreadPoolExecutor and each worker performs the actual
    RTSP DESCRIBE handshake for its host. When a ``progress`` sink is supplied,
    each worker publishes its host's live state to it so the dashboard can render
    the colour-coded discovery grid in real time.
    """
    start = clock()
    hosts = _target_hosts(discovery)
    if hosts:
        LOGGER.info(
            "Discovery: probing :%d%s across %d host(s) (%s ... %s), max %d at a time",
            discovery.rtsp_port,
            discovery.stream_path,
            len(hosts),
            hosts[0],
            hosts[-1],
            discovery.max_workers,
        )
    else:
        LOGGER.info("Discovery: no hosts in scope to scan")

    if progress is not None:
        progress.begin(hosts, _network_prefix(hosts))

    def _probe(host: str) -> _ProbeOutcome:
        # Logs + marks state in the worker thread so progress streams in real
        # time (in completion order).
        if progress is not None:
            progress.mark(host, HOST_TESTING)
        LOGGER.debug(
            "Discovery: testing %s:%d%s",
            host,
            discovery.rtsp_port,
            discovery.stream_path,
        )
        outcome = _probe_rtsp(host, discovery, credentials)
        if outcome is _ProbeOutcome.CONFIRMED:
            if progress is not None:
                progress.mark(host, HOST_FOUND)
            LOGGER.info("Discovery: %s CONFIRMED (auth + %s OK)", host, discovery.stream_path)
        elif outcome is _ProbeOutcome.AUTH_FAILED:
            if progress is not None:
                progress.mark(host, HOST_FAILED)
            LOGGER.warning("Discovery: %s credentials REJECTED (check TAPO_CREDENTIALS)", host)
        elif outcome is _ProbeOutcome.STREAM_NOT_FOUND:
            if progress is not None:
                progress.mark(host, HOST_FAILED)
            LOGGER.info(
                "Discovery: %s reachable but %s not available", host, discovery.stream_path
            )
        elif outcome is _ProbeOutcome.PORT_CLOSED:
            if progress is not None:
                progress.mark(host, HOST_FAILED)
            LOGGER.debug("Discovery: %s did not answer on :%d", host, discovery.rtsp_port)
        else:
            if progress is not None:
                progress.mark(host, HOST_FAILED)
            LOGGER.debug("Discovery: %s probe error", host)
        return outcome

    try:
        with ThreadPoolExecutor(max_workers=discovery.max_workers) as pool:
            outcomes = list(pool.map(_probe, hosts))

        reachable = [
            host for host, outcome in zip(hosts, outcomes)
            if outcome is not _ProbeOutcome.PORT_CLOSED
        ]
        if reachable:
            LOGGER.info(
                "Discovery: %d host(s) answered on :%d: %s",
                len(reachable),
                discovery.rtsp_port,
                ", ".join(reachable),
            )
        else:
            LOGGER.info("Discovery: no host answered on :%d", discovery.rtsp_port)

        cameras: list[DiscoveredCamera] = []
        auth_failures = 0
        for host, outcome in zip(hosts, outcomes):
            if outcome is _ProbeOutcome.CONFIRMED:
                rtsp_url = build_rtsp_url(
                    credentials, host, discovery.rtsp_port, discovery.stream_path
                )
                cameras.append(
                    DiscoveredCamera(
                        host=host,
                        port=discovery.rtsp_port,
                        stream_path=discovery.stream_path,
                        rtsp_url=rtsp_url,
                    )
                )
            elif outcome is _ProbeOutcome.AUTH_FAILED:
                auth_failures += 1

        elapsed = clock() - start
        LOGGER.info(
            "Discovery: done in %.1fs — %d confirmed, %d auth failure(s), "
            "%d port(s) open of %d scanned",
            elapsed,
            len(cameras),
            auth_failures,
            len(reachable),
            len(hosts),
        )

        return DiscoveryResult(
            cameras=cameras,
            hosts_scanned=len(hosts),
            ports_open=len(reachable),
            auth_failures=auth_failures,
            elapsed_seconds=elapsed,
        )
    finally:
        # Always clear the active flag so the dashboard reverts to the camera
        # band even if the sweep raised partway through.
        if progress is not None:
            progress.end()


def main() -> None:
    """Standalone discovery diagnostic: scan the LAN and print what was found.

    Run it dashboard-free to see exactly which cameras discovery confirms and the
    URLs it builds (password masked), e.g. ``python -m lib.discovery``. Reads
    ``TAPO_CREDENTIALS`` (and optional ``TAPO_DISCOVERY_CIDR``) from the env/.env;
    it does NOT require ``MODEL_PATH`` so it can run independently of the server.
    """
    import os

    from dotenv import load_dotenv

    from lib.config import DiscoveryConfig, _credentials

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    discovery = DiscoveryConfig(cidr=os.environ.get("TAPO_DISCOVERY_CIDR") or None)
    result = discover_cameras(discovery, _credentials())

    print()
    print(f"Scanned {result.hosts_scanned} host(s) in {result.elapsed_seconds:.1f}s")
    print(f"Ports open on :{discovery.rtsp_port}: {result.ports_open}")
    print(f"Credential rejections: {result.auth_failures}")
    print(f"Confirmed cameras: {len(result.cameras)}")
    for camera in result.cameras:
        print(f"  {camera.host} -> {redact_rtsp_url(camera.rtsp_url)}")
    if not result.cameras:
        print("  (none — if a camera you expect is missing, check TAPO_CREDENTIALS,")
        print("   the stream path in DiscoveryConfig, or TAPO_DISCOVERY_CIDR)")


if __name__ == "__main__":
    main()
