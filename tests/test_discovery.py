"""Tests for lib.discovery against localhost fake RTSP servers.

These spin up real TCP servers bound to 127.0.0.1 that emulate a Tapo camera's
RTSP DESCRIBE handshake. They never touch the real network and import only
lib.discovery + lib.config (NOT ultralytics), so they stay fast and isolated.

The "good" server reproduces the camera side of RFC 2617 Digest auth: it issues a
401 with a nonce, then RECOMPUTES the expected response from the parameters the
client echoes back and only returns 200 if they match -- so the test proves our
client builds a correct digest, not merely that it sends *some* Authorization
header.
"""

from __future__ import annotations

import base64
import hashlib
import re
import socket
import threading
import time

from lib.config import CameraCredentials, DiscoveryConfig
from lib.discovery import (
    HOST_FAILED,
    HOST_FOUND,
    HOST_TESTING,
    DiscoveryProgress,
    _ProbeOutcome,
    build_rtsp_url,
    discover_cameras,
    redact_rtsp_url,
)

USERNAME = "admin"
PASSWORD = "hunter2"
REALM = "Tapo RTSP Server"
NONCE = "0123456789abcdef"
STREAM_PATH = "/stream1"

_AUTH_PARAM_RE = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|([^\s,]+))')


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _parse_params(header_value: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for key, quoted, bare in _AUTH_PARAM_RE.findall(header_value):
        params[key.lower()] = quoted if quoted != "" or bare == "" else bare
    return params


def _read_request(conn: socket.socket) -> bytes:
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk
    return data


def _header_value(request: bytes, name: str) -> str | None:
    for line in request.split(b"\r\n"):
        text = line.decode("latin-1")
        key, sep, value = text.partition(":")
        if sep and key.strip().lower() == name.lower():
            return value.strip()
    return None


def _cseq(request: bytes) -> str:
    value = _header_value(request, "CSeq")
    return value or "1"


class _FakeRtspServer:
    """A localhost TCP server that runs a single response strategy per request.

    ``responder(conn, request) -> None`` writes whatever RTSP response the test
    scenario needs. The server accepts connections in a daemon thread until
    closed, so the discovery sweep's stage-1 (port probe) and stage-2 (DESCRIBE)
    connections are both served.
    """

    def __init__(self, responder) -> None:
        self._responder = responder
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._stop = False
        self._lock = threading.Lock()
        self._connections = 0
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            with self._lock:
                self._connections += 1
            try:
                self._responder(conn)
            except OSError:
                pass
            finally:
                conn.close()

    @property
    def connections(self) -> int:
        with self._lock:
            return self._connections

    def close(self) -> None:
        self._stop = True
        try:
            self._sock.close()
        except OSError:
            pass


def _send(conn: socket.socket, status_line: str, headers: list[str]) -> None:
    payload = status_line + "\r\n" + "\r\n".join(headers) + "\r\n\r\n"
    conn.sendall(payload.encode("latin-1"))


def _expected_digest_response(uri: str, params: dict[str, str]) -> str:
    """Recompute the digest response the camera expects from the client."""
    ha1 = _md5(f"{USERNAME}:{REALM}:{PASSWORD}")
    ha2 = _md5(f"DESCRIBE:{uri}")
    qop = params.get("qop")
    if qop == "auth":
        nc = params.get("nc", "")
        cnonce = params.get("cnonce", "")
        return _md5(f"{ha1}:{NONCE}:{nc}:{cnonce}:auth:{ha2}")
    return _md5(f"{ha1}:{NONCE}:{ha2}")


def _digest_good_responder(conn: socket.socket) -> None:
    """401(Digest) -> validate the client's response -> 200."""
    request = _read_request(conn)
    auth = _header_value(request, "Authorization")
    if auth is None:
        _send(
            conn,
            "RTSP/1.0 401 Unauthorized",
            [
                f"CSeq: {_cseq(request)}",
                f'WWW-Authenticate: Digest realm="{REALM}", '
                f'nonce="{NONCE}", qop="auth"',
            ],
        )
        # The client resends on the same connection; read and validate it.
        request = _read_request(conn)
        auth = _header_value(request, "Authorization")

    if auth is None or not auth.lower().startswith("digest"):
        _send(conn, "RTSP/1.0 400 Bad Request", [f"CSeq: {_cseq(request)}"])
        return

    params = _parse_params(auth)
    uri = params.get("uri", "")
    expected = _expected_digest_response(uri, params)
    if params.get("response") == expected:
        _send(
            conn,
            "RTSP/1.0 200 OK",
            [f"CSeq: {_cseq(request)}", "Content-Length: 0"],
        )
    else:
        _send(conn, "RTSP/1.0 401 Unauthorized", [f"CSeq: {_cseq(request)}"])


def _always_401_responder(conn: socket.socket) -> None:
    """Reject every DESCRIBE -- simulates wrong credentials.

    The client sends DESCRIBE, receives the 401 challenge, then RESENDS with an
    Authorization header on the same connection; we must answer that second
    request with another 401 (not close the socket) so the client classifies it
    as AUTH_FAILED rather than a connection ERROR.
    """
    for _ in range(2):
        request = _read_request(conn)
        if not request:
            return
        _send(
            conn,
            "RTSP/1.0 401 Unauthorized",
            [
                f"CSeq: {_cseq(request)}",
                f'WWW-Authenticate: Digest realm="{REALM}", nonce="{NONCE}"',
            ],
        )


def _auth_then_404_responder(conn: socket.socket) -> None:
    """401 first, then 404 after the client authenticates (stream not found)."""
    request = _read_request(conn)
    if _header_value(request, "Authorization") is None:
        _send(
            conn,
            "RTSP/1.0 401 Unauthorized",
            [
                f"CSeq: {_cseq(request)}",
                f'WWW-Authenticate: Digest realm="{REALM}", nonce="{NONCE}"',
            ],
        )
        request = _read_request(conn)
    _send(conn, "RTSP/1.0 404 Not Found", [f"CSeq: {_cseq(request)}"])


def _basic_good_responder(conn: socket.socket) -> None:
    """401(Basic) -> validate the client's Basic credentials -> 200.

    Exercises the Basic-auth fallback branch of the probe (some firmwares only
    offer Basic).
    """
    request = _read_request(conn)
    if _header_value(request, "Authorization") is None:
        _send(
            conn,
            "RTSP/1.0 401 Unauthorized",
            [f"CSeq: {_cseq(request)}", f'WWW-Authenticate: Basic realm="{REALM}"'],
        )
        request = _read_request(conn)
    auth = _header_value(request, "Authorization") or ""
    expected = base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode("ascii")
    if auth == f"Basic {expected}":
        _send(conn, "RTSP/1.0 200 OK", [f"CSeq: {_cseq(request)}", "Content-Length: 0"])
    else:
        _send(conn, "RTSP/1.0 401 Unauthorized", [f"CSeq: {_cseq(request)}"])


def _basic_then_digest_responder(conn: socket.socket) -> None:
    """Advertise Basic AND Digest on separate lines; require Digest.

    Guards the multi-line WWW-Authenticate handling: the client must prefer the
    Digest challenge even though Basic is listed first, so presenting Basic here
    would fail. Only a valid Digest response earns a 200.
    """
    request = _read_request(conn)
    auth = _header_value(request, "Authorization")
    if auth is None:
        _send(
            conn,
            "RTSP/1.0 401 Unauthorized",
            [
                f"CSeq: {_cseq(request)}",
                f'WWW-Authenticate: Basic realm="{REALM}"',
                f'WWW-Authenticate: Digest realm="{REALM}", '
                f'nonce="{NONCE}", qop="auth"',
            ],
        )
        request = _read_request(conn)
        auth = _header_value(request, "Authorization")

    if auth is None or not auth.lower().startswith("digest"):
        _send(conn, "RTSP/1.0 401 Unauthorized", [f"CSeq: {_cseq(request)}"])
        return
    params = _parse_params(auth)
    if params.get("response") == _expected_digest_response(params.get("uri", ""), params):
        _send(conn, "RTSP/1.0 200 OK", [f"CSeq: {_cseq(request)}", "Content-Length: 0"])
    else:
        _send(conn, "RTSP/1.0 401 Unauthorized", [f"CSeq: {_cseq(request)}"])


def _unknown_status_responder(conn: socket.socket) -> None:
    """Answer the first DESCRIBE with an unexpected code (neither 200/401/404)."""
    request = _read_request(conn)
    _send(conn, "RTSP/1.0 503 Service Unavailable", [f"CSeq: {_cseq(request)}"])


def _drop_once_then_digest_responder():
    attempts = 0
    lock = threading.Lock()

    def responder(conn: socket.socket) -> None:
        nonlocal attempts
        with lock:
            attempts += 1
            should_drop = attempts == 1
        if should_drop:
            return
        _digest_good_responder(conn)

    return responder


def _config_for(port: int) -> DiscoveryConfig:
    return DiscoveryConfig(
        hosts=("127.0.0.1",),
        rtsp_port=port,
        stream_path=STREAM_PATH,
        connect_timeout_seconds=2.0,
        rtsp_timeout_seconds=2.0,
    )


def _credentials() -> CameraCredentials:
    return CameraCredentials(username=USERNAME, password=PASSWORD)


def test_discover_confirms_camera_with_valid_digest() -> None:
    server = _FakeRtspServer(_digest_good_responder)
    try:
        result = discover_cameras(_config_for(server.port), _credentials())
    finally:
        server.close()

    assert len(result.cameras) == 1
    camera = result.cameras[0]
    assert camera.host == "127.0.0.1"
    assert camera.port == server.port
    assert camera.stream_path == STREAM_PATH
    assert camera.rtsp_url == (
        f"rtsp://{USERNAME}:{PASSWORD}@127.0.0.1:{server.port}{STREAM_PATH}"
    )
    assert result.hosts_scanned == 1
    assert result.ports_open == 1
    assert result.auth_failures == 0


def test_discover_uses_one_rtsp_connection_per_successful_host() -> None:
    server = _FakeRtspServer(_digest_good_responder)
    try:
        result = discover_cameras(_config_for(server.port), _credentials())
    finally:
        server.close()

    assert len(result.cameras) == 1
    assert server.connections == 1


def test_discover_retries_transient_rtsp_drop() -> None:
    server = _FakeRtspServer(_drop_once_then_digest_responder())
    config = DiscoveryConfig(
        hosts=("127.0.0.1",),
        rtsp_port=server.port,
        stream_path=STREAM_PATH,
        connect_timeout_seconds=2.0,
        rtsp_timeout_seconds=2.0,
        probe_attempts=2,
        probe_retry_delay_seconds=0.0,
    )
    try:
        result = discover_cameras(config, _credentials())
    finally:
        server.close()

    assert len(result.cameras) == 1
    assert result.ports_open == 1
    assert server.connections == 2


def test_discover_limits_concurrent_rtsp_probes(monkeypatch) -> None:
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_probe(host, discovery, credentials):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return _ProbeOutcome.PORT_CLOSED

    monkeypatch.setattr("lib.discovery._probe_rtsp", fake_probe)
    config = DiscoveryConfig(
        hosts=tuple(f"10.0.0.{index}" for index in range(1, 13)),
        max_workers=3,
        connect_timeout_seconds=0.01,
        rtsp_timeout_seconds=0.01,
    )

    result = discover_cameras(config, _credentials())

    assert result.hosts_scanned == 12
    assert max_active <= 3


def test_discover_publishes_live_progress(monkeypatch) -> None:
    snapshots: list[dict] = []

    def fake_probe(host, discovery, credentials):
        return _ProbeOutcome.CONFIRMED if host.endswith(".2") else _ProbeOutcome.PORT_CLOSED

    monkeypatch.setattr("lib.discovery._probe_rtsp", fake_probe)
    config = DiscoveryConfig(
        hosts=("10.0.0.1", "10.0.0.2"),
        max_workers=1,
        connect_timeout_seconds=0.01,
        rtsp_timeout_seconds=0.01,
    )

    result = discover_cameras(config, _credentials(), progress_callback=snapshots.append)

    assert len(result.cameras) == 1
    assert snapshots[0]["counts"][HOST_TESTING] == 0
    assert any(snapshot["counts"][HOST_TESTING] == 1 for snapshot in snapshots)
    assert any(snapshot["counts"][HOST_FOUND] == 1 for snapshot in snapshots)
    assert snapshots[-1]["active"] is False


def test_discover_counts_auth_failure() -> None:
    server = _FakeRtspServer(_always_401_responder)
    try:
        result = discover_cameras(_config_for(server.port), _credentials())
    finally:
        server.close()

    assert result.cameras == []
    assert result.ports_open == 1
    assert result.auth_failures == 1


def test_discover_stream_not_found() -> None:
    server = _FakeRtspServer(_auth_then_404_responder)
    try:
        result = discover_cameras(_config_for(server.port), _credentials())
    finally:
        server.close()

    assert result.cameras == []
    assert result.ports_open == 1
    assert result.auth_failures == 0


def test_discover_confirms_camera_with_basic_auth() -> None:
    server = _FakeRtspServer(_basic_good_responder)
    try:
        result = discover_cameras(_config_for(server.port), _credentials())
    finally:
        server.close()

    assert len(result.cameras) == 1
    assert result.cameras[0].rtsp_url == (
        f"rtsp://{USERNAME}:{PASSWORD}@127.0.0.1:{server.port}{STREAM_PATH}"
    )
    assert result.auth_failures == 0


def test_discover_prefers_digest_over_basic_when_both_offered() -> None:
    server = _FakeRtspServer(_basic_then_digest_responder)
    try:
        result = discover_cameras(_config_for(server.port), _credentials())
    finally:
        server.close()

    assert len(result.cameras) == 1
    assert result.auth_failures == 0


def test_discover_unknown_status_is_not_a_camera() -> None:
    server = _FakeRtspServer(_unknown_status_responder)
    try:
        result = discover_cameras(_config_for(server.port), _credentials())
    finally:
        server.close()

    assert result.cameras == []
    assert result.ports_open == 1
    assert result.auth_failures == 0


def test_discover_logs_progress_per_host(caplog) -> None:
    server = _FakeRtspServer(_digest_good_responder)
    try:
        with caplog.at_level("INFO", logger="lib.discovery"):
            discover_cameras(_config_for(server.port), _credentials())
    finally:
        server.close()

    messages = "\n".join(record.getMessage() for record in caplog.records)
    # The scope, the confirmation, and the summary appear at INFO level without
    # logging every dead address in a full /24 sweep.
    assert "probing :" in messages
    assert "127.0.0.1 CONFIRMED" in messages
    assert "1 confirmed" in messages


def test_discover_publishes_progress_states() -> None:
    progress = DiscoveryProgress()
    server = _FakeRtspServer(_digest_good_responder)
    try:
        discover_cameras(_config_for(server.port), _credentials(), progress=progress)
    finally:
        server.close()

    snap = progress.snapshot()
    # The sweep cleared its active flag in the finally, the host was confirmed,
    # and the counts reflect exactly one found camera.
    assert snap["active"] is False
    assert snap["states"]["127.0.0.1"] == HOST_FOUND
    assert snap["counts"][HOST_FOUND] == 1


def test_discover_progress_marks_failure_for_auth_reject() -> None:
    progress = DiscoveryProgress()
    server = _FakeRtspServer(_always_401_responder)
    try:
        discover_cameras(_config_for(server.port), _credentials(), progress=progress)
    finally:
        server.close()

    snap = progress.snapshot()
    assert snap["states"]["127.0.0.1"] == HOST_FAILED
    assert snap["counts"][HOST_FOUND] == 0


def test_build_rtsp_url_percent_encodes_credentials() -> None:
    # A Tapo password with URL-special characters must be encoded so ffmpeg gets
    # a well-formed URL (an unencoded one is what leaves a camera "connecting").
    url = build_rtsp_url(
        CameraCredentials("user", "p@ss:w/rd"), "192.168.1.20", 554, "/stream1"
    )
    assert url == "rtsp://user:p%40ss%3Aw%2Frd@192.168.1.20:554/stream1"


def test_redact_rtsp_url_masks_password() -> None:
    assert (
        redact_rtsp_url("rtsp://user:secret@10.0.0.1:554/stream1")
        == "rtsp://user:***@10.0.0.1:554/stream1"
    )


def test_discover_elapsed_uses_injected_clock() -> None:
    server = _FakeRtspServer(_digest_good_responder)
    ticks = iter([100.0, 100.5])
    try:
        result = discover_cameras(
            _config_for(server.port), _credentials(), clock=lambda: next(ticks)
        )
    finally:
        server.close()
    assert result.elapsed_seconds == 0.5
