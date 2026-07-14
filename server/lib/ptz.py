"""Best-effort ONVIF pan/tilt control for the ``/find`` patrol.

The Tapo pan-tilt cameras (e.g. the C216) speak ONVIF on port 2020 at
``/onvif/service``. While ``/find`` hunts for a bird, it sweeps these cameras so
one tucked out of the current frame still gets found. Everything here is
*best-effort*: a camera that doesn't answer ONVIF, isn't a PTZ model, or rejects
a move is simply skipped — ``/find`` works fine without any movement.

We speak raw ONVIF SOAP over ``requests`` rather than pulling in a heavyweight
ONVIF/zeep stack, mirroring how :mod:`lib.discovery` speaks raw RTSP. The SOAP
envelope, the WS-Security ``UsernameToken`` (PasswordDigest) and the response
parsing are split into pure functions so they can be unit-tested without a
camera; only :func:`_soap_call` touches the network.

Auth uses the same Tapo camera account as RTSP (``TAPO_CREDENTIALS``).
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Callable

import requests

from lib.config import CameraCredentials


LOGGER = logging.getLogger("lib.ptz")

ONVIF_PORT = 2020
ONVIF_PATH = "/onvif/service"

_NS_DEVICE = "http://www.onvif.org/ver10/device/wsdl"
_NS_MEDIA = "http://www.onvif.org/ver10/media/wsdl"
_NS_PTZ = "http://www.onvif.org/ver20/ptz/wsdl"
_NS_SCHEMA = "http://www.onvif.org/ver10/schema"
_NS_WSSE = (
    "http://docs.oasis-open.org/wss/2004/01/"
    "oasis-200401-wss-wssecurity-secext-1.0.xsd"
)
_PW_DIGEST_TYPE = (
    "http://docs.oasis-open.org/wss/2004/01/"
    "oasis-200401-wss-username-token-profile-1.0#PasswordDigest"
)
_NONCE_ENCODING = (
    "http://docs.oasis-open.org/wss/2004/01/"
    "oasis-200401-wss-soap-message-security-1.0#Base64Binary"
)

# The patrol scans a grid of the camera's full range (pan x tilt) so it covers
# the Y axis too, not just left-right. Defaults: 4 columns x 3 rows = 12 cells.
DEFAULT_SCAN_COLS = 4
DEFAULT_SCAN_ROWS = 3

# Hold each grid cell STATIONARY this long before moving on, so the ~1fps
# detector gets clean frames instead of mid-pan blur.
DEFAULT_DWELL_SECONDS = 2.0
# AbsoluteMove returns before the camera physically arrives; allow this much
# travel time on top of the dwell before the next move is issued.
PATROL_SETTLE_SECONDS = 2.0


def grid_cells(cols: int, rows: int) -> list[tuple[float, float]]:
    """Cell-centre (pan, tilt) positions covering the normalised -1..1 range.

    Boustrophedon (snake) order — each row scanned in the opposite direction to
    the last — so the camera travels the shortest path between cells. Centres are
    inset from the edges (e.g. 4 cols -> -0.75, -0.25, 0.25, 0.75) so a cell looks
    at the middle of its area rather than the extreme limit.
    """
    cols = max(1, cols)
    rows = max(1, rows)
    cells: list[tuple[float, float]] = []
    for row in range(rows):
        tilt = -1.0 + (row + 0.5) * (2.0 / rows)
        columns = range(cols) if row % 2 == 0 else range(cols - 1, -1, -1)
        for col in columns:
            pan = -1.0 + (col + 0.5) * (2.0 / cols)
            cells.append((round(pan, 4), round(tilt, 4)))
    return cells


# -- pure SOAP construction -------------------------------------------------


def build_wss_header(username: str, password: str, *, nonce: bytes, created: str) -> str:
    """A WS-Security UsernameToken header with a PasswordDigest.

    digest = base64(sha1(nonce + created + password)). ``nonce`` and ``created``
    are injectable so the digest is deterministic under test.
    """
    digest = base64.b64encode(
        hashlib.sha1(nonce + created.encode() + password.encode()).digest()
    ).decode()
    return (
        f'<Security s:mustUnderstand="1" xmlns="{_NS_WSSE}">'
        f"<UsernameToken><Username>{username}</Username>"
        f'<Password Type="{_PW_DIGEST_TYPE}">{digest}</Password>'
        f'<Nonce EncodingType="{_NONCE_ENCODING}">'
        f"{base64.b64encode(nonce).decode()}</Nonce>"
        f"<Created>{created}</Created></UsernameToken></Security>"
    )


def build_envelope(header: str, body: str) -> str:
    """Wrap a SOAP header + body in a soap-envelope document."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">'
        f"<s:Header>{header}</s:Header><s:Body>{body}</s:Body></s:Envelope>"
    )


def get_capabilities_body() -> str:
    return f'<GetCapabilities xmlns="{_NS_DEVICE}"><Category>All</Category></GetCapabilities>'


def get_profiles_body() -> str:
    return f'<GetProfiles xmlns="{_NS_MEDIA}"/>'


def continuous_move_body(token: str, pan: float, tilt: float) -> str:
    return (
        f'<ContinuousMove xmlns="{_NS_PTZ}"><ProfileToken>{token}</ProfileToken>'
        f'<Velocity><PanTilt x="{pan}" y="{tilt}" xmlns="{_NS_SCHEMA}"/></Velocity>'
        f"</ContinuousMove>"
    )


def stop_body(token: str) -> str:
    return (
        f'<Stop xmlns="{_NS_PTZ}"><ProfileToken>{token}</ProfileToken>'
        f"<PanTilt>true</PanTilt><Zoom>false</Zoom></Stop>"
    )


def get_status_body(token: str) -> str:
    return f'<GetStatus xmlns="{_NS_PTZ}"><ProfileToken>{token}</ProfileToken></GetStatus>'


def absolute_move_body(token: str, pan: float, tilt: float) -> str:
    # Pan/tilt only — the Tapo pan-tilt models expose no zoom axis, so omit it.
    return (
        f'<AbsoluteMove xmlns="{_NS_PTZ}"><ProfileToken>{token}</ProfileToken>'
        f'<Position><PanTilt x="{pan}" y="{tilt}" xmlns="{_NS_SCHEMA}"/></Position>'
        f"</AbsoluteMove>"
    )


def parse_pantilt_position(response_text: str) -> tuple[float, float] | None:
    """Pull (pan, tilt) out of a GetStatus ``<PanTilt x=.. y=..>`` (any namespace)."""
    match = re.search(
        r'PanTilt[^>]*\bx="([^"]+)"[^>]*\by="([^"]+)"', response_text
    )
    if not match:
        return None
    try:
        return float(match.group(1)), float(match.group(2))
    except ValueError:
        return None


def goto_home_body(token: str) -> str:
    return (
        f'<GotoHomePosition xmlns="{_NS_PTZ}"><ProfileToken>{token}</ProfileToken>'
        f"</GotoHomePosition>"
    )


def get_presets_body(token: str) -> str:
    return f'<GetPresets xmlns="{_NS_PTZ}"><ProfileToken>{token}</ProfileToken></GetPresets>'


def goto_preset_body(profile_token: str, preset_token: str) -> str:
    return (
        f'<GotoPreset xmlns="{_NS_PTZ}"><ProfileToken>{profile_token}</ProfileToken>'
        f"<PresetToken>{preset_token}</PresetToken></GotoPreset>"
    )


def parse_first_preset_token(response_text: str) -> str | None:
    """Token of the FIRST saved preset — the Tapo cameras' user-set "home"."""
    match = re.search(r'<\w*:?Preset\b[^>]*token="([^"]+)"', response_text)
    return match.group(1) if match else None


def capabilities_have_ptz(response_text: str) -> bool:
    """True if a GetCapabilities response advertises a PTZ service."""
    # The PTZ capability block carries an <XAddr> with a /ptz or /onvif URL; a
    # bare <...PTZ> element is enough to know the model is pan-tilt.
    return bool(re.search(r"<\w*:?PTZ[ >]", response_text))


def parse_profile_token(response_text: str) -> str | None:
    """Pull the first media-profile token out of a GetProfiles response."""
    match = re.search(r'token="([^"]+)"', response_text)
    return match.group(1) if match else None


def _utc_now_iso(clock: Callable[[], datetime]) -> str:
    return clock().strftime("%Y-%m-%dT%H:%M:%SZ")


# -- network ----------------------------------------------------------------


def _soap_call(
    host: str,
    body: str,
    action: str,
    credentials: CameraCredentials,
    *,
    timeout: float,
    now: Callable[[], datetime],
) -> tuple[int | None, str]:
    """POST one SOAP request to ``host``'s ONVIF endpoint; return (status, text).

    Returns ``(None, "")`` on any socket/transport error so callers stay
    best-effort. Auth is freshly stamped per call (nonce + UTC created).
    """
    header = build_wss_header(
        credentials.username,
        credentials.password,
        nonce=os.urandom(16),
        created=_utc_now_iso(now),
    )
    envelope = build_envelope(header, body)
    url = f"http://{host}:{ONVIF_PORT}{ONVIF_PATH}"
    try:
        response = requests.post(
            url,
            data=envelope.encode("utf-8"),
            headers={
                "Content-Type": f'application/soap+xml; charset=utf-8; action="{action}"'
            },
            timeout=timeout,
        )
        return response.status_code, response.text
    except requests.RequestException as exc:
        LOGGER.debug("ONVIF call %s to %s failed: %s", action, host, exc)
        return None, ""


class OnvifPtzCamera:
    """A single pan-tilt camera addressed over ONVIF. Every method best-effort."""

    def __init__(
        self,
        host: str,
        credentials: CameraCredentials,
        *,
        timeout: float = 6.0,
        attempts: int = 3,
        retry_delay: float = 0.25,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.host = host
        self._credentials = credentials
        self._timeout = timeout
        self._attempts = max(1, attempts)
        self._retry_delay = max(0.0, retry_delay)
        self._now = now
        self._token: str | None = None
        self._token_resolved = False

    def _call(self, body: str, action: str) -> tuple[int | None, str]:
        result: tuple[int | None, str] = (None, "")
        for attempt in range(1, self._attempts + 1):
            result = _soap_call(
                self.host,
                body,
                action,
                self._credentials,
                timeout=self._timeout,
                now=self._now,
            )
            code, _ = result
            if code is not None and 200 <= code < 500:
                return result
            if attempt < self._attempts and self._retry_delay > 0:
                time.sleep(self._retry_delay)
        return result

    def profile_token(self) -> str | None:
        """The media profile token, fetched once and cached."""
        if not self._token_resolved:
            _, text = self._call(get_profiles_body(), f"{_NS_MEDIA}/GetProfiles")
            self._token = parse_profile_token(text)
            # Do not permanently cache a missing token: on lossy WiFi a single
            # dropped GetProfiles would otherwise disable PTZ for the whole run.
            self._token_resolved = self._token is not None
        return self._token

    def has_ptz(self) -> bool:
        code, text = self._call(get_capabilities_body(), f"{_NS_DEVICE}/GetCapabilities")
        return code == 200 and capabilities_have_ptz(text)

    def move(self, pan: float, tilt: float = 0.0) -> bool:
        token = self.profile_token()
        if token is None:
            return False
        code, _ = self._call(
            continuous_move_body(token, pan, tilt), f"{_NS_PTZ}/ContinuousMove"
        )
        return code == 200

    def stop(self) -> bool:
        token = self.profile_token()  # resolve like every other call, not the raw cache
        if token is None:
            return False
        code, _ = self._call(stop_body(token), f"{_NS_PTZ}/Stop")
        return code == 200

    def get_position(self) -> tuple[float, float] | None:
        """Current (pan, tilt), so we can return here after a patrol."""
        token = self.profile_token()
        if token is None:
            return None
        code, text = self._call(get_status_body(token), f"{_NS_PTZ}/GetStatus")
        if code != 200:
            return None
        return parse_pantilt_position(text)

    def absolute_move(self, pan: float, tilt: float) -> bool:
        token = self.profile_token()
        if token is None:
            return False
        code, _ = self._call(
            absolute_move_body(token, pan, tilt), f"{_NS_PTZ}/AbsoluteMove"
        )
        return code == 200

    def first_preset_token(self) -> str | None:
        """The first saved preset's token (the camera's user-set home), if any."""
        token = self.profile_token()
        if token is None:
            return None
        code, text = self._call(get_presets_body(token), f"{_NS_PTZ}/GetPresets")
        if code != 200:
            return None
        return parse_first_preset_token(text)

    def goto_preset(self, preset_token: str) -> bool:
        token = self.profile_token()
        if token is None:
            return False
        code, _ = self._call(
            goto_preset_body(token, preset_token), f"{_NS_PTZ}/GotoPreset"
        )
        return code == 200

    def goto_home(self) -> bool:
        token = self.profile_token()  # resolve like every other call, not the raw cache
        if token is None:
            return False
        code, _ = self._call(goto_home_body(token), f"{_NS_PTZ}/GotoHomePosition")
        return code == 200


class PtzManager:
    """Discovers and caches which hosts are PTZ-capable; builds patrols.

    Capability probing is cached per host (a host is a stable per-camera
    identity), so repeated ``/find`` calls don't re-probe ONVIF every time.
    """

    def __init__(
        self,
        credentials: CameraCredentials,
        *,
        timeout: float = 6.0,
        attempts: int = 3,
        retry_delay: float = 0.25,
        scan_cols: int = DEFAULT_SCAN_COLS,
        scan_rows: int = DEFAULT_SCAN_ROWS,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._credentials = credentials
        self._timeout = timeout
        self._attempts = max(1, attempts)
        self._retry_delay = max(0.0, retry_delay)
        self._scan_cols = scan_cols
        self._scan_rows = scan_rows
        self._now = now
        self._lock = threading.Lock()
        self._cache: dict[str, OnvifPtzCamera | None] = {}

    def camera_for(self, host: str) -> OnvifPtzCamera | None:
        """Return a PTZ camera for ``host`` if it supports PTZ, else ``None``."""
        with self._lock:
            if host in self._cache:
                return self._cache[host]
        camera = OnvifPtzCamera(
            host,
            self._credentials,
            timeout=self._timeout,
            attempts=self._attempts,
            retry_delay=self._retry_delay,
            now=self._now,
        )
        result = camera if camera.has_ptz() else None
        with self._lock:
            # Positive capability is stable and worth caching. A negative probe
            # can just be packet loss or a camera still booting, so do not cache
            # it unless a test/operator explicitly pre-seeded the cache.
            if result is not None:
                self._cache[host] = result
        if result is not None:
            LOGGER.info("PTZ available on %s", host)
        return result

    def build_patrol(self, hosts) -> "OnvifPatrol | None":
        """A patrol over whichever of ``hosts`` are PTZ-capable, or ``None``."""
        cameras = [cam for host in sorted(hosts) if (cam := self.camera_for(host))]
        if not cameras:
            return None
        return OnvifPatrol(cameras, cols=self._scan_cols, rows=self._scan_rows)

    def go_home(self, hosts) -> tuple[int, int, int]:
        """Send every PTZ camera among ``hosts`` to its first saved preset.

        Returns ``(homed, total_ptz, without_preset)`` — where ``without_preset``
        counts PTZ cameras that had no saved home preset to aim at, so ``/home``
        can tell the user to set one in the Tapo app. Used by ``/home`` and run
        after discovery so the cameras always face their saved viewpoint.
        """
        cameras = [cam for host in sorted(hosts) if (cam := self.camera_for(host))]
        homed = 0
        without_preset = 0
        for camera in cameras:
            try:
                preset = camera.first_preset_token()
            except Exception:
                LOGGER.exception("PTZ get-presets failed on %s", camera.host)
                preset = None
            if preset is None:
                without_preset += 1
                LOGGER.warning("PTZ %s has no saved preset to home to", camera.host)
                continue
            try:
                if camera.goto_preset(preset):
                    homed += 1
                    LOGGER.info("PTZ %s sent to home preset %s", camera.host, preset)
            except Exception:
                LOGGER.exception("PTZ home failed on %s", camera.host)
        return homed, len(cameras), without_preset


class OnvifPatrol:
    """Scans a set of PTZ cameras through a grid of their range while a search runs.

    Implements the ``PtzPatrol`` protocol (start/step/stop) that
    :class:`lib.find.BirdFinder` drives. ``start`` records each camera's saved
    "home" preset (the user's first Tapo preset) and its current facing. ``step``
    moves every camera to the NEXT grid cell (an AbsoluteMove), so over a sweep it
    covers the full pan AND tilt range — not just left-right; each cell is then
    HELD (travel settle + a stationary dwell) so the detector gets clean frames,
    and step() calls inside that hold window are no-ops. ``stop`` halts and
    returns every camera HOME — preferring the user's saved preset (so it lands
    exactly where they parked it, never on a wall), falling back to the captured
    position then the ONVIF home command.
    """

    def __init__(
        self,
        cameras: list[OnvifPtzCamera],
        *,
        cols: int = DEFAULT_SCAN_COLS,
        rows: int = DEFAULT_SCAN_ROWS,
        dwell_seconds: float = DEFAULT_DWELL_SECONDS,
        settle_seconds: float = PATROL_SETTLE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cameras = cameras
        self._cells = grid_cells(cols, rows)
        self._steps = 0
        self._dwell_seconds = dwell_seconds
        self._settle_seconds = settle_seconds
        self._clock = clock
        # When the last AbsoluteMove was issued; None = no move yet this patrol.
        self._last_move_at: float | None = None
        # camera.host -> (pan, tilt) captured at start(), for fallback restore.
        self._home: dict[str, tuple[float, float]] = {}
        # camera.host -> the user's saved "home" preset token (preferred restore).
        self._home_preset: dict[str, str] = {}

    @property
    def cameras(self) -> list[OnvifPtzCamera]:
        return self._cameras

    @property
    def cells(self) -> list[tuple[float, float]]:
        return self._cells

    def start(self) -> None:
        # Capture the user's saved home preset (preferred) and the current facing
        # (fallback) so stop() can put each camera back. Resetting the counter
        # makes a fresh patrol always start the same way.
        self._steps = 0
        self._last_move_at = None
        self._home = {}
        self._home_preset = {}
        for camera in self._cameras:
            try:
                preset = camera.first_preset_token()
            except Exception:
                LOGGER.exception("PTZ get-presets failed on %s", camera.host)
                preset = None
            if preset is not None:
                self._home_preset[camera.host] = preset
            try:
                position = camera.get_position()
            except Exception:
                LOGGER.exception("PTZ get_position failed on %s", camera.host)
                position = None
            if position is not None:
                self._home[camera.host] = position

    def step(self) -> None:
        # Move every camera to the next grid cell (absolute pan+tilt) — but only
        # once the previous cell has had its full stop: settle (travel time for
        # the move to physically finish) + dwell (stationary, so the detector
        # sees clean frames, not mid-pan blur). The search loop calls step()
        # every poll (~2s); calls that land inside the hold window do nothing.
        if self._last_move_at is not None:
            held = self._clock() - self._last_move_at
            if held < self._settle_seconds + self._dwell_seconds:
                return
        pan, tilt = self._cells[self._steps % len(self._cells)]
        for camera in self._cameras:
            try:
                camera.absolute_move(pan, tilt)
            except Exception:
                LOGGER.exception("PTZ move failed on %s", camera.host)
        self._steps += 1
        self._last_move_at = self._clock()

    def stop(self) -> None:
        for camera in self._cameras:
            try:
                camera.stop()
                # ALWAYS prefer the user's saved home preset (lands exactly where
                # they parked it). Use the token captured at start, but if that
                # capture missed (a transient GetPresets failure), fetch it fresh
                # NOW — otherwise we'd restore a stale captured position that may
                # itself be a wall. Fall back to captured position, then the
                # ONVIF home command, only when there's genuinely no preset.
                preset = self._home_preset.get(camera.host)
                if preset is None:
                    preset = camera.first_preset_token()
                if preset is not None and camera.goto_preset(preset):
                    LOGGER.info("PTZ %s returned to home preset %s", camera.host, preset)
                    continue
                home = self._home.get(camera.host)
                if home is not None:
                    camera.absolute_move(home[0], home[1])
                    LOGGER.info("PTZ %s restored to captured position (no preset)", camera.host)
                else:
                    camera.goto_home()
                    LOGGER.info("PTZ %s sent to ONVIF home (no preset/position)", camera.host)
            except Exception:
                LOGGER.exception("PTZ stop/restore failed on %s", camera.host)
