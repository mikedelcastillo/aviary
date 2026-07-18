"""Proprietary Tapo camera control: the white-light flash + remote reboot.

Tapo's ONVIF surface cannot drive the cameras' built-in spotlights or reboot
them (probed 2026-07: GetNodes advertises no auxiliary commands — the firmware
ACKs and silently ignores SendAuxiliaryCommand — and imaging/deviceIO expose no
light or relay controls). This module talks to each camera's LOCAL HTTPS API
via ``pytapo`` instead, which authenticates as ``admin`` with the Tapo CLOUD
account password — NOT the camera-account credentials that drive RTSP/ONVIF.
The whole module is therefore DORMANT (every call a cheap no-op) until
``TAPO_CLOUD_PASSWORD`` is set, and everything here is best-effort: a camera
that rejects a call is logged and skipped, never fatal.

Three collaborators:

* :class:`TapoControl` — per-host pytapo client cache plus the flash and
  reboot primitives. Flash ownership is tracked per host: a manual flag for
  the ``/flash`` command plus a REFCOUNTED set of per-search tokens, so a
  search never turns off a lamp the user forced on, and overlapping searches
  (a replaced /find whose teardown runs late, or the console and Telegram
  finders running at once) share the lamp — it goes off only when the LAST
  owner leaves. Forcing a lamp at night makes the camera's frames go
  full-colour, which would fake a day transition to every IR consumer (the
  sleep tracker would log a false disturbance; auto-find could re-enable at
  night) — so the IR flag is frozen via ``ir_hold`` BEFORE the lamp-on command
  is issued, and released a few seconds AFTER the lamp goes off (the camera
  needs a moment to fall back into IR, and an instant release would leak a
  one-frame false "daylight" blip). Lamp operations are serialised per host,
  and pending releases are cancelled by a re-light, so hold/release can never
  interleave into a lost hold.
* :class:`RebootGuard` — decides when a camera stream is WEDGED (unhealthy for
  a sustained window despite the monitor's reconnect loop) and fires one
  best-effort reboot on a background thread. The camera loop drives it with
  plain ``healthy``/``unhealthy`` signals; policy lives here, not in the loop.
* :class:`FindFlash` — the lights hook :class:`lib.find.BirdFinder` drives:
  ``start()`` lights the cameras currently in IR for the search and returns a
  user-facing note, ``stop()`` restores them.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

LOGGER = logging.getLogger("lib.tapo")

# Minimum gap between reboots of the same camera, so a camera that stays broken
# (or takes a while to boot) is never power-cycled in a loop.
REBOOT_COOLDOWN_SECONDS = 600.0
# How long a stream must be continuously unhealthy before it counts as wedged.
# Well past the monitor's 60s max reconnect backoff, so several reconnect
# attempts have genuinely failed first.
DEFAULT_WEDGE_SECONDS = 180.0
# Keep the IR flag frozen this long after a lamp turns off: the camera takes a
# couple of frames to drop back into IR, and releasing instantly would stamp a
# one-frame false "daylight" transition into the sleep/auto-find state.
IR_RELEASE_DELAY_SECONDS = 5.0



def _default_connect(host: str, cloud_password: str):
    # Imported lazily so the server (and the test suite) never pays for — or
    # depends on — pytapo unless the cloud password is actually configured.
    from pytapo import Tapo

    return Tapo(host, "admin", cloud_password)


def _daemon_spawn(fn: Callable[[], None], name: str) -> None:
    threading.Thread(target=fn, name=name, daemon=True).start()


def match_camera_hosts(
    target: str,
    hosts: list[str],
    display: Callable[[str], str] = lambda name: name,
) -> list[str]:
    """Resolve a user-typed camera reference to hosts, for ``/flash on <camera>``.

    Accepts the raw IP ("192.168.1.19"), a last-octet shorthand (".19" / "19"),
    or a case-insensitive fragment of the friendly display name ("cockatiel").
    A NUMERIC target never falls through to name matching — "19" must address
    exactly the .19 camera, not every display name (or IP-derived fallback
    name) that happens to contain "19". Name fragments may match several
    cameras at once ("cage" addresses all the cage cameras).
    """
    wanted = target.strip().lower()
    if not wanted:
        return []
    numeric = wanted.replace(".", "").isdigit()
    matched: list[str] = []
    for host in hosts:
        octet = host.rsplit(".", 1)[-1]
        if wanted in (host, f".{octet}", octet):
            matched.append(host)
        elif not numeric and wanted in display(f"camera-{host}").lower():
            matched.append(host)
    return matched


class TapoControl:
    """Best-effort flash + reboot over the Tapo local HTTPS API.

    Lamp state is tracked locally (the camera is assumed to start un-forced,
    which is how the firmware boots); the tracked state is what ownership
    decisions read. ``lamp_state`` additionally asks the device, so ``/flash``
    can report truth even after a server restart lost the tracked state.

    Locking: ``_lock`` guards the maps and is never held across network I/O;
    a per-host operation lock serialises whole lamp operations so two racing
    commands can't commit state opposite to the physical outcome. The IR
    hold/release callbacks are invoked under ``_lock`` so a pending release
    can never destroy the hold a concurrent re-light just placed — they must
    stay cheap and never call back into this class (lib.ir's are both).
    """

    def __init__(
        self,
        cloud_password: str | None,
        *,
        connect: Callable[[str, str], object] | None = None,
        clock: Callable[[], float] = time.monotonic,
        ir_hold: Callable[[str], None] | None = None,
        ir_release: Callable[[str], None] | None = None,
        release_delay: float = IR_RELEASE_DELAY_SECONDS,
    ) -> None:
        self._password = (cloud_password or "").strip()
        self._connect = connect or _default_connect
        self._clock = clock
        self._ir_hold = ir_hold
        self._ir_release = ir_release
        self._release_delay = release_delay
        self._lock = threading.Lock()
        self._op_locks: dict[str, threading.Lock] = {}
        self._clients: dict[str, object] = {}
        self._lamp_on: dict[str, bool] = {}
        # Who wants the lamp on: the /flash command's flag, plus a REFCOUNTED
        # set of search tokens. A find's lamp goes off only when the last
        # owning search leaves; a manual off evicts every owner (the user's
        # word is final); a manual on outlives every search.
        self._manual: dict[str, bool] = {}
        self._find_owners: dict[str, set] = {}
        # host -> the pending IR-release timer, so a re-light can cancel it.
        self._release_timers: dict[str, threading.Timer] = {}
        self._last_reboot: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return bool(self._password)

    def _host_lock(self, host: str) -> threading.Lock:
        with self._lock:
            return self._op_locks.setdefault(host, threading.Lock())

    # -- client cache --------------------------------------------------------

    def _client(self, host: str):
        if not self.enabled:
            return None
        with self._lock:
            client = self._clients.get(host)
        if client is not None:
            return client
        try:
            # The pytapo handshake does network I/O, so it runs outside the
            # lock; two racing creators just cache one and drop the other.
            client = self._connect(host, self._password)
        except Exception as exc:
            LOGGER.warning("Tapo API connect to %s failed: %s", host, exc)
            return None
        with self._lock:
            return self._clients.setdefault(host, client)

    def _drop_client(self, host: str) -> None:
        # A failed call (or a reboot) likely killed the session; reconnect fresh
        # next time instead of retrying on a dead one.
        with self._lock:
            self._clients.pop(host, None)

    # -- IR hold/release ------------------------------------------------------

    def _hold_ir_locked(self, host: str) -> None:
        """Freeze the camera's IR flag. Caller holds ``_lock``."""
        timer = self._release_timers.pop(host, None)
        if timer is not None:
            timer.cancel()
        if self._ir_hold is None:
            return
        try:
            self._ir_hold(f"camera-{host}")
        except Exception:
            LOGGER.exception("IR hold failed for %s", host)

    def _hold_ir(self, host: str) -> None:
        with self._lock:
            self._hold_ir_locked(host)

    def _release_ir_later(self, host: str) -> None:
        """Schedule the IR-flag release for after the lamp has physically
        dimmed. The timer is registered under the lock and re-checks, under
        the lock, that it is still the CURRENT timer and the lamp is still
        off — so a re-light (which cancels + re-holds under the same lock)
        can never have its fresh hold destroyed by a stale timer."""
        if self._ir_release is None:
            return

        def release() -> None:
            with self._lock:
                if self._release_timers.get(host) is not timer:
                    return  # superseded or cancelled by a re-light
                self._release_timers.pop(host, None)
                if self._lamp_on.get(host):
                    return
                try:
                    self._ir_release(f"camera-{host}")
                except Exception:
                    LOGGER.exception("IR release failed for %s", host)

        timer = threading.Timer(self._release_delay, release)
        timer.daemon = True
        with self._lock:
            previous = self._release_timers.pop(host, None)
            if previous is not None:
                previous.cancel()
            self._release_timers[host] = timer
        timer.start()

    # -- flash ----------------------------------------------------------------

    def is_on(self, host: str) -> bool:
        with self._lock:
            return self._lamp_on.get(host, False)

    def _lamp_burning(self, client) -> bool | None:
        """Whether the manual lamp burn is physically running, or None if the
        camera can't say. ``rest_time > 0`` is the reliable marker (verified
        live on both models 2026-07-14): the C320WS leaves ``status`` at 1
        after its burn ends, but ``rest_time`` always drops to 0."""
        try:
            status = client.getWhitelampStatus()
        except Exception:
            return None
        if not isinstance(status, dict):
            return None
        try:
            return int(status.get("rest_time", 0) or 0) > 0
        except (TypeError, ValueError):
            return None

    def lamp_state(self, host: str) -> bool:
        """The lamp state as the DEVICE reports it, falling back to the tracked
        guess when the camera is unreachable. Used by ``/flash`` status and
        toggle so a restart that lost the tracked state (with a lamp still
        burning) reads — and toggles — truthfully."""
        client = self._client(host)
        if client is not None:
            burning = self._lamp_burning(client)
            if burning is not None:
                return burning
            try:
                state = client.getForceWhitelampState()
                if isinstance(state, bool):
                    return state
            except Exception as exc:
                LOGGER.debug("Whitelamp state read failed on %s: %s", host, exc)
        return self.is_on(host)

    def _set_lamp(self, host: str, on: bool) -> bool:
        """Drive the lamp; True when the camera verifiably (or plausibly) obeyed.

        Primary path: the manual whitelamp BURN (``reverseWhitelampStatus``) —
        the only call that physically fires the lamp while a camera sits in
        ``inf_night_vision`` (IR-only) mode, verified live on both the C216
        and the C320WS (whose floodlight API is unsupported). It is a toggle
        with a firmware countdown (C216 ~30 min, C320WS ~5 min — a free
        safety-net if the server dies mid-burn), so the current state is read
        first and the toggle only sent when it differs, then read back.

        Fallbacks for other models/firmware: the ``force_wtl_state`` flag
        (honoured in the full-colour night modes, inert in IR-only mode) and
        the floodlight op.
        """
        client = self._client(host)
        if client is None:
            return False
        burning = self._lamp_burning(client)
        if burning is not None:
            if burning == on:
                LOGGER.info("Flash already %s on %s", "ON" if on else "off", host)
                return True
            try:
                client.reverseWhitelampStatus()
                after = self._lamp_burning(client)
                if after is None or after == on:
                    LOGGER.info("Flash %s on %s (whitelamp burn)", "ON" if on else "off", host)
                    return True
                LOGGER.warning(
                    "Whitelamp toggle on %s did not take (still %s)", host, after
                )
                return False
            except Exception as exc:
                LOGGER.debug("Whitelamp toggle on %s failed (%s); trying force flag", host, exc)
        try:
            client.setForceWhitelampState(on)
            LOGGER.info("Flash %s on %s (whitelamp force flag)", "ON" if on else "off", host)
            return True
        except Exception as exc:
            LOGGER.debug("Whitelamp force %s on %s failed (%s); trying floodlight", on, host, exc)
        try:
            client.manualFloodlightOp(on)
            LOGGER.info("Flash %s on %s (floodlight)", "ON" if on else "off", host)
            return True
        except Exception as exc:
            LOGGER.warning("Flash %s failed on %s: %s", "on" if on else "off", host, exc)
            self._drop_client(host)
            return False

    def _lamp_on_locked_precheck(self, host: str) -> bool:
        """Hold the IR flag and cancel any pending release BEFORE the lamp-on
        command goes out — the lamp may light the scene mid-call, and the flag
        must already be frozen by then. Returns whether the lamp was already
        tracked on (for failure rollback)."""
        with self._lock:
            was_on = self._lamp_on.get(host, False)
            self._hold_ir_locked(host)
        return was_on

    def manual_set(self, host: str, on: bool) -> bool:
        """The ``/flash`` command's on/off. Manual ownership wins: a lamp forced
        on here is never turned off by a finishing search, and a manual OFF
        evicts every owning search too (the user's word is final)."""
        if not self.enabled:
            return False
        with self._host_lock(host):
            if on:
                was_on = self._lamp_on_locked_precheck(host)
                if not self._set_lamp(host, True):
                    if not was_on:
                        # Undo the early hold; the lamp never lit.
                        self._release_ir_later(host)
                    return False
                with self._lock:
                    self._lamp_on[host] = True
                    self._manual[host] = True
                return True
            if not self._set_lamp(host, False):
                return False
            with self._lock:
                self._lamp_on[host] = False
                self._manual[host] = False
                self._find_owners.pop(host, None)
            self._release_ir_later(host)
            return True

    def find_start(self, hosts: list[str], token: object) -> list[str]:
        """Light every host a search may share; return the ones it now co-owns.

        A host already lit by another search — a replaced /find whose teardown
        overlaps this start (in EITHER order), or the other interface's finder
        running concurrently — is JOINED, not stolen: this search's token is
        added to the owner set, so whichever search finishes first leaves the
        lamp burning for the survivors and only the last one out turns it off.
        Manually forced lamps are never touched.
        """
        lit: list[str] = []
        for host in hosts:
            with self._host_lock(host):
                with self._lock:
                    if self._manual.get(host):
                        continue  # the user's lamp; already lit, leave it alone
                    owners = self._find_owners.get(host)
                    if owners:
                        # Another search's lamp — already on, IR already held.
                        owners.add(token)
                        lit.append(host)
                        continue
                    if self._lamp_on.get(host):
                        continue  # lit but unowned (a failed restore); leave it
                self._lamp_on_locked_precheck(host)
                if not self._set_lamp(host, True):
                    # The lamp never lit (it was off — checked above), so undo
                    # the early hold.
                    self._release_ir_later(host)
                    continue
                with self._lock:
                    self._find_owners.setdefault(host, set()).add(token)
                    self._lamp_on[host] = True
                lit.append(host)
        return lit

    def find_stop(self, hosts: list[str], token: object) -> None:
        """Leave the owner set of every lamp this search lit; turn a lamp off
        only when this was its LAST owner and the user hasn't taken it over
        with ``/flash``."""
        for host in hosts:
            with self._host_lock(host):
                with self._lock:
                    owners = self._find_owners.get(host)
                    if owners is None or token not in owners:
                        continue  # evicted by /flash off, or a reboot's reset
                    owners.discard(token)
                    if owners or self._manual.get(host):
                        continue  # the lamp still has owners; leave it burning
                if self._set_lamp(host, False):
                    with self._lock:
                        self._lamp_on[host] = False
                    self._release_ir_later(host)
                else:
                    # The off command failed, so the lamp may physically still
                    # be on. The owner set is already empty (a /flash can
                    # retake it), but keep the lamp marked lit and its IR flag
                    # held — a frozen-True flag under a still-burning lamp is
                    # the correct reading.
                    LOGGER.warning("Find flash restore failed on %s; lamp may still be on", host)

    # -- reboot ----------------------------------------------------------------

    def reboot(self, host: str) -> bool:
        """One best-effort reboot, rate-limited per host.

        The cooldown stamp is written BEFORE the attempt so a camera that
        accepts the call but takes minutes to come back — or one that keeps
        failing — is never hammered with repeat reboots. A successful reboot
        also RESETS the host's lamp bookkeeping: the firmware boots with the
        lamp un-forced, and without the reset a flashed camera's IR flag
        would stay frozen forever (blinding the sleep tracker and auto-find).
        """
        if not self.enabled:
            return False
        # The same per-host serialisation as the lamp paths, so a reboot's
        # bookkeeping reset can't interleave with a /flash committing state.
        with self._host_lock(host):
            now = self._clock()
            with self._lock:
                last = self._last_reboot.get(host)
                if last is not None and now - last < REBOOT_COOLDOWN_SECONDS:
                    return False
                self._last_reboot[host] = now
            client = self._client(host)
            if client is None:
                return False
            try:
                client.reboot()
            except Exception as exc:
                LOGGER.warning("Reboot of %s failed: %s", host, exc)
                self._drop_client(host)
                return False
            LOGGER.info("Rebooted camera %s via Tapo API", host)
            # The session dies with the reboot; reconnect fresh next call.
            self._drop_client(host)
            with self._lock:
                self._lamp_on[host] = False
                self._manual[host] = False
                self._find_owners.pop(host, None)
            # Unfreeze the IR flag once the camera is back: while it boots no
            # frames arrive anyway, and the first decoded frame re-stamps truth.
            self._release_ir_later(host)
            return True


class RebootGuard:
    """Turns the camera loop's health signals into at-most-one reboot per wedge.

    The monitor thread calls ``healthy(host)`` on every good frame (and while
    deliberately paused) and ``unhealthy(host)`` each time a session fails.
    Once a host has been continuously unhealthy for ``wedge_seconds`` the guard
    fires ``reboot`` on a background thread — the pytapo call does network I/O
    and must never stall the capture loop's reconnect cadence. After any
    attempt the wedge window restarts, so the camera gets a full boot's worth
    of time before it can be judged wedged again. The supervisor calls
    ``forget`` when it retires a camera, so a stale window can't instantly
    reboot the same host if it is re-added hours later.
    """

    def __init__(
        self,
        reboot: Callable[[str], bool],
        *,
        wedge_seconds: float = DEFAULT_WEDGE_SECONDS,
        notify: Callable[[str, float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        spawn: Callable[[Callable[[], None], str], None] = _daemon_spawn,
    ) -> None:
        self._reboot = reboot
        self._wedge_seconds = wedge_seconds
        self._notify = notify
        self._clock = clock
        self._spawn = spawn
        self._lock = threading.Lock()
        self._down_since: dict[str, float] = {}
        self._in_flight: set[str] = set()

    def healthy(self, host: str) -> None:
        with self._lock:
            self._down_since.pop(host, None)

    def forget(self, host: str) -> None:
        """Drop a host's window entirely (its camera was stopped/retired)."""
        self.healthy(host)

    def unhealthy(self, host: str) -> None:
        now = self._clock()
        with self._lock:
            since = self._down_since.setdefault(host, now)
            if now - since < self._wedge_seconds or host in self._in_flight:
                return
            self._in_flight.add(host)
        downtime = now - since
        self._spawn(lambda: self._fire(host, downtime), f"tapo-reboot-{host}")

    def _fire(self, host: str, downtime: float) -> None:
        try:
            ok = False
            try:
                ok = self._reboot(host)
            except Exception:
                LOGGER.exception("Wedge reboot of %s failed", host)
            if ok and self._notify is not None:
                try:
                    self._notify(host, downtime)
                except Exception:
                    LOGGER.exception("Wedge reboot notification failed for %s", host)
        finally:
            with self._lock:
                self._in_flight.discard(host)
                # Restart the window: the camera needs a full boot before its
                # continued silence can mean "still wedged". Only while the
                # window still exists — a forget() (camera retired) or
                # healthy() that landed during the reboot call must stand, or
                # the stale stamp would instantly reboot the host if it is
                # re-added hours later.
                if host in self._down_since:
                    self._down_since[host] = self._clock()


class FindFlash:
    """Lights for one search: flash the cameras that are dark, restore after.

    Built fresh per search (like the PTZ patrol), carrying its own ownership
    token so an overlapping search's teardown can't douse it — lamps are
    co-owned and only the last search out turns them off. ``start``
    reads which ready cameras are in IR at that moment and lights those — a
    camera that goes dark mid-search is left alone, keeping the behaviour
    predictable. The lamp calls are sequential HTTPS requests (a few seconds
    worst case), paid once at the start of a search that runs minutes.
    """

    def __init__(
        self,
        control: TapoControl,
        ir_cameras: Callable[[], set[str]],
        hosts: Callable[[], set[str]],
        display: Callable[[str], str] = lambda name: name,
    ) -> None:
        self._control = control
        self._ir_cameras = ir_cameras
        self._hosts = hosts
        self._display = display
        self._token = object()
        self._lit: list[str] = []

    def start(self) -> str | None:
        ir = self._ir_cameras()
        candidates = [host for host in sorted(self._hosts()) if f"camera-{host}" in ir]
        if not candidates:
            return None
        self._lit = self._control.find_start(candidates, self._token)
        if not self._lit:
            return None
        names = ", ".join(self._display(f"camera-{host}") for host in self._lit)
        return f"💡 It's dark on {names} — spotlight on while I search."

    def stop(self) -> None:
        lit, self._lit = self._lit, []
        if lit:
            self._control.find_stop(lit, self._token)
