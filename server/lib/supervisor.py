"""Runtime camera supervisor: discovery -> live capture threads.

Cameras are no longer a static config list; they are discovered on the LAN and
started while the app is already running (the ``/discover`` Telegram command, and
an initial sweep at boot). This module owns that dynamic lifecycle:

  * it runs :func:`lib.discovery.discover_cameras`,
  * for each confirmed camera it has not already started, it spins up a daemon
    thread running :func:`lib.camera.monitor_camera`, and
  * it registers a :class:`CameraStats` for that camera into the dict the
    dashboard and ``/status`` read from.

Identity is keyed on the camera's IP (``host``): DHCP leases are stable in
practice, so the same physical camera keeps the same name (``camera-<host>``)
across rediscovery and a second ``/discover`` won't double-start it.

Thread-safety matters here: the shared ``stats`` dict is mutated from this
supervisor (any thread that calls ``/discover``) while the dashboard render
thread iterates it. We only ever touch that dict under ``stats_lock`` so the
dashboard never sees a half-inserted entry or a "dict changed size during
iteration" error. A second internal lock (``_threads_lock``) makes the dedup
check-then-start in ``start_camera`` atomic, so two concurrent ``/discover``
commands can't both decide the same host is new and double-start it. The slow
network scan itself runs outside that lock, so a second command isn't blocked
behind an in-flight sweep.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from lib.alerts import AlertDispatcher, AlertState
from lib.camera import monitor_camera
from lib.config import AppConfig, CameraConfig
from lib.control import RuntimeControl
from lib.ir import IRState
from lib.detector import ObjectDetector
from lib.discovery import (
    DiscoveryProgress,
    DiscoveryResult,
    HOST_FAILED,
    HOST_FOUND,
    HOST_PENDING,
    HOST_TESTING,
    build_rtsp_url,
    discover_cameras,
    redact_rtsp_url,
)
from lib.detection_log import DetectionLogger
from lib.netid import format_mac, mac_for_ip, normalize_mac
from lib.objects import ObjectRegistry
from lib.quality import StreamQualityController
from lib.stats import CameraStats
from lib.watchlist import CameraRegistry


LOGGER = logging.getLogger("lib.supervisor")

# A camera is retired only after this many CONSECUTIVE sweeps in which it was
# active but not reconfirmed. A single dropped RTSP DESCRIBE (or a brief LAN
# blip) must never tear down a healthy, streaming camera. With the 10-minute
# auto-sweep this is a ~30-minute grace before a genuinely-gone camera is dropped.
RETIRE_AFTER_MISSES = 3


@dataclass
class DiscoveryApplied:
    """Outcome of a single ``discover_and_apply`` run.

    Distinguishes what the scan *saw* (``result``) from what the supervisor
    *did* (``added`` / ``already_active``) so the ``/discover`` reply can report
    both "found N cameras" and "started M new ones".
    """

    result: DiscoveryResult
    added: list[str]        # names of cameras newly started this run
    already_active: int     # confirmed cameras skipped as already running
    removed: list[str] = field(default_factory=list)  # retired because rediscovery missed them
    # Confirmed cameras NOT started because they are off the watchlist, as
    # display strings ("<ip> (<MAC>)"). Empty when the watchlist isn't enforcing.
    blocked: list[str] = field(default_factory=list)
    # host -> normalized MAC for every confirmed camera whose MAC resolved,
    # so reports can show hardware identity next to the IP-derived name.
    macs: dict[str, str] = field(default_factory=dict)


@dataclass
class _CameraRuntime:
    thread: threading.Thread
    stop_event: threading.Event


class _CameraStopEvent:
    """Per-camera stop signal that also observes server shutdown."""

    def __init__(self, global_stop: threading.Event, local_stop: threading.Event) -> None:
        self._global_stop = global_stop
        self._local_stop = local_stop

    def is_set(self) -> bool:
        return self._global_stop.is_set() or self._local_stop.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        if self.is_set():
            return True
        if timeout is None:
            while not self.is_set():
                self._local_stop.wait(0.2)
            return True
        deadline = time.monotonic() + max(0.0, timeout)
        while not self.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self.is_set()
            self._local_stop.wait(min(0.2, remaining))
        return True


def format_discovery_report(applied: DiscoveryApplied) -> str:
    """Render the final ``/discover`` summary: a one-line headline of what the
    scan found, then a short list of what actually changed on the cameras.

    Grouped to be skimmable — the scan stats on top, the camera changes (started
    / already running / retired) as a bulleted list below, and any credential
    problem called out last so it isn't lost in the noise.
    """
    result = applied.result
    cameras = len(result.cameras)

    def with_mac(name: str) -> str:
        # Names are "camera-<host>"; show the hardware identity alongside when
        # the sweep resolved it.
        mac = applied.macs.get(name.removeprefix("camera-"))
        return f"{name} (`{format_mac(mac)}`)" if mac else name

    lines = [
        f"✅ **Discovery complete** — scanned {result.hosts_scanned} hosts "
        f"in {result.elapsed_seconds:.1f}s",
        f"Found {cameras} camera(s) · {result.ports_open} host(s) with port :554 open",
        "",
        "📷 **Cameras**",
    ]
    if applied.added:
        lines.append("  🟢 Started: " + ", ".join(with_mac(name) for name in applied.added))
    else:
        lines.append("  • No new cameras started.")
    if applied.already_active:
        lines.append(f"  ✔️ {applied.already_active} already running")
    if applied.removed:
        lines.append("  🔴 Stopped stale: " + ", ".join(applied.removed))
    if applied.blocked:
        lines.append(
            "  ⚪ Not on the watchlist (not streamed): " + ", ".join(applied.blocked)
        )
    if result.auth_failures:
        # Surface bad creds explicitly: a camera that answers :554 but rejects
        # the password is almost always a TAPO_CREDENTIALS mismatch, which is
        # otherwise invisible (it just never shows up as a camera).
        lines.append("")
        lines.append(
            f"⚠️ {result.auth_failures} host(s) rejected the credentials "
            "— check TAPO_CREDENTIALS"
        )
    return "\n".join(lines)


def _progress_bar(fraction: float, width: int = 16) -> str:
    """A filled/empty block bar, matching the /machine telemetry frame."""
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


def format_discovery_progress(progress: dict) -> str:
    """Render a live /discover frame for Telegram: a header, a progress bar, and
    a per-state breakdown — the same visual language as the /machine dashboard."""
    counts = progress.get("counts", {})
    order = list(progress.get("order", []))
    states = dict(progress.get("states", {}))
    network = progress.get("network") or ""
    total = len(order)
    testing = counts.get(HOST_TESTING, 0)
    found = counts.get(HOST_FOUND, 0)
    failed = counts.get(HOST_FAILED, 0)
    checked = found + failed

    where = f" — {network}0/24" if network else ""
    bar = _progress_bar(checked / total if total else 0.0)
    lines = [
        f"🔍 **Discovering cameras**{where}",
        f"`[{bar}]` {checked}/{total} hosts",
        "",
        f"testing {testing} · found {found} · failed {failed}",
    ]
    found_hosts = [host for host in order if states.get(host) == HOST_FOUND]
    if found_hosts:
        lines.append("")
        lines.append("**Found so far**")
        lines.extend(f"  🟢 {host}" for host in found_hosts)
    return "\n".join(lines)


class CameraSupervisor:
    """Owns the live set of camera capture threads and their stats.

    The supervisor is the single writer of the shared ``stats`` dict and the
    owner of the monitor threads. It is created once in ``main`` and then driven
    by ``discover_and_apply`` (initial sweep + every ``/discover``).
    """

    def __init__(
        self,
        app_config: AppConfig,
        detector: ObjectDetector,
        alert_state: AlertState,
        dispatcher: AlertDispatcher,
        registry: ObjectRegistry,
        stats: dict[str, CameraStats],
        stats_lock: threading.Lock,
        stop_event: threading.Event,
        progress: DiscoveryProgress | None = None,
        control: RuntimeControl | None = None,
        ir_state: IRState | None = None,
        quality: StreamQualityController | None = None,
        detection_logger: DetectionLogger | None = None,
        rgb_reaction: Callable[[str, list, bool], None] | None = None,
        watchlist: CameraRegistry | None = None,
        mac_resolver: Callable[[str], str | None] = mac_for_ip,
        wedge_guard=None,
    ) -> None:
        self._app_config = app_config
        self._detector = detector
        self._alert_state = alert_state
        self._dispatcher = dispatcher
        self._registry = registry
        self._stats = stats
        self._stats_lock = stats_lock
        self._stop_event = stop_event
        self._ir_state = ir_state
        self._quality = quality
        self._detection_logger = detection_logger
        # Optional RGB status-display hook, threaded down to every monitor thread.
        self._rgb_reaction = rgb_reaction
        # Optional reboot-on-wedge guard (lib.tapo.RebootGuard), threaded down to
        # every monitor thread so a wedged stream can power-cycle its camera.
        self._wedge_guard = wedge_guard
        # Shared privacy/pause state. Passed to every monitor thread so a pause
        # stops all cameras consuming their streams at once.
        self._control = control
        # MAC-keyed camera cache + allowlist. When present, every confirmed
        # camera is recorded, and only watchlist-permitted cameras stream.
        self._watchlist = watchlist
        # Injected for tests; reads the kernel ARP table by default (fresh for
        # every confirmed host — the RTSP probe just talked to it).
        self._mac_resolver = mac_resolver
        # Live per-host sweep state for the dashboard's discovery grid. Shared
        # with the Dashboard so the camera band can switch to "discover mode"
        # while a scan runs. Optional: discovery works fine without it.
        self._progress = progress

        # host -> monitor thread + per-camera stop signal. Used both for dedup
        # and to retire stale IPs after rediscovery without stopping the server.
        self._threads: dict[str, _CameraRuntime] = {}
        # Serialises every mutation of (and read of) ``_threads`` so the dedup
        # check-then-start in ``start_camera`` is atomic. An RLock (not a plain
        # Lock) because ``discover_and_apply`` may already hold it when it calls
        # ``start_camera`` re-entrantly.
        self._threads_lock = threading.RLock()
        # host -> consecutive sweeps it was active but unconfirmed. Drives the
        # miss-grace before a camera is retired (see RETIRE_AFTER_MISSES).
        self._misses: dict[str, int] = {}
        # Only one discovery sweep may own the shared DiscoveryProgress sink at
        # a time. Initial discovery, auto-discovery, natural language, and
        # /discover all come through here.
        self._discovery_lock = threading.Lock()

    def start_camera(self, camera: CameraConfig) -> bool:
        """Start a monitor thread for ``camera`` unless its host is already live.

        Returns True if a new thread was started, False if the host was already
        being monitored (dedup by ``camera.host``). The whole check-then-start is
        atomic under ``_threads_lock`` so two callers can't both decide the same
        host is new and double-start it; the stats entry is inserted under
        ``stats_lock`` so the dashboard never observes a partial dict.
        """
        host = camera.host
        with self._threads_lock:
            if host in self._threads:
                return False
            if self._quality is not None:
                self._quality.register(host, camera.rtsp_url)

            camera_stats = CameraStats(
                camera.name,
                camera.sample_fps,
                self._registry,
                filter_objects=self._app_config.filter.objects,
            )
            # Publish the stats entry before the thread starts so the dashboard
            # sees the camera as soon as it appears (in "connecting" state).
            with self._stats_lock:
                self._stats[camera.name] = camera_stats

            camera_stop = threading.Event()
            thread = threading.Thread(
                target=monitor_camera,
                args=(
                    camera,
                    self._detector,
                    self._alert_state,
                    self._dispatcher,
                    camera_stats,
                    _CameraStopEvent(self._stop_event, camera_stop),
                    self._control,
                    self._ir_state,
                    self._quality,
                    self._detection_logger,
                    self._rgb_reaction,
                    self._wedge_guard,
                ),
                name=f"camera-{host}",
                daemon=True,
            )
            self._threads[host] = _CameraRuntime(thread=thread, stop_event=camera_stop)
            thread.start()
        LOGGER.info("Started camera %s -> %s", camera.name, redact_rtsp_url(camera.rtsp_url))
        return True

    def _stop_camera(self, host: str) -> str | None:
        with self._threads_lock:
            runtime = self._threads.pop(host, None)
            if runtime is None:
                return None
            runtime.stop_event.set()
            if self._quality is not None:
                self._quality.unregister(host)
            name = f"camera-{host}"
            with self._stats_lock:
                self._stats.pop(name, None)
        # Drop the retired camera's IR vote too, mirroring the pause path
        # (lib.camera). A lingering stale vote would otherwise wedge all_ir().
        if self._ir_state is not None:
            self._ir_state.forget(name)
        # And its wedge window: a stale unhealthy stamp would count the hours a
        # camera spent deliberately retired as downtime and reboot it the
        # moment it is re-added.
        if self._wedge_guard is not None:
            try:
                self._wedge_guard.forget(host)
            except Exception:
                LOGGER.exception("Wedge guard forget failed for %s", host)
        LOGGER.info("Stopped camera %s", name)
        return name

    def active_hosts(self) -> set[str]:
        """Hosts currently being monitored."""
        with self._threads_lock:
            return set(self._threads)

    def _resolve_mac(self, host: str) -> str | None:
        try:
            return self._mac_resolver(host)
        except Exception:
            LOGGER.exception("MAC resolution failed for %s", host)
            return None

    def start_host(self, host: str) -> bool:
        """Start monitoring ``host`` right now (the ``/watchlist allow`` path).

        The monitor thread's reconnect loop keeps retrying while the camera is
        offline, so this is safe to call for a cached camera that isn't up yet
        — it begins streaming the moment the camera answers.
        """
        rtsp_url = build_rtsp_url(
            self._app_config.credentials,
            host,
            self._app_config.discovery.rtsp_port,
            self._app_config.discovery.stream_path,
        )
        return self.start_camera(
            CameraConfig(name=f"camera-{host}", enabled=True, rtsp_url=rtsp_url, host=host)
        )

    def stop_host(self, host: str) -> str | None:
        """Stop monitoring ``host`` right now (the ``/watchlist remove`` path).

        Returns the stopped camera's name, or None if it wasn't active. The
        miss counter is cleared so a later re-allow starts with a clean slate.
        """
        with self._threads_lock:
            self._misses.pop(host, None)
        return self._stop_camera(host)

    # -- watchlist operations (the /watchlist command) -----------------------

    def _enforce_on_active(self) -> list[str]:
        """Stop active cameras whose cached MAC is off the (enforcing) list.

        Called when an ``allow`` turns enforcement on: cameras that were
        streaming under the empty-list-permits-all default must stop NOW, not
        on the next sweep. Hosts with no cached MAC are left for the sweep,
        which resolves and stops them there.
        """
        if self._watchlist is None or not self._watchlist.enforcing():
            return []
        stopped: list[str] = []
        for host in sorted(self.active_hosts()):
            mac = self._watchlist.mac_for_host(host)
            if mac is not None and not self._watchlist.permits(mac):
                name = self.stop_host(host)
                if name is not None:
                    stopped.append(name)
        return stopped

    def allow_camera(self, mac_raw: str) -> str:
        """Add a MAC to the watchlist and start its stream immediately when the
        camera's IP is already known from a past sighting."""
        if self._watchlist is None:
            return "Watchlist is not available."
        try:
            mac = normalize_mac(mac_raw)
        except ValueError:
            return f"That doesn't look like a MAC address: {mac_raw}"
        newly_added = self._watchlist.allow(mac)
        label = format_mac(mac)
        prefix = "🟢 Added" if newly_added else "✔️ Already on the watchlist:"
        # The list may just have flipped from "empty = allow all" to enforcing;
        # cameras streaming on that default lose their stream immediately.
        stopped = self._enforce_on_active()
        suffix = (
            "\n🔴 Stopped (not on the watchlist): " + ", ".join(stopped) if stopped else ""
        )
        host = self._watchlist.ip_for_mac(mac)
        if host is None:
            return (
                f"{prefix} `{label}`. That camera hasn't been seen on the network "
                f"yet — it will start streaming as soon as discovery finds it.{suffix}"
            )
        if host in self.active_hosts():
            return f"{prefix} `{label}` — camera-{host} is already streaming.{suffix}"
        live_mac = self._resolve_mac(host)
        if live_mac is not None and live_mac != mac:
            # DHCP moved the address since the last sighting; starting it would
            # stream a DIFFERENT device under this camera's identity. The next
            # sweep re-binds MAC -> IP and starts it at its real address.
            return (
                f"{prefix} `{label}`. Its last-known address {host} now belongs "
                f"to a different device — it will start streaming when discovery "
                f"finds it again.{suffix}"
            )
        if self.start_host(host):
            return f"{prefix} `{label}` — starting the stream from {host} now.{suffix}"
        return f"{prefix} `{label}`.{suffix}"

    def remove_camera(self, mac_raw: str) -> str:
        """Drop a MAC from the watchlist and stop its stream immediately."""
        if self._watchlist is None:
            return "Watchlist is not available."
        try:
            mac = normalize_mac(mac_raw)
        except ValueError:
            return f"That doesn't look like a MAC address: {mac_raw}"
        was_listed = self._watchlist.remove(mac)
        label = format_mac(mac)
        host = self._watchlist.ip_for_mac(mac)
        stopped = None
        if host is not None:
            live_mac = self._resolve_mac(host)
            if live_mac is None or live_mac == mac:
                stopped = self.stop_host(host)
            else:
                # DHCP moved the address to a DIFFERENT camera since the last
                # sighting — stopping by the cached IP would kill the wrong
                # stream. The removed camera (wherever it is now) is caught by
                # the next sweep's watchlist check.
                LOGGER.info(
                    "Watchlist remove %s: cached IP %s now belongs to %s; not stopping it",
                    label,
                    host,
                    format_mac(live_mac),
                )
        if not was_listed and stopped is None:
            return f"`{label}` wasn't on the watchlist."
        lines = []
        if was_listed:
            lines.append(f"🔴 Removed `{label}` from the watchlist.")
        if stopped is not None:
            lines.append(f"Stopped {stopped}.")
        if not self._watchlist.enforcing():
            # Empty list means filtering is OFF, which would quietly restart
            # this camera on the next sweep — say so instead of surprising.
            lines.append(
                "⚠️ The watchlist is now empty, so EVERY discovered camera "
                "will stream again. Add one back with /watchlist allow <MAC>."
            )
        return "\n".join(lines)

    def watchlist_text(self, display: Callable[[str], str] | None = None) -> str:
        """The ``/watchlist`` listing: cameras on the list and off it, each with
        IP + MAC, so the owner can curate by copy-pasting a MAC."""
        if self._watchlist is None:
            return "Watchlist is not available."
        known = self._watchlist.known()
        allowed = self._watchlist.allowed_macs()
        active = self.active_hosts()

        def entry(mac: str, host: str | None) -> str:
            if not host:
                return f"  ⚪ `{format_mac(mac)}` — not seen on the network yet"
            name = f"camera-{host}"
            shown = display(name) if display is not None else host
            live = host in active
            dot = "🟢" if live else "⚪"
            state = "streaming" if live else "not streaming"
            return f"  {dot} {shown} — {host} · `{format_mac(mac)}` ({state})"

        lines = ["📋 **Camera watchlist**"]
        if not allowed:
            lines.append(
                "The watchlist is empty — every discovered camera streams. "
                "Add a camera to restrict streaming to listed cameras only."
            )
        lines.extend(["", "**On the watchlist:**"])
        if allowed:
            for mac in sorted(allowed):
                record = known.get(mac)
                lines.append(entry(mac, record.ip if record else None))
        else:
            lines.append("  • none")
        others = {mac: record for mac, record in known.items() if mac not in allowed}
        lines.extend(["", "**Discovered, not on the watchlist:**"])
        if others:
            for mac in sorted(others):
                lines.append(entry(mac, others[mac].ip))
        else:
            lines.append("  • none")
        lines.extend(["", "/watchlist allow <MAC> · /watchlist remove <MAC>"])
        return "\n".join(lines)

    def discover_and_apply(
        self,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> DiscoveryApplied:
        """Sweep the LAN and start any confirmed-but-not-yet-active camera.

        The slow network scan is serialized because every scan publishes into
        the same live progress sink. This keeps dashboard and Telegram progress
        monotonic instead of interleaving two independent subnet sweeps.
        """
        with self._discovery_lock:
            result = discover_cameras(
                self._app_config.discovery,
                self._app_config.credentials,
                progress=self._progress,
                progress_callback=progress_callback,
            )
            found_hosts: set[str] = set()
            added: list[str] = []
            already_active = 0
            removed: list[str] = []
            blocked: list[str] = []
            macs: dict[str, str] = {}
            for found in result.cameras:
                # The probe just spoke to this host, so its MAC is sitting in
                # the ARP table. Cache every credential-confirmed camera by MAC
                # — including ones the watchlist keeps off-stream — so
                # /watchlist can offer them.
                mac = self._resolve_mac(found.host)
                if mac is not None:
                    macs[found.host] = mac
                    if self._watchlist is not None:
                        self._watchlist.record_sighting(mac, found.host)
                if self._watchlist is not None and not self._watchlist.permits(mac):
                    # Off the watchlist (or MAC unresolvable while the list is
                    # enforcing): confirmed but not streamed — and if it IS
                    # currently streaming (lost its spot since it started),
                    # stopped right here. The miss counter can't be relied on
                    # for this: it never runs on a sweep whose permitted set
                    # is empty.
                    identity = format_mac(mac) if mac else "MAC unknown"
                    label = f"{found.host} ({identity})"
                    if self.stop_host(found.host) is not None:
                        label += " — stream stopped"
                    blocked.append(label)
                    continue
                found_hosts.add(found.host)
                # Stable per-IP name so the same camera keeps its identity across
                # rediscovery and shows consistently in /status + the dashboard.
                camera = CameraConfig(
                    name=f"camera-{found.host}",
                    enabled=True,
                    rtsp_url=found.rtsp_url,
                    host=found.host,
                )
                if self.start_camera(camera):
                    added.append(camera.name)
                else:
                    already_active += 1
            # Retire stale cameras conservatively: never on a sweep that confirmed
            # NOTHING (almost always a transient network/scope blip rather than
            # every camera vanishing at once), and only after RETIRE_AFTER_MISSES
            # consecutive misses so one dropped probe can't tear down a live camera.
            if found_hosts:
                for host in sorted(self.active_hosts()):
                    if host in found_hosts:
                        self._misses.pop(host, None)
                        continue
                    misses = self._misses.get(host, 0) + 1
                    if misses >= RETIRE_AFTER_MISSES:
                        self._misses.pop(host, None)
                        name = self._stop_camera(host)
                        if name is not None:
                            removed.append(name)
                    else:
                        self._misses[host] = misses
            return DiscoveryApplied(
                result=result,
                added=added,
                already_active=already_active,
                removed=removed,
                blocked=blocked,
                macs=macs,
            )

    def join(self, timeout: float = 5.0) -> None:
        """Join the monitor threads on shutdown (best-effort within ``timeout``).

        ``stop_event`` is expected to already be set by the caller; the monitor
        loops observe it and exit, so this just waits for them to wind down.

        ``timeout`` bounds the TOTAL wait, not each thread: a shared deadline is
        used so a fleet of cameras all stalled inside a blocking ``capture.read``
        can't stretch shutdown to ``len(threads) * timeout``.
        """
        with self._threads_lock:
            runtimes = list(self._threads.values())
            for runtime in runtimes:
                runtime.stop_event.set()
        deadline = time.monotonic() + timeout
        for runtime in runtimes:
            runtime.thread.join(timeout=max(0.0, deadline - time.monotonic()))
