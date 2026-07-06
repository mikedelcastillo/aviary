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
    discover_cameras,
    redact_rtsp_url,
)
from lib.detection_log import DetectionLogger
from lib.objects import ObjectRegistry
from lib.quality import StreamQualityController
from lib.stats import CameraStats


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
    lines = [
        f"✅ **Discovery complete** — scanned {result.hosts_scanned} hosts "
        f"in {result.elapsed_seconds:.1f}s",
        f"Found {cameras} camera(s) · {result.ports_open} host(s) with port :554 open",
        "",
        "📷 **Cameras**",
    ]
    if applied.added:
        lines.append("  🟢 Started: " + ", ".join(applied.added))
    else:
        lines.append("  • No new cameras started.")
    if applied.already_active:
        lines.append(f"  ✔️ {applied.already_active} already running")
    if applied.removed:
        lines.append("  🔴 Stopped stale: " + ", ".join(applied.removed))
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
        # Shared privacy/pause state. Passed to every monitor thread so a pause
        # stops all cameras consuming their streams at once.
        self._control = control
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
        LOGGER.info("Retired stale camera %s", name)
        return name

    def active_hosts(self) -> set[str]:
        """Hosts currently being monitored."""
        with self._threads_lock:
            return set(self._threads)

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
            found_hosts = {camera.host for camera in result.cameras}
            added: list[str] = []
            already_active = 0
            removed: list[str] = []
            for found in result.cameras:
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
