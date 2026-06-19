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
from dataclasses import dataclass

from lib.alerts import AlertDispatcher, AlertState
from lib.camera import monitor_camera
from lib.config import AppConfig, CameraConfig
from lib.detector import ObjectDetector
from lib.discovery import (
    DiscoveryProgress,
    DiscoveryResult,
    discover_cameras,
    redact_rtsp_url,
)
from lib.objects import ObjectRegistry
from lib.stats import CameraStats


LOGGER = logging.getLogger("lib.supervisor")


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


def format_discovery_report(applied: DiscoveryApplied) -> str:
    """Render a human-readable ``/discover`` reply from a DiscoveryApplied."""
    result = applied.result
    lines = [
        "Discovery complete.",
        (
            f"Scanned {result.hosts_scanned} hosts in "
            f"{result.elapsed_seconds:.1f}s: "
            f"{result.ports_open} with :554 open, "
            f"{len(result.cameras)} confirmed camera(s)."
        ),
    ]
    if result.auth_failures:
        # Surface bad creds explicitly: a camera that answers :554 but rejects
        # the password is almost always a TAPO_CREDENTIALS mismatch, which is
        # otherwise invisible (it just never shows up as a camera).
        lines.append(
            f"{result.auth_failures} host(s) rejected the credentials "
            "(check TAPO_CREDENTIALS)."
        )
    if applied.added:
        lines.append("Started: " + ", ".join(applied.added) + ".")
    else:
        lines.append("No new cameras started.")
    if applied.already_active:
        lines.append(f"{applied.already_active} already running.")
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
    ) -> None:
        self._app_config = app_config
        self._detector = detector
        self._alert_state = alert_state
        self._dispatcher = dispatcher
        self._registry = registry
        self._stats = stats
        self._stats_lock = stats_lock
        self._stop_event = stop_event
        # Live per-host sweep state for the dashboard's discovery grid. Shared
        # with the Dashboard so the camera band can switch to "discover mode"
        # while a scan runs. Optional: discovery works fine without it.
        self._progress = progress

        # host -> monitor thread. Used both for dedup (an active host is never
        # restarted) and to join the threads on shutdown.
        self._threads: dict[str, threading.Thread] = {}
        # Serialises every mutation of (and read of) ``_threads`` so the dedup
        # check-then-start in ``start_camera`` is atomic. An RLock (not a plain
        # Lock) because ``discover_and_apply`` may already hold it when it calls
        # ``start_camera`` re-entrantly.
        self._threads_lock = threading.RLock()

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

            thread = threading.Thread(
                target=monitor_camera,
                args=(
                    camera,
                    self._detector,
                    self._alert_state,
                    self._dispatcher,
                    camera_stats,
                    self._stop_event,
                ),
                name=f"camera-{host}",
                daemon=True,
            )
            self._threads[host] = thread
            thread.start()
        LOGGER.info("Started camera %s -> %s", camera.name, redact_rtsp_url(camera.rtsp_url))
        return True

    def active_hosts(self) -> set[str]:
        """Hosts currently being monitored."""
        with self._threads_lock:
            return set(self._threads)

    def discover_and_apply(self) -> DiscoveryApplied:
        """Sweep the LAN and start any confirmed-but-not-yet-active camera.

        The slow network scan runs WITHOUT any lock held so a concurrent
        ``/discover`` (or ``/status``) on the Telegram poll thread isn't blocked
        behind it for the full sweep. Overlapping scans are harmless: the actual
        thread-start in :meth:`start_camera` is atomic per host, so dedup holds
        even if two scans both report the same new camera.
        """
        result = discover_cameras(
            self._app_config.discovery,
            self._app_config.credentials,
            progress=self._progress,
        )
        added: list[str] = []
        already_active = 0
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
        return DiscoveryApplied(
            result=result, added=added, already_active=already_active
        )

    def join(self, timeout: float = 5.0) -> None:
        """Join the monitor threads on shutdown (best-effort within ``timeout``).

        ``stop_event`` is expected to already be set by the caller; the monitor
        loops observe it and exit, so this just waits for them to wind down.
        """
        with self._threads_lock:
            threads = list(self._threads.values())
        for thread in threads:
            thread.join(timeout=timeout)
