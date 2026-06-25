"""RTSP camera capture and inference loop."""

from __future__ import annotations

import logging
import os
import threading
import time

import cv2

from lib.alerts import AlertDispatcher, AlertState
from lib.config import CameraConfig
from lib.control import RuntimeControl
from lib.detector import ObjectDetector
from lib.discovery import redact_rtsp_url
from lib.objects import frame_size_from_shape
from lib.stats import CameraStats


LOGGER = logging.getLogger("lib.camera")

# Number of consecutive failed/empty reads tolerated before we tear the stream
# down and reconnect. Absorbs the occasional dropped frame without paying for a
# full RTSP re-handshake on every hiccup.
READ_FAILURE_LIMIT = 3

# How often a paused (privacy-mode) camera re-checks whether to resume. Short so
# a /play or a lapsed timed pause brings the stream back within ~a second.
PAUSE_POLL_SECONDS = 1.0


def configure_ffmpeg_capture(cameras: list[CameraConfig] | None = None) -> None:
    """Pin OpenCV's FFmpeg backend to TCP with a socket timeout.

    ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` is read by the FFmpeg backend each time a
    ``VideoCapture`` is constructed, so setting it once before the worker
    threads start covers every camera.

    Cameras are now discovered at runtime, so this may be called before any
    camera exists (empty/``None`` list). In that case we fall back to the
    ``CameraConfig`` field DEFAULTS: every discovered camera is built with those
    same defaults (discovery never customises transport/timeout), so the env
    option is identical whether or not a camera happens to be known yet.
    """
    if cameras:
        transport = cameras[0].rtsp_transport
        # FFmpeg's ``timeout`` is the socket I/O timeout in microseconds.
        timeout_us = int(max(camera.read_timeout_seconds for camera in cameras) * 1_000_000)
    else:
        # No cameras yet: read the shared defaults straight off the dataclass
        # fields so the FFmpeg backend is configured before the first /discover.
        defaults = CameraConfig.__dataclass_fields__
        transport = defaults["rtsp_transport"].default
        timeout_us = int(defaults["read_timeout_seconds"].default * 1_000_000)

    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        f"rtsp_transport;{transport}|timeout;{timeout_us}",
    )


def open_capture(camera: CameraConfig) -> cv2.VideoCapture | None:
    """Open an RTSP capture with bounded open/read timeouts."""
    params: list[int] = []
    for prop_name, seconds in (
        ("CAP_PROP_OPEN_TIMEOUT_MSEC", camera.open_timeout_seconds),
        ("CAP_PROP_READ_TIMEOUT_MSEC", camera.read_timeout_seconds),
    ):
        prop = getattr(cv2, prop_name, None)
        if prop is not None:
            params += [prop, int(seconds * 1000)]

    try:
        capture = cv2.VideoCapture(camera.rtsp_url, cv2.CAP_FFMPEG, params)
    except Exception:
        LOGGER.exception("Error opening stream for %s", camera.name)
        return None

    if not capture.isOpened():
        capture.release()
        return None

    # Keep the buffer shallow so inference runs on a fresh frame after any stall
    # instead of draining a backlog of stale ones.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def monitor_camera(
    camera: CameraConfig,
    detector: ObjectDetector,
    alert_state: AlertState,
    dispatcher: AlertDispatcher,
    stats: CameraStats,
    stop_event: threading.Event,
    control: RuntimeControl | None = None,
) -> None:
    # Log the exact URL (password masked) so a camera stuck on "connecting" can
    # be diagnosed: if this line appears but "Stream opened" never follows, the
    # ffmpeg open is blocking on this URL — compare it to a known-good one.
    LOGGER.info("Starting camera %s -> %s", camera.name, redact_rtsp_url(camera.rtsp_url))
    min_frame_interval = 1.0 / camera.sample_fps
    backoff = camera.reconnect_seconds

    while not stop_event.is_set():
        # Privacy mode: don't even open the stream. Pulling no bytes off the
        # camera is the whole point — idle here until /play (or a timed pause
        # lapsing) flips us back. The status reads "paused" so the dashboard and
        # /status show why the camera has gone quiet.
        if control is not None and control.is_paused():
            stats.set_status("paused")
            stop_event.wait(PAUSE_POLL_SECONDS)
            continue

        stats.set_status("connecting")
        capture = open_capture(camera)
        if capture is None:
            LOGGER.warning("Could not open stream for %s; retrying in %.1fs", camera.name, backoff)
            stats.set_status("reconnecting", backoff)
            stats.record_reconnect()
            stop_event.wait(backoff)
            backoff = min(backoff * 2, camera.max_reconnect_seconds)
            continue

        LOGGER.info("Stream opened for %s", camera.name)
        stats.set_status("connected")
        next_inference_at = 0.0
        consecutive_failures = 0
        paused_out = False
        try:
            while not stop_event.is_set():
                # Privacy mode flipped on mid-session: drop the stream now. The
                # ``finally`` releases the capture; ``paused_out`` then skips the
                # unhealthy-reconnect backoff so we go straight back to idling.
                if control is not None and control.is_paused():
                    LOGGER.info("Privacy mode on; releasing %s", camera.name)
                    paused_out = True
                    break

                ok, frame = capture.read()
                if not ok or frame is None:
                    consecutive_failures += 1
                    stats.record_read_failure()
                    if consecutive_failures >= READ_FAILURE_LIMIT:
                        LOGGER.warning(
                            "Stream read failed for %s (%d consecutive); reconnecting",
                            camera.name,
                            consecutive_failures,
                        )
                        break
                    continue

                consecutive_failures = 0
                # A real frame arrived: the stream is healthy, so drop back to
                # the base reconnect delay for the next outage.
                backoff = camera.reconnect_seconds

                # Publish the freshest frame for on-demand snapshots regardless of
                # the inference throttle below, so /snapshot returns a near-live
                # image even on cameras sampled at a low FPS. A reference swap
                # under a dedicated lock, so it costs nothing on the hot path.
                stats.set_latest_frame(frame)

                now = time.monotonic()
                if now < next_inference_at:
                    continue
                next_inference_at = now + min_frame_interval

                detections = detector.predict(frame)
                frame_size = frame_size_from_shape(frame.shape)
                # Record after inference returns, so the cell's FPS reflects true
                # capture+YOLO throughput, not the raw stream rate.
                stats.record_inference(
                    [detection.label for detection in detections],
                    detections,
                    frame_size,
                )
                if detections:
                    # Eligibility check is cheap (a lock + dict lookup); the
                    # slow snapshot + Telegram work is handed to the dispatcher
                    # so this loop returns to reading frames immediately.
                    eligible = alert_state.eligible(camera.name, detections, frame_size)
                    # Re-check privacy right before delivery: a pause that landed
                    # after this frame was read must not let a stray alert escape.
                    if eligible and not (control is not None and control.is_paused()):
                        stats.record_alert(len(eligible))
                        stats.record_object_alert(eligible)
                        dispatcher.submit(camera, frame, eligible)
        except Exception:
            LOGGER.exception("Camera loop error for %s", camera.name)
        finally:
            capture.release()

        # Released for privacy, not a fault: loop back to the paused idle at the
        # top of the outer loop without the unhealthy-reconnect backoff.
        if paused_out:
            continue

        # The session ended unhealthily (stall, dropped stream, or decode
        # error). Pace before reconnecting and grow the delay while the camera
        # stays down; a healthy read above already reset it to the base.
        if not stop_event.is_set():
            stats.set_status("reconnecting", backoff)
            stats.record_reconnect()
            stop_event.wait(backoff)
            backoff = min(backoff * 2, camera.max_reconnect_seconds)

    stats.set_status("stopped")
    LOGGER.info("Stopped camera %s", camera.name)
