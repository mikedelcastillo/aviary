"""RTSP camera capture and inference loop."""

from __future__ import annotations

import logging
import os
import threading
import time

import cv2

from lib.alerts import AlertDispatcher, AlertState
from lib.config import CameraConfig
from lib.detector import ObjectDetector
from lib.stats import CameraStats


LOGGER = logging.getLogger("lib.camera")

# Number of consecutive failed/empty reads tolerated before we tear the stream
# down and reconnect. Absorbs the occasional dropped frame without paying for a
# full RTSP re-handshake on every hiccup.
READ_FAILURE_LIMIT = 3


def configure_ffmpeg_capture(cameras: list[CameraConfig]) -> None:
    """Pin OpenCV's FFmpeg backend to TCP with a socket timeout.

    ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` is read by the FFmpeg backend each time a
    ``VideoCapture`` is constructed, so setting it once before the worker
    threads start covers every camera.
    """
    if not cameras:
        return
    transport = cameras[0].rtsp_transport
    # FFmpeg's ``timeout`` is the socket I/O timeout in microseconds.
    timeout_us = int(max(camera.read_timeout_seconds for camera in cameras) * 1_000_000)
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
) -> None:
    LOGGER.info("Starting camera %s", camera.name)
    min_frame_interval = 1.0 / camera.sample_fps
    backoff = camera.reconnect_seconds

    while not stop_event.is_set():
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
        try:
            while not stop_event.is_set():
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

                now = time.monotonic()
                if now < next_inference_at:
                    continue
                next_inference_at = now + min_frame_interval

                detections = detector.predict(frame)
                # Record after inference returns, so the cell's FPS reflects true
                # capture+YOLO throughput, not the raw stream rate.
                stats.record_inference([detection.label for detection in detections])
                if detections:
                    # Cooldown check is cheap (a lock + dict lookup); the slow
                    # snapshot + Telegram work is handed to the dispatcher so
                    # this loop returns to reading frames immediately.
                    eligible = alert_state.eligible(camera.name, detections)
                    if eligible:
                        stats.record_alert(len(eligible))
                        dispatcher.submit(camera, frame, eligible)
        except Exception:
            LOGGER.exception("Camera loop error for %s", camera.name)
        finally:
            capture.release()

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
