"""Dashboard logging and native stderr redirection helpers."""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from collections.abc import Callable

try:  # OpenCV is always present (transitive via ultralytics); guard anyway.
    import cv2
except Exception:  # pragma: no cover - cv2 import is exercised everywhere else
    cv2 = None


class EventLogHandler(logging.Handler):
    """Feeds Python log records into the dashboard's events panel."""

    def __init__(self, add_event: Callable[[str, str], None]) -> None:
        super().__init__()
        self._add_event = add_event

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._add_event(record.getMessage(), record.levelname)
        except Exception:  # never let logging crash the app
            pass


def configure_dashboard_logging(
    *,
    add_event: Callable[[str, str], None],
    is_tty: bool,
    log_level: int,
    logfile: str,
) -> None:
    root = logging.getLogger()
    root.setLevel(log_level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # Always tee everything to the logfile so h264/timeout detail and full
    # tracebacks survive for post-mortem, regardless of render mode. Rotated so
    # a long-lived server can't grow the file without bound (it once reached
    # 208MB); ~3 generations of 50MB is plenty of post-mortem window. The raw
    # fd-2 redirect (NativeStderrRedirect) holds its own handle, so its ffmpeg
    # noise follows a rotated-out file until the next restart — acceptable.
    file_handler = logging.handlers.RotatingFileHandler(
        logfile, maxBytes=50_000_000, backupCount=2
    )
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(file_handler)

    if is_tty:
        # On a terminal the events panel is the on-screen log; a stream handler
        # would scribble over the Live render.
        root.addHandler(EventLogHandler(add_event))
    else:
        # Non-TTY (docker logs): plain stream logging is the right thing.
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        root.addHandler(stream_handler)


class NativeStderrRedirect:
    """Redirect native fd 2 noise into the dashboard logfile while active."""

    def __init__(self, logfile: str) -> None:
        self._logfile_path = logfile
        self._saved_stderr_fd: int | None = None
        self._logfile = None

    def start(self) -> None:
        if cv2 is not None:
            try:
                cv2.setLogLevel(0)  # 0 == LOG_LEVEL_SILENT
            except Exception:
                pass
        try:
            self._logfile = open(self._logfile_path, "a", buffering=1)
            self._saved_stderr_fd = os.dup(2)
            os.dup2(self._logfile.fileno(), 2)
        except Exception:
            self._saved_stderr_fd = None

    def stop(self) -> None:
        if self._saved_stderr_fd is not None:
            try:
                sys.stderr.flush()
            except Exception:
                pass
            os.dup2(self._saved_stderr_fd, 2)
            os.close(self._saved_stderr_fd)
            self._saved_stderr_fd = None
        if self._logfile is not None:
            self._logfile.close()
            self._logfile = None
