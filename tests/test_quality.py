from __future__ import annotations

from lib.config import CameraCredentials
from lib.quality import StreamQualityController


def test_quality_defaults_to_stream1() -> None:
    controller = StreamQualityController(CameraCredentials("user", "pass"))
    controller.register("10.0.0.5", "/stream1")

    assert controller.mode() == "stream1"
    assert controller.selected("10.0.0.5")[0] == "/stream1"


def test_quality_mode_forces_stream_path_and_url() -> None:
    controller = StreamQualityController(CameraCredentials("user", "pass"), mode="auto")
    controller.register("10.0.0.5", "/stream1")

    url, path, version = controller.rtsp_url("10.0.0.5")
    assert path == "/stream2"
    assert "/stream2" in url

    message = controller.set_mode("stream1")
    url, path, new_version = controller.rtsp_url("10.0.0.5")

    assert "stream1" in message
    assert path == "/stream1"
    assert "/stream1" in url
    assert new_version > version


def test_quality_auto_promotes_and_downgrades_from_fps() -> None:
    controller = StreamQualityController(
        CameraCredentials("user", "pass"),
        mode="auto",
        promote_samples=2,
        downgrade_samples=2,
    )
    controller.register("10.0.0.5", "/stream1")

    assert controller.selected("10.0.0.5")[0] == "/stream2"
    healthy = {
        "status": "connected",
        "fps": 1.0,
        "since_frame": 0.2,
        "consecutive_failures": 0,
    }
    controller.observe("10.0.0.5", healthy, target_fps=1.0)
    controller.observe("10.0.0.5", healthy, target_fps=1.0)
    assert controller.selected("10.0.0.5")[0] == "/stream1"

    weak = {
        "status": "connected",
        "fps": 0.2,
        "since_frame": 0.2,
        "consecutive_failures": 0,
    }
    controller.observe("10.0.0.5", weak, target_fps=1.0)
    controller.observe("10.0.0.5", weak, target_fps=1.0)
    assert controller.selected("10.0.0.5")[0] == "/stream2"


def test_quality_rejects_unknown_mode() -> None:
    controller = StreamQualityController(CameraCredentials("user", "pass"))
    assert "Usage" in controller.set_mode("ultra")
