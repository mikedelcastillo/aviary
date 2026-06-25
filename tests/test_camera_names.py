from __future__ import annotations

import threading

from lib.camera_names import CameraNamer, fallback_name, name_cameras, unique_name


def test_fallback_name_uses_last_octet_not_a_decimal() -> None:
    assert fallback_name("camera-192.168.1.8") == "Cam 8"
    assert fallback_name("studio") == "studio"


def test_unique_name_disambiguates() -> None:
    assert unique_name("Window Perch", set()) == "Window Perch"
    assert unique_name("Window Perch", {"Window Perch"}) == "Window Perch 2"
    assert unique_name("Window Perch", {"Window Perch", "Window Perch 2"}) == "Window Perch 3"


def test_namer_display_falls_back_until_set() -> None:
    namer = CameraNamer()
    assert namer.display("camera-192.168.1.8") == "Cam 8"
    namer.set("camera-192.168.1.8", "Window Perch")
    assert namer.display("camera-192.168.1.8") == "Window Perch"
    assert namer.has("camera-192.168.1.8")


class FakeVlmClient:
    def __init__(self, name: str, distinct: str | None = None) -> None:
        self.name = name
        self.distinct = distinct

    def generate(self, model, prompt, *, images=None, timeout_seconds=None, **kwargs):
        # The disambiguation prompt mentions names already taken.
        if self.distinct is not None and "already" in prompt:
            return self.distinct
        return self.name


def test_name_cameras_assigns_unique_vlm_names() -> None:
    namer = CameraNamer()
    client = FakeVlmClient("Window Perch")
    # Both cameras get the same VLM suggestion -> second is disambiguated.
    name_cameras(
        namer,
        ["camera-192.168.1.8", "camera-192.168.1.9"],
        grab_frame=lambda cam: b"jpeg",
        client=client,
        model="qwen2.5vl:7b",
        stop_event=threading.Event(),
        frame_attempts=1,
    )
    names = {namer.display("camera-192.168.1.8"), namer.display("camera-192.168.1.9")}
    assert names == {"Window Perch", "Window Perch 2"}


def test_name_cameras_disambiguates_collision_via_vlm() -> None:
    namer = CameraNamer()
    # Both views suggest "Big Cage"; the second re-asks the VLM, which returns a
    # distinct "Window Ledge" instead of a numeric suffix.
    client = FakeVlmClient("Big Cage", distinct="Window Ledge")
    name_cameras(
        namer,
        ["camera-192.168.1.8", "camera-192.168.1.9"],
        grab_frame=lambda cam: b"jpeg",
        client=client,
        model="qwen2.5vl:7b",
        stop_event=threading.Event(),
        frame_attempts=1,
    )
    names = {namer.display("camera-192.168.1.8"), namer.display("camera-192.168.1.9")}
    assert names == {"Big Cage", "Window Ledge"}


def test_name_cameras_skips_cameras_without_frames() -> None:
    namer = CameraNamer()
    name_cameras(
        namer,
        ["camera-192.168.1.8"],
        grab_frame=lambda cam: None,  # never produces a frame
        client=FakeVlmClient("X"),
        model="m",
        stop_event=threading.Event(),
        frame_attempts=2,
    )
    # Unnamed -> still the fallback, not crashed.
    assert namer.display("camera-192.168.1.8") == "Cam 8"
    assert not namer.has("camera-192.168.1.8")
