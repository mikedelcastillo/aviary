from __future__ import annotations

import threading

from lib.camera_names import CameraNamer, fallback_name, name_cameras, unique_name


def test_fallback_name_is_the_ip() -> None:
    # Until the VLM names it, show the honest IP — not a made-up name.
    assert fallback_name("camera-192.168.1.8") == "192.168.1.8"
    assert fallback_name("studio") == "studio"


def test_unique_name_disambiguates() -> None:
    assert unique_name("Window Perch", set()) == "Window Perch"
    assert unique_name("Window Perch", {"Window Perch"}) == "Window Perch 2"
    assert unique_name("Window Perch", {"Window Perch", "Window Perch 2"}) == "Window Perch 3"


def test_namer_display_falls_back_until_set() -> None:
    namer = CameraNamer()
    assert namer.display("camera-192.168.1.8") == "192.168.1.8"
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
    assert namer.display("camera-192.168.1.8") == "192.168.1.8"
    assert not namer.has("camera-192.168.1.8")


def test_name_cameras_leaves_unnamed_when_vlm_gives_nothing() -> None:
    namer = CameraNamer()
    # The model only ever returns banned/empty content -> cleans to "".
    name_cameras(
        namer,
        ["camera-192.168.1.45"],
        grab_frame=lambda cam: b"jpeg",
        client=FakeVlmClient(""),
        model="m",
        stop_event=threading.Event(),
        frame_attempts=1,
        name_attempts=3,
    )
    # Left UNNAMED (placeholder shown, retried later) — never locked to "Cam 45".
    assert not namer.has("camera-192.168.1.45")
    assert namer.display("camera-192.168.1.45") == "192.168.1.45"


def test_name_cameras_force_renames_existing() -> None:
    namer = CameraNamer()
    namer.set("camera-192.168.1.8", "Old Name")
    name_cameras(
        namer, ["camera-192.168.1.8"], grab_frame=lambda c: b"jpeg",
        client=FakeVlmClient("New Ledge"), model="m", stop_event=threading.Event(),
        frame_attempts=1, force=True,
    )
    assert namer.display("camera-192.168.1.8") == "New Ledge"


def test_name_cameras_force_same_name_keeps_no_suffix() -> None:
    namer = CameraNamer()
    namer.set("camera-192.168.1.8", "Big Cage")
    # Re-naming to the SAME name must not become "Big Cage 2" (self-exclusion).
    name_cameras(
        namer, ["camera-192.168.1.8"], grab_frame=lambda c: b"jpeg",
        client=FakeVlmClient("Big Cage"), model="m", stop_event=threading.Event(),
        frame_attempts=1, force=True,
    )
    assert namer.display("camera-192.168.1.8") == "Big Cage"


def test_name_cameras_force_keeps_old_name_when_vlm_fails() -> None:
    namer = CameraNamer()
    namer.set("camera-192.168.1.8", "Good Name")
    name_cameras(
        namer, ["camera-192.168.1.8"], grab_frame=lambda c: b"jpeg",
        client=FakeVlmClient(""), model="m", stop_event=threading.Event(),
        frame_attempts=1, name_attempts=2, force=True,
    )
    # Re-name failed -> keep the existing good name.
    assert namer.display("camera-192.168.1.8") == "Good Name"


def test_camera_namer_persists_and_reloads(tmp_path) -> None:
    cache = tmp_path / "camera_names.json"
    namer = CameraNamer(cache_path=cache)
    namer.set("camera-192.168.1.8", "Window Perch")
    namer.set("camera-192.168.1.9", "Food Bowl")
    assert cache.exists()
    # A fresh namer (e.g. after a restart) loads the cached names immediately.
    reloaded = CameraNamer(cache_path=cache)
    assert reloaded.display("camera-192.168.1.8") == "Window Perch"
    assert reloaded.display("camera-192.168.1.9") == "Food Bowl"
    assert reloaded.has("camera-192.168.1.8")


def test_camera_namer_missing_cache_is_fine(tmp_path) -> None:
    namer = CameraNamer(cache_path=tmp_path / "nope.json")
    # Nothing cached yet -> IP fallback, no crash.
    assert namer.display("camera-192.168.1.8") == "192.168.1.8"


def test_force_rename_skips_fixed_already_named_cameras() -> None:
    namer = CameraNamer()
    namer.set("camera-192.168.1.30", "Big Cage")   # pan-tilt (movable)
    namer.set("camera-192.168.1.44", "Window Sill")  # fixed
    client = FakeVlmClient("New Ledge")
    movable = {"camera-192.168.1.30"}
    name_cameras(
        namer,
        ["camera-192.168.1.30", "camera-192.168.1.44"],
        grab_frame=lambda c: b"jpeg",
        client=client,
        model="m",
        stop_event=threading.Event(),
        frame_attempts=1,
        force=True,
        is_movable=lambda cam: cam in movable,
    )
    # The movable (PTZ) camera was re-named; the fixed one kept its cached name.
    assert namer.display("camera-192.168.1.30") == "New Ledge"
    assert namer.display("camera-192.168.1.44") == "Window Sill"


def test_force_rename_still_names_unnamed_fixed_camera() -> None:
    namer = CameraNamer()
    name_cameras(
        namer,
        ["camera-192.168.1.44"],  # fixed AND unnamed -> must still be named
        grab_frame=lambda c: b"jpeg",
        client=FakeVlmClient("Window Sill"),
        model="m",
        stop_event=threading.Event(),
        frame_attempts=1,
        force=True,
        is_movable=lambda cam: False,
    )
    assert namer.display("camera-192.168.1.44") == "Window Sill"


def test_claim_assigns_and_dedups() -> None:
    namer = CameraNamer()
    assert namer.claim("camera-1", "Big Cage") == "Big Cage"
    # A second camera claiming the same base gets a numeric suffix.
    assert namer.claim("camera-2", "Big Cage") == "Big Cage 2"
    assert namer.display("camera-1") == "Big Cage"
    assert namer.display("camera-2") == "Big Cage 2"


def test_claim_same_camera_same_base_keeps_name() -> None:
    namer = CameraNamer()
    namer.set("camera-1", "Big Cage")
    # Re-claiming the same descriptive name for the SAME camera keeps it (the
    # camera's own current name is excluded from the collision check).
    assert namer.claim("camera-1", "Big Cage") == "Big Cage"


def test_claim_is_atomic_under_concurrency() -> None:
    # Many naming passes claiming the SAME base for DIFFERENT cameras at once must
    # never hand out a duplicate display name (the boot + /discover race).
    namer = CameraNamer()
    results: list[str] = []
    results_lock = threading.Lock()
    start = threading.Event()

    def claim(index: int) -> None:
        start.wait()  # line everyone up to maximise contention
        display = namer.claim(f"camera-{index}", "Perch")
        with results_lock:
            results.append(display)

    threads = [threading.Thread(target=claim, args=(i,)) for i in range(16)]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join()

    assert len(results) == 16
    assert len(set(results)) == 16  # every camera got a distinct name
