"""Tests for the RGB LED status display (lib.rgb). All hardware-free: they use a
VirtualSurface and call scene/engine logic directly (no OpenRGB server needed)."""
from __future__ import annotations

import threading

import pytest

from lib.rgb import birds
from lib.rgb.engine import RGBController, parse_color, parse_duration
from lib.rgb.layout import Layout, default_layout, load_layout
from lib.rgb.palette import BLACK, WHITE, Color, pulse
from lib.rgb.scenes import SceneState, Sighting
from lib.rgb import scenes
from lib.rgb.surface import LedInfo, VirtualSurface


# --- palette -----------------------------------------------------------------
def test_color_from_hex_and_roundtrip():
    assert Color.from_hex("#FF9500") == Color(255, 149, 0)
    assert Color.from_hex("00FF09").to_hex() == "#00FF09"


def test_scale_endpoints():
    assert WHITE.scale(0.0) == BLACK
    assert WHITE.scale(1.0) == WHITE
    mid = WHITE.scale(0.5)
    assert 0 < mid.r < 255


def test_blend_endpoints():
    assert WHITE.blend(BLACK, 0.0) == WHITE
    assert WHITE.blend(BLACK, 1.0) == BLACK


def test_pulse_range():
    vals = [pulse(t / 10.0, 0.2, 1.0, 2.0) for t in range(40)]
    assert min(vals) >= 0.2 - 1e-9
    assert max(vals) <= 1.0 + 1e-9


# --- birds / roster ----------------------------------------------------------
def test_bird_colors_match_legend():
    assert birds.color_for("draft") == Color.from_hex("#FFFFFF")
    assert birds.color_for("matcha") == Color.from_hex("#00FF09")
    assert birds.color_for("budgie") == Color.from_hex("#FFD000")


def test_night_species_classification():
    assert birds.is_night_species("cockatiel")
    assert not birds.is_night_species("draft")
    assert birds.is_individual("draft")
    assert not birds.is_individual("cockatiel")


def test_roster_validation_clean():
    # The LED roster must stay in sync with training/roster.yaml live labels.
    assert birds.validate_against() == []


# --- parsing -----------------------------------------------------------------
@pytest.mark.parametrize("token,expected", [
    ("red", Color(255, 0, 0)),
    ("#00ff00", Color(0, 255, 0)),
    ("0f0", Color(0, 255, 0)),
    ("draft", Color.from_hex("#FFFFFF")),
    ("off", BLACK),
])
def test_parse_color_ok(token, expected):
    assert parse_color(token) == expected


def test_parse_color_bad():
    assert parse_color("banana") is None
    assert parse_color("") is None


@pytest.mark.parametrize("text,secs", [
    ("10 mins", 600), ("10m", 600), ("30s", 30), ("1h", 3600),
    ("10", 600), ("2 minutes", 120), ("90 sec", 90),
])
def test_parse_duration_ok(text, secs):
    assert parse_duration(text) == secs


def test_parse_duration_bad():
    assert parse_duration("soon") is None
    assert parse_duration("") is None


def test_parse_duration_bare_unit():
    # bare number defaults to minutes, but callers can request seconds
    assert parse_duration("30") == 1800
    assert parse_duration("30", bare_unit=1.0) == 30
    assert parse_duration("30s", bare_unit=1.0) == 30  # explicit unit unaffected


def test_rgb_test_duration_is_seconds():
    import time
    ctrl = _controller()
    ctrl.command("test 30")  # 30 seconds, not 30 minutes
    assert 25 < (ctrl._manual_until - time.monotonic()) <= 30


def test_boot_handles_tiny_and_normal_bars():
    # No index errors for degenerate bar sizes; correct frame length.
    for n in (1, 2, 5, 9):
        lay = default_layout(_surface(n))
        for t in (0.0, 0.5, 1.3, 2.7, 3.9):
            frame = scenes.boot(t, lay, duration=4.0)
            assert len(frame) == n


# --- layout ------------------------------------------------------------------
def _surface(n: int) -> VirtualSurface:
    leds = [LedInfo(i, 0, "Virtual", "z", i >= 4, f"led{i}") for i in range(n)]
    return VirtualSurface(leds=leds)


def test_default_layout_bar_and_night():
    lay = default_layout(_surface(5))
    assert lay.bar == [0, 1, 2, 3, 4]  # stack flows in surface order
    assert lay.night_led == 4          # last LED is the night LED
    lay9 = default_layout(_surface(9))
    assert lay9.bar == list(range(9))
    assert lay9.night_led == 8


def test_layout_save_load_roundtrip(tmp_path):
    lay = Layout(size=9, bar=[2, 1, 0], night_led=8)
    path = tmp_path / "rgb_layout.json"
    lay.save(path)
    loaded = load_layout(_surface(9), path=path)
    assert loaded.bar == [2, 1, 0]
    assert loaded.night_led == 8


# --- scenes ------------------------------------------------------------------
def test_discovery_bar_fills_with_fraction():
    lay = default_layout(_surface(8))
    st = SceneState(discovery_active=True, discovery_fraction=0.5)
    frame = scenes.discovery(st, t=0.0, layout=lay)
    # The filled portion is clearly bright; the track ahead is a faint glow.
    bright = [c for c in frame if max(c.as_tuple()) > 90]
    assert 3 <= len(bright) <= 5  # ~half of the 8-LED bar
    # and the filled half is brighter than the unfilled half
    first_half = sum(max(frame[i].as_tuple()) for i in range(4))
    second_half = sum(max(frame[i].as_tuple()) for i in range(4, 8))
    assert first_half > second_half


def test_night_scene_lights_only_night_led():
    lay = default_layout(_surface(5))  # night_led == 4
    st = SceneState(night=Sighting("lovebird", Color(255, 0, 0), 0.9, last_seen=10.0, first_seen=10.0))
    frame = scenes.night(st, t=10.0, layout=lay)
    assert all(frame[i].is_dark() for i in range(4))
    assert not frame[4].is_dark()
    assert frame[4].r > frame[4].g  # red-ish lovebird


def test_night_scene_dark_when_no_sighting():
    lay = default_layout(_surface(5))
    frame = scenes.night(SceneState(), t=0.0, layout=lay)
    assert all(c.is_dark() for c in frame)


def _stack_state(labels_in_order, t=5.0):
    """Build a SceneState whose queue is labels_in_order (front first)."""
    st = SceneState(queue=list(labels_in_order))
    for lbl in labels_in_order:
        b = birds.get(lbl)
        st.day[lbl] = Sighting(lbl, b.color, 0.9, last_seen=t, first_seen=t)
    return st


def test_detection_stack_orders_newest_first():
    lay = default_layout(_surface(5))
    # queue front (index 0) = newest = matcha; percy is older
    st = _stack_state(["matcha", "percy"])
    frame = scenes.detection(st, t=5.0, layout=lay)
    assert frame[0].as_tuple() != (0, 0, 0)  # front lit (matcha, green)
    assert frame[1].as_tuple() != (0, 0, 0)  # next (percy, orange)
    # front (newest) is brighter than the one behind it (recency gradient)
    assert max(frame[0].as_tuple()) >= max(frame[1].as_tuple())
    assert all(frame[i].is_dark() for i in (2, 3, 4))  # nothing else


def test_detection_stack_full_flock_is_party():
    lay = default_layout(_surface(9))
    st = _stack_state([b.name for b in birds.INDIVIDUALS])
    frame = scenes.detection(st, t=5.0, layout=lay)
    assert sum(0 if c.is_dark() else 1 for c in frame) >= 6  # rainbow across the bar


# --- engine ------------------------------------------------------------------
def _controller(n: int = 5) -> RGBController:
    # Pass an explicit layout so the test never reads the on-disk calibration file
    # (keeps it hermetic regardless of any data/server/rgb_layout.json present).
    surf = _surface(n)
    return RGBController(threading.Event(), surface=surf, layout=default_layout(surf), brightness=1.0)


def test_command_manual_then_auto():
    ctrl = _controller()
    msg = ctrl.command("red 5m")
    assert "red" in msg and "5m" in msg
    # manual override renders solid red regardless of state
    frame = ctrl._render(t=ctrl._manual_until - 1.0)
    assert all(c == Color(255, 0, 0) for c in frame)
    ctrl.command("auto")
    assert ctrl._manual_color is None


def test_command_default_duration_is_ten_minutes():
    ctrl = _controller()
    ctrl.command("blue")
    # ~600s in the future
    import time
    assert 590 < (ctrl._manual_until - time.monotonic()) <= 600


def test_command_unknown_color_message():
    ctrl = _controller()
    assert "Unknown color" in ctrl.command("banana")


class _Det:
    def __init__(self, label, conf):
        self.label = label
        self.confidence = conf


def test_on_detection_populates_day_state():
    ctrl = _controller()
    ctrl.on_detection("cam1", [_Det("percy", 0.9)], is_ir=False)
    assert "percy" in ctrl._state.day
    # night species on a day frame is ignored; individual on IR frame ignored
    ctrl.on_detection("cam1", [_Det("lovebird", 0.8)], is_ir=True)
    assert ctrl._state.night is not None and ctrl._state.night.label == "lovebird"


def test_detection_stack_pushes_new_bird_to_front():
    ctrl = _controller()
    ctrl.on_detection("c", [_Det("percy", 0.9)], is_ir=False)
    ctrl.on_detection("c", [_Det("matcha", 0.9)], is_ir=False)
    ctrl.on_detection("c", [_Det("jynx", 0.9)], is_ir=False)
    # newest first
    assert ctrl._queue == ["jynx", "matcha", "percy"]


def test_detection_stack_present_bird_keeps_position():
    ctrl = _controller()
    ctrl.on_detection("c", [_Det("percy", 0.9)], is_ir=False)
    ctrl.on_detection("c", [_Det("matcha", 0.9)], is_ir=False)
    # percy seen again while already in the stack -> does NOT jump to front
    ctrl.on_detection("c", [_Det("percy", 0.95)], is_ir=False)
    assert ctrl._queue == ["matcha", "percy"]


def test_detection_stack_prunes_decayed_on_render():
    import time
    ctrl = _controller(5)
    ctrl._boot_start = -100.0
    ctrl.on_detection("c", [_Det("percy", 0.9)], is_ir=False)
    # force percy's sighting to be old so it decays out
    s = ctrl._state.day["percy"]
    ctrl._state.day["percy"] = Sighting("percy", s.color, 0.9,
                                        last_seen=time.monotonic() - 999, first_seen=s.first_seen)
    ctrl._render(time.monotonic())  # pruning happens here
    assert ctrl._queue == []


def test_on_discovery_sets_fraction():
    ctrl = _controller()
    ctrl.on_discovery({
        "active": True, "order": ["a", "b", "c", "d"], "states": {},
        "counts": {"pending": 2, "testing": 0, "found": 1, "failed": 1},
    })
    assert ctrl._state.discovery_active
    assert ctrl._state.discovery_fraction == pytest.approx(0.5)


def test_render_priority_discovery_over_detection():
    ctrl = _controller(8)
    ctrl._boot_start = -100.0  # skip boot
    # both a day bird and an active discovery -> discovery wins
    ctrl.on_detection("c", [type("D", (), {"label": "draft", "confidence": 0.9})()], is_ir=False)
    ctrl.on_discovery({"active": True, "order": list("abcd"), "states": {},
                       "counts": {"pending": 0, "testing": 0, "found": 2, "failed": 2}})
    import time
    frame = ctrl._render(time.monotonic())
    # discovery (full bar) lights most LEDs; a single detection would light few
    assert sum(0 if c.is_dark() else 1 for c in frame) >= 6
