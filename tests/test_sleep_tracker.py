from __future__ import annotations

import threading
from datetime import datetime

from lib.sleep.model import load_current, load_recent
from lib.sleep.tracker import SleepTracker


class Env:
    """Drives a SleepTracker with an injected clock + fake IR/motion, no threads."""

    def __init__(self, sleep_dir, *, morning_report: bool = False) -> None:
        self.now = datetime(2026, 6, 25, 19, 0)
        self.dark = False
        self.cameras = 2
        self.move = 0.0
        self.sent: list[str] = []
        self.tracker = SleepTracker(
            self.sent.append,
            threading.Event(),
            all_ir=lambda: self.dark,
            camera_count=lambda: self.cameras,
            movement=lambda: self.move,
            sleep_dir=sleep_dir,
            now=lambda: self.now,
            morning_report=morning_report,
        )

    def at(self, dt: datetime) -> "Env":
        self.now = dt
        return self

    def ir(self) -> None:
        self.tracker.on_ir("cam", self.dark)

    def tick(self) -> None:
        self.tracker._tick(self.now)


def _full_night(env: Env) -> None:
    # Lights out 20:00.
    env.at(datetime(2026, 6, 25, 20, 0)); env.dark = True; env.ir()
    env.at(datetime(2026, 6, 25, 20, 6)).tick()  # confirm dark -> night opens
    # A brief light 23:00-23:05.
    env.at(datetime(2026, 6, 25, 23, 0)); env.dark = False; env.ir()
    env.at(datetime(2026, 6, 25, 23, 5)); env.dark = True; env.ir()
    # Wake 07:00, sustained.
    env.at(datetime(2026, 6, 26, 7, 0)); env.dark = False; env.ir()
    env.at(datetime(2026, 6, 26, 7, 11)).tick()  # confirm wake -> finalize


def test_full_night_finalizes_and_persists(tmp_path) -> None:
    env = Env(tmp_path)
    _full_night(env)

    nights = load_recent(tmp_path, 5)
    assert len(nights) == 1
    night = nights[0]
    assert night.finalized and night.score is not None
    assert night.lights_out == datetime(2026, 6, 25, 20, 0)
    assert night.first_light == datetime(2026, 6, 26, 7, 0)
    assert night.dark_minutes == 655  # 11h (660) minus the 5-min light
    # The in-progress sidecar is cleared on finalize.
    assert load_current(tmp_path) is None
    assert env.tracker.in_progress() is None


def test_morning_report_off_by_default(tmp_path) -> None:
    env = Env(tmp_path, morning_report=False)
    _full_night(env)
    assert env.sent == []


def test_morning_report_fires_once_when_enabled(tmp_path) -> None:
    env = Env(tmp_path, morning_report=True)
    _full_night(env)
    assert len(env.sent) == 1
    assert "Sleep report" in env.sent[0]


def test_restart_resumes_in_progress_night(tmp_path) -> None:
    env1 = Env(tmp_path)
    env1.at(datetime(2026, 6, 25, 20, 0)); env1.dark = True; env1.ir()
    env1.at(datetime(2026, 6, 25, 20, 6)).tick()  # night open, sidecar written
    assert env1.tracker.in_progress() is not None

    # A fresh tracker on the same dir resumes the open night.
    env2 = Env(tmp_path)
    resumed = env2.tracker.in_progress()
    assert resumed is not None
    assert resumed.lights_out == datetime(2026, 6, 25, 20, 0)
    assert resumed.partial_coverage is True  # restart loses some fidelity


def test_camera_drop_midnight_marks_partial_coverage(tmp_path) -> None:
    env = Env(tmp_path)
    env.at(datetime(2026, 6, 25, 20, 0)); env.dark = True; env.ir()
    env.at(datetime(2026, 6, 25, 20, 6)).tick()
    # A camera drops offline (membership shrinks) but the room stays dark.
    env.cameras = 1
    env.at(datetime(2026, 6, 25, 21, 0)); env.ir()
    assert env.tracker.in_progress().partial_coverage is True


def test_night_fright_recorded_and_scored(tmp_path) -> None:
    env = Env(tmp_path)
    env.at(datetime(2026, 6, 25, 20, 0)); env.dark = True; env.ir()
    env.at(datetime(2026, 6, 25, 20, 6)).tick()
    # A thrash in the dark (no concurrent light) at 2am.
    env.move = 45.0
    env.at(datetime(2026, 6, 26, 2, 0)).tick()
    env.move = 0.0
    env.at(datetime(2026, 6, 26, 7, 0)); env.dark = False; env.ir()
    env.at(datetime(2026, 6, 26, 7, 11)).tick()

    night = load_recent(tmp_path, 1)[0]
    assert any(d.kind == "night_fright" for d in night.disturbances)
    assert night.score < 100  # the fright dents the score
