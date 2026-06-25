from __future__ import annotations

from datetime import date, datetime

import pytest

from lib.sleep.model import Disturbance, SleepNight
from lib.sleep.score import (
    Baseline,
    consistency_score,
    darkness_score,
    disturbance_score,
    duration_score,
    rolling_baseline,
    score_night,
)

D = date(2026, 6, 25)


def _night(**kw) -> SleepNight:
    base = dict(night_of=D, camera_count_at_dark=2, camera_count_at_wake=2)
    base.update(kw)
    return SleepNight(**base)


@pytest.mark.parametrize(
    "hours,expected",
    [(11, 1.0), (10, 1.0), (12, 1.0), (14, 1.0), (15, 0.9), (9, 0.8), (8, 0.6), (7, 0.4), (6, 0.2), (3, 0.1), (0, 0.0)],
)
def test_duration_score_bands(hours, expected) -> None:
    assert duration_score(hours) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "dev,expected",
    [(0, 1.0), (30, 1.0), (45, 0.8), (60, 0.6), (90, 0.4), (120, 0.2), (180, 0.0)],
)
def test_consistency_deviation_bands(dev, expected) -> None:
    baseline = Baseline(lights_out_min=20 * 60, first_light_min=7 * 60)
    night = _night(
        lights_out=datetime(2026, 6, 25, 20, dev) if dev < 60 else datetime(2026, 6, 25, 20 + dev // 60, dev % 60),
        first_light=datetime(2026, 6, 26, 7, 0),  # wake exactly on baseline
    )
    # consistency = mean(out_dev_score, 1.0); solve back to the out_dev band score.
    out_only = 2 * consistency_score(night, baseline) - 1.0
    assert out_only == pytest.approx(expected, abs=1e-9)


def test_consistency_neutral_without_baseline() -> None:
    night = _night(lights_out=datetime(2026, 6, 25, 20, 40), first_light=datetime(2026, 6, 26, 7, 0))
    assert consistency_score(night, None) == 0.7


def test_rolling_baseline_uses_median_ignoring_outlier() -> None:
    prior = [
        SleepNight(night_of=date(2026, 6, d), lights_out=datetime(2026, 6, d, 20, 30),
                   first_light=datetime(2026, 6, d + 1, 7, 0), finalized=True)
        for d in (22, 23, 24)
    ]
    # An outlier night (very late) shouldn't move the median much.
    prior.append(SleepNight(night_of=date(2026, 6, 21), lights_out=datetime(2026, 6, 21, 23, 30),
                            first_light=datetime(2026, 6, 22, 9, 0), finalized=True))
    baseline = rolling_baseline(prior)
    assert baseline is not None
    assert baseline.lights_out_min == pytest.approx(20 * 60 + 30, abs=30)


def test_rolling_baseline_none_with_too_few_nights() -> None:
    assert rolling_baseline([]) is None
    one = [SleepNight(night_of=D, lights_out=datetime(2026, 6, 25, 20, 0), first_light=datetime(2026, 6, 26, 7, 0), finalized=True)]
    assert rolling_baseline(one) is None


def test_darkness_score_by_light_duration() -> None:
    assert darkness_score([]) == 1.0
    assert darkness_score([Disturbance(datetime(2026, 6, 25, 23, 0), "light", minutes=5)]) == pytest.approx(0.85)
    assert darkness_score([Disturbance(datetime(2026, 6, 25, 23, 0), "light", minutes=20)]) == pytest.approx(0.75)
    assert darkness_score([Disturbance(datetime(2026, 6, 25, 23, 0), "light", minutes=50)]) == pytest.approx(0.60)
    # Penalties stack and clamp at 0.
    many = [Disturbance(datetime(2026, 6, 25, 23, 0), "light", minutes=50) for _ in range(4)]
    assert darkness_score(many) == 0.0


def test_disturbance_score_frights_and_motion() -> None:
    assert disturbance_score([]) == 1.0
    assert disturbance_score([Disturbance(datetime(2026, 6, 25, 2, 0), "night_fright")]) == pytest.approx(0.60)
    assert disturbance_score([Disturbance(datetime(2026, 6, 25, 2, 0), "night_fright")] * 2) == pytest.approx(0.20)
    assert disturbance_score([Disturbance(datetime(2026, 6, 25, 2, 0), "motion")]) == pytest.approx(0.90)
    # A light event is NOT re-penalised here (it's in darkness_score).
    assert disturbance_score([Disturbance(datetime(2026, 6, 25, 23, 0), "light", minutes=20)]) == 1.0


# -- the four worked examples from the design spec ---------------------------

BASELINE = Baseline(lights_out_min=20 * 60 + 40, first_light_min=7 * 60 + 25)


def test_worked_example_a_good_night_97() -> None:
    night = _night(
        lights_out=datetime(2026, 6, 25, 20, 40), first_light=datetime(2026, 6, 26, 7, 25),
        dark_minutes=645,  # 10.75h
        disturbances=[Disturbance(datetime(2026, 6, 25, 23, 10), "light", minutes=5)],
    )
    score, components, _ = score_night(night, BASELINE)
    assert score == 97
    assert components == {"duration": 1.0, "consistency": 1.0, "darkness": 0.85, "disturbance": 1.0}


def test_worked_example_b_short_with_light_on_60() -> None:
    # Baseline bedtime 20:50 so tonight's 22:50 is exactly 2h late (out dev 0.2).
    baseline_b = Baseline(lights_out_min=20 * 60 + 50, first_light_min=7 * 60)
    night = _night(
        lights_out=datetime(2026, 6, 25, 22, 50), first_light=datetime(2026, 6, 26, 7, 0),
        dark_minutes=438,  # ~7.3h after excluding the lit span
        disturbances=[Disturbance(datetime(2026, 6, 25, 23, 0), "light", minutes=50)],
    )
    score, _, _ = score_night(night, baseline_b)
    assert score == 60


def test_worked_example_c_night_fright_94() -> None:
    night = _night(
        lights_out=datetime(2026, 6, 25, 20, 40), first_light=datetime(2026, 6, 26, 7, 40),
        dark_minutes=660,  # 11h
        disturbances=[Disturbance(datetime(2026, 6, 26, 2, 14), "night_fright", detail="possible")],
    )
    score, _, _ = score_night(night, BASELINE)
    assert score == 94


def test_worked_example_d_cold_start_92() -> None:
    night = _night(
        lights_out=datetime(2026, 6, 25, 20, 40), first_light=datetime(2026, 6, 26, 7, 40),
        dark_minutes=660,
    )
    score, components, confidence = score_night(night, None)  # no baseline
    assert score == 92
    assert components["consistency"] == 0.7
    assert confidence == pytest.approx(0.85)  # no-baseline hedge only
