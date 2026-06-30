"""Daily/weekly bird-care reminders, anchored to the sun and observed light.

Good care runs on a consistent rhythm: fresh food and water at wake-up, produce
pulled before it spoils, enrichment midday, a calm wind-down, a consistent
bedtime with 10-12 hours of dark, and a weekly deep-clean + weigh-in. This
scheduler sends those nudges, keyed to the flock's actual day:

  * Morning and bedtime fire on the OBSERVED light — the moment the cameras see
    the room light up (leave IR) or go dark (all IR), which is the real signal
    the birds wake/roost by — OR at local sunrise/sunset (see :mod:`lib.suntimes`),
    whichever comes first. The sun anchor means the nudge still arrives without a
    transition (no cameras, or a view that never changes IR).
  * The produce-pickup nudge fires ~2 hours after the morning one (food safety).
  * Midday enrichment, the pre-sunset wind-down, and the weekly clean/weigh fire
    on local PH time.

Each reminder fires at most once per day (or week), and only within a window
after its anchor, so a server that starts mid-evening never backfires a day's
worth of missed nudges at once. The logic in :meth:`_tick` is driven by an
injected ``now`` and light state, so it is fully unit-testable without real time.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable

from lib.care import FRESH_FOOD_MINUTES
from lib.clock import now_ph
from lib.suntimes import sun_times as default_sun_times

LOGGER = logging.getLogger("lib.care_scheduler")

# The reminder copy (warm, specific, grounded in the verified care research).
MESSAGES: dict[str, str] = {
    "morning": (
        "🌅 **Good morning!** Uncover the cage and let the flock wake with the light. "
        "Fresh pellets, clean water, and clear yesterday's leftovers."
    ),
    "food_pickup": (
        "🥗 Been about 2 hours — pull any leftover veggies and greens before they spoil."
    ),
    "midday": (
        "🧸 **Enrichment break!** Rotate in fresh foraging and shredding toys, and offer "
        "out-of-cage time — the cockatiels need a good 1–3 hours out."
    ),
    "winddown": (
        "🌇 **Winding down** — dim the lights and lower the noise so the flock can settle. "
        "They need 10–12h of uninterrupted dark, so keep bedtime consistent."
    ),
    "bedtime": (
        "🌙 **Lights out.** Cover the cage — leave a dim red nightlight for the cockatiels "
        "(prone to night frights) — and keep the room quiet."
    ),
    "weekly_clean": (
        "🧽 **Weekly deep-clean day:** wash perches, toys and dishes in hot water with "
        "bird-safe soap and replace the substrate. Six birds — don't let it slide."
    ),
    "weekly_weigh": (
        "⚖️ **Weigh-in day!** Each bird on the gram scale before breakfast, and log it — "
        "a ~10% drop is the early warning a bird needs a vet."
    ),
}


def _iso_week(day: date) -> str:
    iso = day.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _within(now_t: time, start: time, minutes: int) -> bool:
    """True if ``now_t`` is in ``[start, start + minutes]`` on the same day."""
    start_m = start.hour * 60 + start.minute
    now_m = now_t.hour * 60 + now_t.minute
    return start_m <= now_m <= start_m + minutes


class CareScheduler:
    """Sends daily/weekly care reminders on a background thread."""

    def __init__(
        self,
        notify_all: Callable[[str], None],
        stop_event: threading.Event,
        *,
        all_ir: Callable[[], bool] | None = None,
        camera_count: Callable[[], int] | None = None,
        sun_times: Callable[[date], tuple[time | None, time | None]] | None = None,
        now: Callable[[], datetime] = now_ph,
        poll_seconds: float = 60.0,
        morning_fallback: time = time(7, 0),
        bedtime_fallback: time = time(20, 0),
        midday: time = time(13, 0),
        window_minutes: int = 90,
        weekly_day: int = 6,  # Sunday (Mon=0 .. Sun=6)
        sleep_summary: Callable[[], str | None] | None = None,
        weekly_tip: Callable[[], str | None] | None = None,
        state_path: Path | None = None,
    ) -> None:
        self._notify_all = notify_all
        self._stop = stop_event
        self._all_ir = all_ir or (lambda: False)
        self._camera_count = camera_count or (lambda: 0)
        self._sun_times = sun_times or (lambda day: default_sun_times(day))
        self._now = now
        self._poll = poll_seconds
        self._morning_fallback = morning_fallback
        self._bedtime_fallback = bedtime_fallback
        self._midday = midday
        self._window = window_minutes
        self._weekly_day = weekly_day
        # Optional providers folded into a reminder: last night's sleep one-liner
        # appended to the morning nudge (when finalized), and a rotating care tip
        # appended to the weekly deep-clean nudge.
        self._sleep_summary = sleep_summary
        self._weekly_tip = weekly_tip

        self._fired: dict[str, str] = {}  # reminder key -> period key it last fired for
        self._was_dark: bool | None = None
        self._morning_fired_at: datetime | None = None
        # Persist the dedup state so a restart mid-window (a crash, /restart, or
        # boot autostart) doesn't re-send a reminder that already went out today.
        self._state_path = state_path
        self._load_state()

    def _load_state(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            LOGGER.exception("Could not read care state %s", self._state_path)
            return
        fired = data.get("fired")
        if isinstance(fired, dict):
            self._fired = {str(k): str(v) for k, v in fired.items()}
        if isinstance(data.get("was_dark"), bool):
            self._was_dark = data["was_dark"]
        stamp = data.get("morning_fired_at")
        if isinstance(stamp, str):
            try:
                self._morning_fired_at = datetime.fromisoformat(stamp)
            except ValueError:
                pass

    def _save_state(self) -> None:
        if self._state_path is None:
            return
        payload = {
            "fired": self._fired,
            "was_dark": self._was_dark,
            "morning_fired_at": (
                self._morning_fired_at.isoformat() if self._morning_fired_at else None
            ),
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(self._state_path)
        except OSError:
            LOGGER.exception("Could not write care state %s", self._state_path)

    def start(self) -> None:
        threading.Thread(target=self._run, name="care-scheduler", daemon=True).start()

    def _run(self) -> None:
        LOGGER.info("Care scheduler started (poll %.0fs)", self._poll)
        while not self._stop.is_set():
            if self._stop.wait(self._poll):
                break
            try:
                self._tick(self._now())
            except Exception:
                LOGGER.exception("Care scheduler tick failed")

    # -- firing bookkeeping ------------------------------------------------

    def _due(self, key: str, period: str) -> bool:
        return self._fired.get(key) != period

    def _fire(self, key: str, period: str) -> None:
        self._fired[key] = period
        self._save_state()
        try:
            self._notify_all(self._compose(key))
        except Exception:
            LOGGER.exception("Care reminder %s failed to send", key)
        LOGGER.info("Care reminder sent: %s", key)

    def _compose(self, key: str) -> str:
        """The reminder text, with the optional sleep/tip extra folded in."""
        message = MESSAGES[key]
        provider = None
        if key == "morning":
            provider = self._sleep_summary
        elif key == "weekly_clean":
            provider = self._weekly_tip
        if provider is not None:
            try:
                extra = provider()
            except Exception:
                LOGGER.exception("Care reminder extra for %s failed", key)
                extra = None
            if extra:
                message = f"{message}\n\n{extra}"
        return message

    # -- the schedule ------------------------------------------------------

    def _tick(self, now: datetime) -> None:
        today = now.date()
        day_key = today.isoformat()
        week_key = _iso_week(today)
        now_t = now.time()

        # Observed-light transitions drive the morning/bedtime nudges, but ONLY
        # when cameras are actually reporting. Losing all coverage (privacy mode
        # forgets each camera's IR vote, or every camera drops at once) must NOT
        # be read as the lights changing — hold the last known state instead, so a
        # coverage gap can't fire a spurious "good morning" / "lights out".
        lights_came_on = False
        lights_went_off = False
        if self._camera_count() > 0:
            dark = self._all_ir()
            lights_came_on = self._was_dark is True and not dark
            lights_went_off = self._was_dark is False and dark
            self._was_dark = dark

        sunrise, sunset = self._sun_times(today)
        morning_anchor = sunrise or self._morning_fallback
        bedtime_anchor = sunset or self._bedtime_fallback
        winddown_anchor = _minus(sunset, 60) if sunset else _minus(self._bedtime_fallback, 60)

        # Morning: the room lighting up is the real wake signal; otherwise fall
        # back to the sunrise/clock anchor. Guarded to daytime hours so a stray
        # IR flicker at 3am can't trigger "good morning".
        if self._due("morning", day_key):
            if (lights_came_on and time(4, 0) <= now_t <= time(11, 0)) or _within(now_t, morning_anchor, self._window):
                self._fire("morning", day_key)
                self._morning_fired_at = now
                self._save_state()  # persist the stamp so produce-pickup survives a restart

        # Produce pickup: ~2h after the morning nudge fired (food safety).
        if self._fired.get("morning") == day_key and self._due("food_pickup", day_key):
            if self._morning_fired_at is not None and (now - self._morning_fired_at) >= timedelta(minutes=FRESH_FOOD_MINUTES):
                self._fire("food_pickup", day_key)

        # Midday enrichment (clock-based, only during the day).
        if self._due("midday", day_key) and _within(now_t, self._midday, self._window):
            self._fire("midday", day_key)

        # Wind-down ~1h before sunset/target bedtime. Its window is capped below
        # the 60-min offset so it always lands BEFORE sunset/bedtime, never after
        # (a "winding down ~1h before sunset" nudge must not fire post-sunset).
        if self._due("winddown", day_key) and _within(now_t, winddown_anchor, min(self._window, 55)):
            self._fire("winddown", day_key)

        # Bedtime: the room going dark is the real signal; otherwise the sunset/
        # clock anchor. Guarded to evening so it can't fire just after midnight.
        if self._due("bedtime", day_key):
            if (lights_went_off and now_t >= time(16, 0)) or _within(now_t, bedtime_anchor, self._window):
                self._fire("bedtime", day_key)

        # Weekly deep-clean (on the weekly day, around midday).
        if now.weekday() == self._weekly_day and self._due("weekly_clean", week_key):
            if _within(now_t, self._midday, self._window):
                self._fire("weekly_clean", week_key)

        # Weekly weigh-in (on the weekly day, once the morning routine has run).
        if now.weekday() == self._weekly_day and self._due("weekly_weigh", week_key):
            if self._fired.get("morning") == day_key:
                self._fire("weekly_weigh", week_key)


def _minus(t: time, minutes: int) -> time:
    """``t`` shifted earlier by ``minutes`` (clamped within the same day)."""
    total = max(0, (t.hour * 60 + t.minute) - minutes)
    return time(total // 60, total % 60)
