from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime

from lib.ai.client import OllamaUnavailableError
from lib.journal import load_entries, memory_jsonl_path
from lib.memory_maker import MemoryMaker


@dataclass
class FakeDetection:
    """Stand-in for lib.detector.Detection (label + confidence + bbox_xyxy)."""

    label: str
    confidence: float = 0.9
    bbox_xyxy: tuple = (1, 2, 3, 4)


class FakeNotifier:
    user_ids = ["A"]

    def __init__(self) -> None:
        self.tracked: list[str] = []
        self.albums: list = []
        self.edits: list[tuple] = []
        self.caption_edits: list[tuple] = []

    def broadcast_album_tracked(self, items):
        self.albums.append(items)
        return {"A": 100}

    def broadcast_text_tracked(self, text):
        self.tracked.append(text)
        return {"A": 100}

    def edit_message_text(self, chat_id, message_id, text):
        self.edits.append((chat_id, message_id, text))
        return True

    def edit_message_caption(self, chat_id, message_id, caption):
        self.caption_edits.append((chat_id, message_id, caption))
        return True


class FakeRegistry:
    def __init__(self, rows):
        self._rows = rows

    def snapshot(self):
        return list(self._rows)


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0
        self.messages: list = []

    def chat(self, model, messages, **kwargs):
        self.calls += 1
        self.messages.append(messages)
        return "Percy preened on the perch."


def row(label, camera="camera-192.168.1.8", since=1.0, movement=0.0):
    return {"camera": camera, "label": label, "since": since, "movement_percent": movement}


def _analyze_stub(img, labels):
    # Mirror analyze_frame's shape: a scene + one per-bird record per label.
    return {
        "scene": "perched",
        "birds": [{"label": b, "activity": "resting", "posture": "perched", "health": ""} for b in labels],
    }


def _maker(
    memories, registry, notifier, now_dt, clock_val,
    *,
    detect=lambda img: [FakeDetection("percy")],
    analyze=_analyze_stub,
    client=None,
):
    return MemoryMaker(
        memories,
        registry,
        grab_frame=lambda cam: b"\xff\xd8jpeg-" + cam.encode(),
        client=client or FakeClient(),
        llm_model="gemma3:4b",
        notifier=notifier,
        stop_event=threading.Event(),
        detect_frame=detect,
        analyze=analyze,
        detector_model="live-019.pt",
        vlm_model="qwen2.5vl:3b",
        interval_seconds=300,
        poll_seconds=30,
        fresh_seconds=15,
        camera_display=lambda n: "Big Cage",
        clock=clock_val if callable(clock_val) else (lambda: clock_val),
        now=lambda: now_dt,
    )


def test_report_saves_images_writes_memory_and_broadcasts(tmp_path) -> None:
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()
    maker = _maker(memories, FakeRegistry([row("percy")]), notifier, now_dt, 1000.0)

    assert maker._report({"percy": ["camera-192.168.1.8"]}) is True

    # One album sent (summary as first caption), image saved, memory written.
    assert notifier.albums and not notifier.tracked
    items = notifier.albums[0]
    assert items[0][1] and "Percy" in items[0][1]  # caption on the first item
    images = list((memories / "images").glob("*.jpg"))
    assert images, "expected a memory image to be saved"
    entries = load_entries(memories, now_dt.date())
    assert entries and entries[0].birds == ["percy"]
    assert entries[0].photos and entries[0].photos[0].endswith(".jpg")
    assert "Percy on Big Cage: perched" in entries[0].note
    assert entries[0].observations
    assert entries[0].observations[0].camera == "Big Cage"
    assert entries[0].observations[0].birds == ["percy"]
    assert entries[0].observations[0].note == "perched"


def test_tick_reports_on_new_bird(tmp_path) -> None:
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()
    maker = _maker(memories, FakeRegistry([row("percy")]), notifier, now_dt, 1000.0)
    maker._tick()
    assert notifier.albums


def test_tick_edits_in_place_when_quiet(tmp_path) -> None:
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 3, 0)
    notifier = FakeNotifier()
    maker = _maker(memories, FakeRegistry([]), notifier, now_dt, 1000.0)
    maker._last_report_at = 1000.0 - 400  # the 5-min beat is due
    maker._activity_msgs = {"A": 100}
    maker._tick()
    assert notifier.tracked == []
    assert notifier.edits and "quiet" in notifier.edits[0][2].lower()


def test_night_mode_refreshes_instead_of_reporting_when_quiet(tmp_path) -> None:
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 2, 0)
    notifier = FakeNotifier()
    maker = _maker(memories, FakeRegistry([row("percy")]), notifier, now_dt, 1000.0)
    maker._night_mode = lambda: True
    maker._reported_set = frozenset({"percy"})  # not a new bird
    maker._activity_msgs = {"A": 100}
    maker._last_report_at = 1000.0 - 10_000  # well past even the slow night beat
    maker._tick()
    # Resting at night with no new bird / motion -> just an in-place refresh.
    assert notifier.albums == []
    assert notifier.edits or notifier.caption_edits


def _raw_observations(memories, day):
    """Read the day's JSONL off disk and return the raw observation dicts."""
    lines = memory_jsonl_path(memories, day).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    assert records, "expected at least one journal record"
    obs = [o for rec in records for o in rec.get("observations", [])]
    assert obs, "expected at least one observation in the journal record"
    return obs


def test_vlm_model_not_stamped_when_analyze_raises(tmp_path) -> None:
    # THE KEY REGRESSION: when the VLM is down (analyze raises
    # OllamaUnavailableError) the observation must be written WITHOUT a
    # "vlm_model" key on disk — that empty marker is what the backfill scanner
    # looks for. Previously the VLM model was falsely stamped, so outage-era
    # entries looked "already done" and could never be repaired.
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()

    def _analyze_down(img, labels):
        raise OllamaUnavailableError("cluster down")

    maker = _maker(memories, FakeRegistry([row("percy")]), notifier, now_dt, 1000.0,
                   analyze=_analyze_down)
    assert maker._report({"percy": ["camera-192.168.1.8"]}) is True

    for obs in _raw_observations(memories, now_dt.date()):
        assert "vlm_model" not in obs  # empty vlm_model is omitted on disk
        assert obs["detector_model"] == "live-019.pt"  # detector provenance kept


def test_vlm_model_stamped_when_analyze_succeeds(tmp_path) -> None:
    # When the VLM actually ran and produced an analysis, the observation
    # carries the vlm_model stamp — backfill must NOT touch it.
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()
    maker = _maker(memories, FakeRegistry([row("percy")]), notifier, now_dt, 1000.0)
    assert maker._report({"percy": ["camera-192.168.1.8"]}) is True

    for obs in _raw_observations(memories, now_dt.date()):
        assert obs["vlm_model"] == "qwen2.5vl:3b"
        assert obs["detector_model"] == "live-019.pt"


def test_vlm_model_not_stamped_when_analyze_returns_empty(tmp_path) -> None:
    # A VLM pass that returns an EMPTY dict produced nothing usable — the
    # observation must stay unstamped (no "vlm_model" key) so backfill can
    # still decorate it later.
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()
    maker = _maker(memories, FakeRegistry([row("percy")]), notifier, now_dt, 1000.0,
                   analyze=lambda img, labels: {})
    assert maker._report({"percy": ["camera-192.168.1.8"]}) is True

    for obs in _raw_observations(memories, now_dt.date()):
        assert "vlm_model" not in obs
        assert obs["detector_model"] == "live-019.pt"


def test_leaving_ir_triggers_immediate_report(tmp_path) -> None:
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 6, 0)
    notifier = FakeNotifier()
    maker = _maker(memories, FakeRegistry([row("percy")]), notifier, now_dt, 1000.0)
    maker._was_night = True          # it was night...
    maker._night_mode = lambda: False  # ...and a camera just left IR (dawn/light)
    maker._reported_set = frozenset({"percy"})  # not new, but the wake-up reports anyway
    maker._tick()
    assert notifier.albums  # an immediate report on waking


def test_report_drops_shot_when_frame_shows_no_birds(tmp_path) -> None:
    # THE EMPTY-PHOTO BUG: the registry said Percy was around, but by the time
    # the frame was grabbed he had left — the detector finds nothing on it.
    # That photo must NOT be sent (it paired an empty floor with a caption
    # about birds), no memory written, and Percy stays unreported for a retry.
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()
    maker = _maker(memories, FakeRegistry([row("percy")]), notifier, now_dt, 1000.0,
                   detect=lambda img: [])

    assert maker._report({"percy": ["camera-192.168.1.8"]}) is False

    assert notifier.albums == [] and notifier.tracked == []
    assert not (memories / "images").exists() or not list((memories / "images").glob("*.jpg"))
    assert load_entries(memories, now_dt.date()) == []
    assert maker._reported_set == frozenset()  # still "new" -> retried next poll


def test_report_keeps_trigger_birds_when_detector_unavailable(tmp_path) -> None:
    # With NO detector wired we cannot verify the frame, so the old fallback
    # (claim the trigger birds) is the only honest option and must survive.
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()
    maker = _maker(memories, FakeRegistry([row("percy")]), notifier, now_dt, 1000.0,
                   detect=None)

    assert maker._report({"percy": ["camera-192.168.1.8"]}) is True
    entries = load_entries(memories, now_dt.date())
    assert entries and entries[0].birds == ["percy"]


def test_report_keeps_only_cameras_whose_frame_has_birds(tmp_path) -> None:
    # Two cameras trigger; only cam1's fresh frame still shows a bird. The
    # album must carry just cam1's shot, and only Percy is claimed/reported —
    # Matcha stays "new" so she is retried.
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()
    maker = _maker(
        memories, FakeRegistry([row("percy", "cam1"), row("matcha", "cam2")]),
        notifier, now_dt, 1000.0,
        detect=lambda img: [FakeDetection("percy")] if b"cam1" in img else [],
    )

    assert maker._report({"percy": ["cam1"], "matcha": ["cam2"]}) is True

    assert len(notifier.albums) == 1 and len(notifier.albums[0]) == 1
    entries = load_entries(memories, now_dt.date())
    assert entries and entries[0].birds == ["percy"]
    assert maker._reported_set == frozenset({"percy"})


def test_stub_notes_skip_llm_and_state_plain_facts(tmp_path) -> None:
    # When the VLM contributed nothing (down/skipped), every note is a "seen"
    # stub. Summarising those made gemma3 INVENT activity and pull absent
    # roster birds out of the pronoun note — so the LLM must not run at all;
    # the caption states the plain sighting instead.
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()
    client = FakeClient()
    maker = _maker(memories, FakeRegistry([row("percy")]), notifier, now_dt, 1000.0,
                   analyze=lambda img, labels: {}, client=client)

    assert maker._report({"percy": ["camera-192.168.1.8"]}) is True

    assert client.calls == 0  # no LLM pass over contentless notes
    caption = notifier.albums[0][0][1]
    assert "Percy seen on Big Cage." in caption


def test_summary_notes_carry_vlm_bird_activity(tmp_path) -> None:
    # The per-bird VLM activity must reach the summariser's notes (it used to
    # be discarded, leaving the LLM only names to embellish from).
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()
    maker = _maker(memories, FakeRegistry([row("percy")]), notifier, now_dt, 1000.0)

    assert maker._report({"percy": ["camera-192.168.1.8"]}) is True
    entries = load_entries(memories, now_dt.date())
    assert entries and "perched, Percy resting" in entries[0].note


def test_report_keeps_trigger_birds_when_detector_raises(tmp_path) -> None:
    # A detector ERROR is not a verified-empty frame — we can't tell whether the
    # bird is there, so the trigger-label fallback must survive (matching the
    # no-detector case), not silently drop the report.
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()

    def _detect_boom(img):
        raise RuntimeError("detector exploded")

    maker = _maker(memories, FakeRegistry([row("percy")]), notifier, now_dt, 1000.0,
                   detect=_detect_boom)
    assert maker._report({"percy": ["camera-192.168.1.8"]}) is True
    entries = load_entries(memories, now_dt.date())
    assert entries and entries[0].birds == ["percy"]


def test_dropped_bird_cools_down_no_duplicate_albums(tmp_path) -> None:
    # Percy verifies on cam1; Matcha stays registry-fresh but her frame keeps
    # verifying empty. Without a cooldown she re-triggers _report every 30s
    # poll, re-sending a near-identical Percy album (and duplicate journal
    # entries) each time. She must wait out a beat instead.
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()
    clock = [1000.0]
    maker = _maker(
        memories, FakeRegistry([row("percy", "cam1"), row("matcha", "cam2")]),
        notifier, now_dt, lambda: clock[0],
        detect=lambda img: [FakeDetection("percy")] if b"cam1" in img else [],
    )

    maker._tick()
    assert len(notifier.albums) == 1  # Percy reported, Matcha dropped + cooled

    for _ in range(4):  # the next few 30s polls must NOT re-report Percy
        clock[0] += 30.0
        maker._tick()
    assert len(notifier.albums) == 1
    assert len(load_entries(memories, now_dt.date())) == 1


def test_failed_report_still_refreshes_on_the_beat(tmp_path) -> None:
    # A registry-fresh bird whose frames never verify makes _report return
    # False every poll. The due beat must still fall back to the in-place
    # refresh — otherwise the "last updated" stamp freezes and the caretaker
    # looks dead.
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()
    maker = _maker(memories, FakeRegistry([row("percy")]), notifier, now_dt, 1000.0,
                   detect=lambda img: [])
    maker._activity_msgs = {"A": 100}
    maker._last_report_at = 1000.0 - 400  # the 5-min beat is due

    maker._tick()

    assert notifier.albums == []
    assert notifier.edits  # the beat was consumed by a refresh
    assert maker._last_report_at == 1000.0


def test_wake_report_retries_until_a_frame_verifies(tmp_path) -> None:
    # At lights-on the first grabbed frame often verifies empty (auto-exposure
    # settling). The one-shot wake edge must stay armed and retry next poll —
    # the birds are already in _reported_set from the night, so no other
    # trigger would ever deliver the dawn report.
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 6, 0)
    notifier = FakeNotifier()
    frames = {"n": 0}

    def _detect_flaky(img):
        frames["n"] += 1
        return [] if frames["n"] == 1 else [FakeDetection("percy")]

    maker = _maker(memories, FakeRegistry([row("percy")]), notifier, now_dt, 1000.0,
                   detect=_detect_flaky)
    maker._was_night = True
    maker._night_mode = lambda: False
    maker._reported_set = frozenset({"percy"})

    maker._tick()
    assert notifier.albums == []  # first post-IR frame was empty — nothing sent
    maker._tick()
    assert notifier.albums  # edge stayed armed; retry delivered the report


def test_mixed_report_keeps_stub_notes_away_from_llm(tmp_path) -> None:
    # cam1's shot has real VLM content, cam2's analyze produced nothing. The
    # stub must NOT ride along into the summariser (where gemma3 invents
    # activity for it) — it gets a plain factual bullet after the summary.
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()
    client = FakeClient()
    maker = _maker(
        memories, FakeRegistry([row("percy", "cam1"), row("matcha", "cam2")]),
        notifier, now_dt, 1000.0,
        detect=lambda img: [FakeDetection("percy")] if b"cam1" in img else [FakeDetection("matcha")],
        analyze=lambda img, labels: _analyze_stub(img, labels) if b"cam1" in img else {},
        client=client,
    )

    assert maker._report({"percy": ["cam1"], "matcha": ["cam2"]}) is True

    assert client.calls == 1
    llm_input = client.messages[0][-1]["content"]
    assert "Matcha" not in llm_input  # the stub never reached the LLM
    # The album goes out immediately with the header + stub bullet; the LLM
    # summary streams into the caption afterwards via caption edits.
    caption = notifier.albums[0][0][1]
    assert "• Matcha seen on Big Cage." in caption
    final_caption = notifier.caption_edits[-1][2]
    assert "Percy preened on the perch." in final_caption
    assert "• Matcha seen on Big Cage." in final_caption


def test_ghost_bird_beat_retry_never_duplicates_the_album(tmp_path) -> None:
    # Percy verifies; Matcha stays registry-fresh but never verifies (a ghost).
    # The due-beat retry must NOT re-send Percy's album or re-journal him when
    # the retry confirms nothing new — the edit-in-place refresh owns "still
    # the same".
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()
    clock = [1000.0]
    maker = _maker(
        memories, FakeRegistry([row("percy", "cam1"), row("matcha", "cam2")]),
        notifier, now_dt, lambda: clock[0],
        detect=lambda img: [FakeDetection("percy")] if b"cam1" in img else [],
    )

    maker._tick()
    assert len(notifier.albums) == 1  # Percy reported once

    for _ in range(20):  # 10 minutes of polls, spanning two due beats
        clock[0] += 30.0
        maker._tick()

    assert len(notifier.albums) == 1  # never re-sent
    assert len(load_entries(memories, now_dt.date())) == 1  # never re-journaled
    assert notifier.caption_edits  # the beat kept refreshing in place


def test_night_cooldown_matches_the_night_beat(tmp_path) -> None:
    # At night the beat slows by night_slowdown; the ghost cooldown must slow
    # with it, or the ghost re-triggers (and re-verifies the sleeping birds)
    # several times per night beat.
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 2, 0)
    notifier = FakeNotifier()
    maker = _maker(memories, FakeRegistry([row("percy")]), notifier, now_dt, 1000.0)
    maker._night_mode = lambda: True
    maker._cooldown({"matcha"})
    assert maker._drop_until["matcha"] == 1000.0 + 300 * 4.0  # night-scaled beat


def test_wake_retry_gives_up_after_the_limit(tmp_path) -> None:
    # A phantom bird at dawn (registry-fresh, frames never verify) must not pin
    # the wake edge forever: after WAKE_RETRY_LIMIT failed attempts the edge is
    # consumed, the pipeline stops churning, and the due refresh still runs.
    from lib.memory_maker import WAKE_RETRY_LIMIT

    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 6, 0)
    notifier = FakeNotifier()
    calls = {"n": 0}

    def _detect_none(img):
        calls["n"] += 1
        return []

    maker = _maker(memories, FakeRegistry([row("percy")]), notifier, now_dt, 1000.0,
                   detect=_detect_none)
    maker._was_night = True
    maker._night_mode = lambda: False
    maker._reported_set = frozenset({"percy"})
    maker._activity_msgs = {"A": 100}
    maker._last_report_at = 1000.0 - 400  # the beat is overdue throughout

    for _ in range(WAKE_RETRY_LIMIT + 5):
        maker._tick()

    assert notifier.albums == []
    assert calls["n"] == WAKE_RETRY_LIMIT + 1  # initial attempt + bounded retries
    assert notifier.edits  # the "last updated" stamp never froze


def test_wake_edge_survives_a_registry_blind_flip_tick(tmp_path) -> None:
    # The IR->color flip can blind the registry for a tick. The wake edge must
    # not be consumed by that empty tick — the birds are already in
    # _reported_set, so no other trigger would ever send the morning album.
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 6, 0)
    notifier = FakeNotifier()
    registry = FakeRegistry([])  # blind on the flip tick
    maker = _maker(memories, registry, notifier, now_dt, 1000.0)
    maker._was_night = True
    maker._night_mode = lambda: False
    maker._reported_set = frozenset({"percy"})

    maker._tick()
    assert notifier.albums == []

    registry._rows = [row("percy")]  # the registry catches up
    maker._tick()
    assert notifier.albums  # the dawn report still went out


def test_first_grab_miss_retries_next_poll(tmp_path) -> None:
    # A brand-new bird whose FIRST frame verifies empty (mid-flap) must be
    # retried on the very next poll — nothing was sent, so an eager retry
    # cannot duplicate anything, and "reports straight away" should hold.
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()
    clock = [1000.0]
    calls = {"n": 0}

    def _detect_flaky(img):
        calls["n"] += 1
        return [] if calls["n"] == 1 else [FakeDetection("percy")]

    maker = _maker(memories, FakeRegistry([row("percy")]), notifier, now_dt,
                   lambda: clock[0], detect=_detect_flaky)

    maker._tick()
    assert notifier.albums == []
    clock[0] += 30.0
    maker._tick()
    assert len(notifier.albums) == 1  # reported 30s later, not a beat later


def test_refresh_keeps_the_stamp_when_summary_is_long(tmp_path) -> None:
    # The stamp is the only part of a refresh that changes; when the summary
    # nears Telegram's 1024-char caption cap the BASE must be clipped, not the
    # stamp — otherwise every edit is byte-identical and silently rejected.
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()
    maker = _maker(memories, FakeRegistry([row("percy")]), notifier, now_dt, 1000.0)
    maker._last_summary = "x" * 1200
    maker._activity_since = now_dt
    maker._activity_msgs = {"A": 100}
    maker._last_is_caption = True

    maker._refresh(frozenset({"percy"}))

    caption = notifier.caption_edits[0][2]
    assert len(caption) <= 1024
    assert "last updated 15:00" in caption


def test_partially_confirmed_ghost_report_is_skipped(tmp_path) -> None:
    # Percy and Matcha are already reported and still around; a ghost label
    # trips the trigger while Matcha's frame happens to verify empty. The
    # attempt confirms nothing new (captured is a strict SUBSET of reported),
    # so it must be skipped — equality-only skipping shipped a duplicate
    # Percy-only album here.
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()
    maker = _maker(
        memories,
        FakeRegistry([row("percy", "cam1"), row("matcha", "cam2"), row("coco", "cam3")]),
        notifier, now_dt, 1000.0,
        detect=lambda img: [FakeDetection("percy")] if b"cam1" in img else [],
    )
    maker._reported_set = frozenset({"percy", "matcha"})

    maker._tick()

    assert notifier.albums == []
    assert maker._reported_set == frozenset({"percy", "matcha"})  # not shrunk


def test_departed_bird_still_updates_the_album(tmp_path) -> None:
    # Matcha genuinely left (gone from the registry). The due beat must send an
    # updated album so the composition reconciles — the subset skip only
    # applies while every reported bird is still around.
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()
    maker = _maker(memories, FakeRegistry([row("percy")]), notifier, now_dt, 1000.0)
    maker._reported_set = frozenset({"percy", "matcha"})
    maker._last_report_at = 1000.0 - 400  # the beat is due

    maker._tick()

    assert len(notifier.albums) == 1
    assert maker._reported_set == frozenset({"percy"})


def test_quiet_beat_resets_reported_so_returning_birds_report_again(tmp_path) -> None:
    # After a genuine "All quiet" beat the flock story ended: the SAME birds
    # returning hours later must produce a fresh report, not an eternal
    # "Still the same" edit on the morning's album.
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()
    registry = FakeRegistry([])
    maker = _maker(memories, registry, notifier, now_dt, 1000.0)
    maker._reported_set = frozenset({"percy"})
    maker._activity_msgs = {"A": 100}
    maker._last_report_at = 1000.0 - 400

    maker._tick()  # quiet beat
    assert notifier.edits and maker._reported_set == frozenset()

    registry._rows = [row("percy")]
    maker._tick()  # Percy returns
    assert len(notifier.albums) == 1


def test_night_motion_reports_once_per_beat_not_every_poll(tmp_path) -> None:
    # A restless bird satisfies the motion trigger on every 30s poll. The
    # night-fright report must go out once, then hold for a base beat — not
    # re-send a full album per poll for the whole restless spell.
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 2, 0)
    notifier = FakeNotifier()
    clock = [1000.0]
    maker = _maker(
        memories, FakeRegistry([row("percy", movement=50.0)]),
        notifier, now_dt, lambda: clock[0],
    )
    maker._night_mode = lambda: True
    maker._was_night = True
    maker._reported_set = frozenset({"percy"})

    maker._tick()
    assert len(notifier.albums) == 1  # the night fright is reported at once

    for _ in range(5):
        clock[0] += 30.0
        maker._tick()
    assert len(notifier.albums) == 1  # continued motion doesn't re-send

    clock[0] += 180.0  # past the base beat since the motion report
    maker._tick()
    assert len(notifier.albums) == 2  # a still-restless bird earns an update


def test_new_birds_camera_survives_the_max_cameras_cut(tmp_path) -> None:
    # Three busier cameras hold only already-reported birds; the newcomer sits
    # on a fourth. The camera cut must prioritise the camera with the
    # unreported bird, or the newcomer is skipped (and cooled) forever.
    memories = tmp_path / "memories"
    now_dt = datetime(2026, 6, 25, 15, 0)
    notifier = FakeNotifier()

    def _detect_by_cam(img):
        for marker, label in ((b"cam_a", "percy"), (b"cam_b", "matcha"),
                              (b"cam_c", "pizza"), (b"cam_z", "newbie")):
            if marker in img:
                return [FakeDetection(label)]
        return []

    maker = _maker(
        memories,
        FakeRegistry([row("percy", "cam_a"), row("matcha", "cam_b"),
                      row("pizza", "cam_c"), row("newbie", "cam_z")]),
        notifier, now_dt, 1000.0,
        detect=_detect_by_cam,
    )
    maker._reported_set = frozenset({"percy", "matcha", "pizza"})

    maker._tick()

    assert len(notifier.albums) == 1
    assert "newbie" in maker._reported_set


def test_contentless_vlm_analysis_is_not_stamped_as_done() -> None:
    # A VLM reply of {"scene": "", "birds": []} is truthy as a dict but says
    # NOTHING — stamping vlm_model on it would mark the observation "done"
    # undecorated, hiding it from every backfill. It must stay unmarked.
    from lib.memory_build import analysis_has_content, build_observation

    class Det:
        label = "percy"
        confidence = 0.9
        bbox_xyxy = (1, 2, 3, 4)

    assert analysis_has_content({"scene": "", "birds": []}) is False
    assert analysis_has_content({"scene": "on the perch", "birds": []}) is True
    assert analysis_has_content({"scene": "", "birds": [{"label": "percy"}]}) is True

    obs = build_observation(
        [Det()], camera="Cage", photo="p.jpg",
        analysis={"scene": "", "birds": []}, vlm_model="qwen2.5vl:7b",
    )
    assert obs.vlm_model == ""  # still needs backfill
