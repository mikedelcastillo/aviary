from __future__ import annotations

import json
from datetime import date, datetime

from lib.journal import (
    MemoryDetection,
    MemoryEntry,
    MemoryObservation,
    append_entry,
    load_day_records,
    load_entries,
    load_recent,
    memory_jsonl_path,
    memory_path,
    observation_record,
    update_day_observations,
)


def test_append_creates_dated_file_with_header(tmp_path) -> None:
    entry = MemoryEntry(datetime(2026, 6, 25, 14, 32), ["percy", "matcha"], "Percy preens by Matcha.", ["a.jpg", "b.jpg"])
    path = append_entry(tmp_path, entry)
    assert path == memory_path(tmp_path, date(2026, 6, 25))
    text = path.read_text()
    assert text.startswith("# Aviary memories — 2026-06-25")
    assert "## 14:32 | percy, matcha" in text
    assert "> photo: a.jpg" in text and "> photo: b.jpg" in text


def test_append_then_load_roundtrip_multiple_photos(tmp_path) -> None:
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 9, 5), ["bambi"], "Bambi eats.", ["b1.jpg", "b2.jpg"]))
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 9, 40), ["percy"], "Percy naps."))
    entries = load_entries(tmp_path, date(2026, 6, 25))
    assert [e.time.strftime("%H:%M") for e in entries] == ["09:05", "09:40"]
    assert entries[0].photos == ["b1.jpg", "b2.jpg"]
    assert entries[1].note == "Percy naps."
    assert entries[1].photos == []


def test_append_writes_structured_jsonl_observations(tmp_path) -> None:
    append_entry(
        tmp_path,
        MemoryEntry(
            datetime(2026, 6, 25, 9, 5, 12),
            ["percy", "matcha"],
            "raw note",
            ["p.jpg"],
            observations=[
                MemoryObservation(
                    camera="Big Cage",
                    birds=["percy"],
                    note="Percy preened alone.",
                    photo="p.jpg",
                )
            ],
        ),
    )
    assert memory_jsonl_path(tmp_path, date(2026, 6, 25)).exists()
    entries = load_entries(tmp_path, date(2026, 6, 25))
    assert len(entries) == 1
    assert entries[0].time.second == 12
    assert entries[0].observations[0].camera == "Big Cage"
    assert entries[0].observations[0].birds == ["percy"]


def test_detections_round_trip_through_jsonl(tmp_path) -> None:
    # Enriched observations carry per-bird detections (label + confidence + bbox);
    # they must survive a write/read round-trip so the memory stays queryable.
    append_entry(
        tmp_path,
        MemoryEntry(
            datetime(2026, 6, 25, 9, 0),
            ["percy"],
            "note",
            ["p.jpg"],
            observations=[
                MemoryObservation(
                    camera="Big Cage",
                    birds=["percy"],
                    note="Percy on the perch.",
                    photo="p.jpg",
                    detections=[MemoryDetection(label="percy", confidence=0.83, bbox=(10, 20, 110, 220))],
                )
            ],
        ),
    )
    obs = load_entries(tmp_path, date(2026, 6, 25))[0].observations[0]
    assert len(obs.detections) == 1
    det = obs.detections[0]
    assert det.label == "percy"
    assert det.confidence == 0.83
    assert det.bbox == (10, 20, 110, 220)


def test_legacy_v1_observation_without_detections_still_loads(tmp_path) -> None:
    # A pre-enrichment record (no "detections" key) must load with an empty list,
    # never crash — back-compat for days logged before the schema bump.
    path = memory_jsonl_path(tmp_path, date(2026, 6, 25))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"version": 1, "time": "2026-06-25T09:00:00", "birds": ["percy"], "note": "n", '
        '"photos": ["p.jpg"], "observations": [{"camera": "Big Cage", "birds": ["percy"], '
        '"note": "n", "photo": "p.jpg"}]}\n',
        encoding="utf-8",
    )
    obs = load_entries(tmp_path, date(2026, 6, 25))[0].observations[0]
    assert obs.birds == ["percy"]
    assert obs.detections == []


def test_note_with_fake_header_does_not_corrupt_parsing(tmp_path) -> None:
    # A VLM note whose line starts with "## HH:MM | ..." must not be read back as
    # a separate (bogus) entry — the leading header hashes are stripped on write.
    append_entry(
        tmp_path,
        MemoryEntry(datetime(2026, 6, 25, 10, 0), ["percy"], "Percy near a sign:\n## 23:59 | draft, pizza"),
    )
    entries = load_entries(tmp_path, date(2026, 6, 25))
    assert len(entries) == 1  # one real entry, not split into two
    assert entries[0].time.strftime("%H:%M") == "10:00"
    assert entries[0].birds == ["percy"]
    assert "23:59 | draft, pizza" in entries[0].note  # text kept, header markers gone


def test_quiet_entry_has_no_birds(tmp_path) -> None:
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 3, 0), [], "All quiet."))
    entries = load_entries(tmp_path, date(2026, 6, 25))
    assert entries[0].birds == []


def test_load_recent_filters_by_time_and_bird(tmp_path) -> None:
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 8, 0), ["percy"], "early percy"))
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 12, 0), ["matcha"], "noon matcha"))
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 13, 0), ["percy"], "afternoon percy"))

    window = load_recent(tmp_path, datetime(2026, 6, 25, 11, 0), datetime(2026, 6, 25, 14, 0))
    assert [e.note for e in window] == ["noon matcha", "afternoon percy"]

    percy = load_recent(
        tmp_path, datetime(2026, 6, 25, 0, 0), datetime(2026, 6, 25, 23, 59), birds={"percy"}
    )
    assert [e.note for e in percy] == ["early percy", "afternoon percy"]


def test_load_missing_day_is_empty(tmp_path) -> None:
    assert load_entries(tmp_path, date(2026, 1, 1)) == []


def test_load_recent_spans_multiple_days(tmp_path) -> None:
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 20, 9, 0), ["jynx"], "monday"))
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 23, 9, 0), ["jynx"], "thursday"))
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 9, 0), ["jynx"], "saturday"))
    week = load_recent(tmp_path, datetime(2026, 6, 19, 0, 0), datetime(2026, 6, 25, 23, 59))
    assert [e.note for e in week] == ["monday", "thursday", "saturday"]


def _day_entries(day: date = date(2026, 6, 25)) -> list[MemoryEntry]:
    # Two entries, the first with two observations — the shape the backfill
    # worker sees: pre-v3 observations (no vlm_model) waiting to be upgraded.
    return [
        MemoryEntry(
            datetime(2026, 6, 25, 9, 5),
            ["percy", "matcha"],
            "morning check",
            ["a1.jpg", "a2.jpg"],
            observations=[
                MemoryObservation(camera="Big Cage", birds=["percy"], note="Percy perched.", photo="a1.jpg"),
                MemoryObservation(camera="Small Cage", birds=["matcha"], note="Matcha at the bowl.", photo="a2.jpg"),
            ],
        ),
        MemoryEntry(
            datetime(2026, 6, 25, 12, 0),
            ["bambi"],
            "noon check",
            ["b1.jpg"],
            observations=[
                MemoryObservation(camera="Big Cage", birds=["bambi"], note="Bambi eats.", photo="b1.jpg"),
            ],
        ),
    ]


def _upgraded_record() -> dict:
    # A post-VLM replacement observation record for a2.jpg, carrying provenance.
    return observation_record(
        MemoryObservation(
            camera="Small Cage",
            birds=["matcha"],
            note="Matcha preening at the bowl.",
            photo="a2.jpg",
            detections=[MemoryDetection(label="matcha", confidence=0.9, bbox=(1, 2, 3, 4), activity="preening")],
            detector_model="live-019.pt",
            vlm_model="qwen2.5vl:3b",
        )
    )


def test_observation_record_omits_empty_model_keys(tmp_path) -> None:
    # Pre-v3 (identity-only) observations must stay compact on disk: no
    # detector_model/vlm_model keys at all when empty — the backfill scanner
    # keys off the ABSENCE of vlm_model to find records to upgrade.
    record = observation_record(
        MemoryObservation(camera=" Big Cage ", birds=["Percy", "matcha", "percy "], note=" n ", photo=" p.jpg ")
    )
    assert record == {
        "camera": "Big Cage",
        "birds": ["matcha", "percy"],  # cleaned: lowercased, deduped, sorted
        "note": "n",
        "photo": "p.jpg",
        "detections": [],
    }
    assert "detector_model" not in record and "vlm_model" not in record


def test_observation_record_keeps_models_and_cleans_detections(tmp_path) -> None:
    # With provenance set, both model keys are present; detection behaviour
    # fields are cleaned and empty ones omitted from the per-detection dict.
    record = observation_record(
        MemoryObservation(
            camera="Big Cage",
            birds=["percy"],
            note="n",
            photo="p.jpg",
            detections=[
                MemoryDetection(label=" Percy ", confidence=0.8349, bbox=(1, 2, 3, 4), activity=" Preening "),
                MemoryDetection(label="   "),  # empty label → dropped
            ],
            detector_model="live-019.pt",
            vlm_model="qwen2.5vl:3b",
        )
    )
    assert record["detector_model"] == "live-019.pt"
    assert record["vlm_model"] == "qwen2.5vl:3b"
    assert record["detections"] == [
        {"label": "percy", "confidence": 0.835, "bbox": [1, 2, 3, 4], "activity": "preening"}
    ]


def test_load_day_records_returns_raw_dicts_and_skips_bad_lines(tmp_path) -> None:
    # The backfill scanner needs the RAW dicts (unknown fields intact, vlm_model
    # visibly absent), with malformed / non-dict lines skipped, never a crash.
    day = date(2026, 6, 25)
    good = {"version": 3, "time": "2026-06-25T09:00:00", "future_field": "kept", "observations": []}
    path = memory_jsonl_path(tmp_path, day)
    path.write_text(
        json.dumps(good) + "\n"
        + "{this is not json\n"
        + "[1, 2, 3]\n"  # valid JSON but not a dict → skipped
        + "\n"  # blank line → skipped
        + '{"time": "2026-06-25T10:00:00"}\n',
        encoding="utf-8",
    )
    records = load_day_records(tmp_path, day)
    assert records == [good, {"time": "2026-06-25T10:00:00"}]
    # Missing day file is an empty list, not an error.
    assert load_day_records(tmp_path, date(2026, 1, 1)) == []


def test_update_day_observations_swaps_only_the_matched_observation(tmp_path) -> None:
    # The (entry time ISO string, photo path) key must swap EXACTLY one
    # observation: sibling observations, the other entry's line (byte-identical),
    # and the touched entry's merge-key fields (time/birds/note/photos) all stay.
    day = date(2026, 6, 25)
    for entry in _day_entries():
        append_entry(tmp_path, entry)
    path = memory_jsonl_path(tmp_path, day)
    before = path.read_text(encoding="utf-8").splitlines()

    new_obs = _upgraded_record()
    replaced = update_day_observations(tmp_path, day, {"2026-06-25T09:05:00": {"a2.jpg": new_obs}})
    assert replaced == 1

    after = path.read_text(encoding="utf-8").splitlines()
    assert len(after) == len(before) == 2
    assert after[1] == before[1]  # untouched entry's line is byte-identical

    old_entry, new_entry = json.loads(before[0]), json.loads(after[0])
    # Entry-level fields (the md/jsonl merge key) are unchanged.
    for key in ("time", "birds", "note", "photos"):
        assert new_entry[key] == old_entry[key]
    # Sibling observation untouched, target swapped for the new record.
    assert new_entry["observations"][0] == old_entry["observations"][0]
    assert new_entry["observations"][1] == new_obs
    assert new_entry["observations"][1]["vlm_model"] == "qwen2.5vl:3b"


def test_update_day_observations_skips_already_upgraded_observation(tmp_path) -> None:
    # Defensive skip: an observation that already carries a vlm_model was
    # produced by a live (trusted) run — a stale backfill batch naming it must
    # NOT clobber it. No replacement → no rewrite at all.
    day = date(2026, 6, 25)
    append_entry(
        tmp_path,
        MemoryEntry(
            datetime(2026, 6, 25, 9, 5),
            ["percy"],
            "n",
            ["p.jpg"],
            observations=[
                MemoryObservation(camera="Big Cage", birds=["percy"], note="live note", photo="p.jpg", vlm_model="qwen2.5vl:3b")
            ],
        ),
    )
    path = memory_jsonl_path(tmp_path, day)
    before = path.read_text(encoding="utf-8")
    replaced = update_day_observations(
        tmp_path, day, {"2026-06-25T09:05:00": {"p.jpg": _upgraded_record()}}
    )
    assert replaced == 0
    assert path.read_text(encoding="utf-8") == before


def test_update_day_observations_keeps_malformed_line_verbatim(tmp_path) -> None:
    # A line we can't parse must survive the rewrite byte-for-byte — the
    # backfill must never destroy data it doesn't understand.
    day = date(2026, 6, 25)
    append_entry(tmp_path, _day_entries()[0])
    path = memory_jsonl_path(tmp_path, day)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{broken json, not: parseable\n")

    replaced = update_day_observations(tmp_path, day, {"2026-06-25T09:05:00": {"a2.jpg": _upgraded_record()}})
    assert replaced == 1  # a real swap happened, so the file WAS rewritten
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[1] == "{broken json, not: parseable"


def test_update_day_observations_noop_leaves_file_untouched(tmp_path) -> None:
    # Empty updates must not even open the file for writing (mtime + content
    # unchanged), and a missing day file returns 0 without creating anything.
    day = date(2026, 6, 25)
    append_entry(tmp_path, _day_entries()[0])
    path = memory_jsonl_path(tmp_path, day)
    before_text = path.read_text(encoding="utf-8")
    before_mtime = path.stat().st_mtime_ns

    assert update_day_observations(tmp_path, day, {}) == 0
    assert path.read_text(encoding="utf-8") == before_text
    assert path.stat().st_mtime_ns == before_mtime

    missing_day = date(2026, 1, 1)
    assert update_day_observations(tmp_path, missing_day, {"2026-01-01T09:00:00": {"x.jpg": _upgraded_record()}}) == 0
    assert not memory_jsonl_path(tmp_path, missing_day).exists()


def test_update_day_observations_preserves_md_jsonl_merge(tmp_path) -> None:
    # CRITICAL MERGE PIN: load_entries merges the md day file with the JSONL on
    # (minute, birds, note, photos). If the rewrite drifted any of those, the
    # upgraded entry would stop matching its md twin and show up TWICE. After an
    # update the day must load with the same entry count, no duplicates, and the
    # upgraded entry must carry the new activity/vlm_model.
    day = date(2026, 6, 25)
    for entry in _day_entries():
        append_entry(tmp_path, entry)
    assert len(load_entries(tmp_path, day)) == 2  # sanity: merged before update

    assert update_day_observations(tmp_path, day, {"2026-06-25T09:05:00": {"a2.jpg": _upgraded_record()}}) == 1

    entries = load_entries(tmp_path, day)
    assert len(entries) == 2  # still merged — no jsonl "extras" duplicating an entry
    assert [e.time.strftime("%H:%M") for e in entries] == ["09:05", "12:00"]
    upgraded = entries[0]
    assert upgraded.birds == ["matcha", "percy"]
    assert upgraded.note == "morning check"
    assert upgraded.photos == ["a1.jpg", "a2.jpg"]
    obs = {o.photo: o for o in upgraded.observations}
    assert obs["a2.jpg"].vlm_model == "qwen2.5vl:3b"
    assert obs["a2.jpg"].detections[0].activity == "preening"
    assert obs["a1.jpg"].vlm_model == ""  # sibling untouched
