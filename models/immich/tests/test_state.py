"""Tests for aviary_immich.state — file/CSV/JSONL persistence helpers."""

from __future__ import annotations

import csv
import json

import pytest

from aviary_immich.state import (
    CsvAppender,
    JsonlAppender,
    _decode_jsonl_records,
    _ensure_trailing_newline,
    append_csv,
    append_jsonl,
    load_jsonl_state,
    load_manifest_ids,
)


# --------------------------------------------------------------------------- _decode_jsonl_records


def test_decode_single_object():
    records = _decode_jsonl_records('{"asset_id": "a1"}')
    assert records == [{"asset_id": "a1"}]


def test_decode_two_concatenated_objects():
    line = '{"asset_id": "a1"}{"asset_id": "a2"}'
    records = _decode_jsonl_records(line)
    assert records == [{"asset_id": "a1"}, {"asset_id": "a2"}]


def test_decode_whitespace_between_objects_tolerated():
    line = '{"asset_id": "a1"}   {"asset_id": "a2"}'
    records = _decode_jsonl_records(line)
    assert records == [{"asset_id": "a1"}, {"asset_id": "a2"}]


def test_decode_skips_non_dict_top_level_values():
    # A bare JSON array and number are valid JSON but not dicts; they must be skipped
    # while the surrounding dict objects are still returned.
    line = '{"asset_id": "a1"}[1, 2]{"asset_id": "a2"}'
    records = _decode_jsonl_records(line)
    assert records == [{"asset_id": "a1"}, {"asset_id": "a2"}]


def test_decode_only_non_dict_returns_empty():
    assert _decode_jsonl_records("42") == []


# --------------------------------------------------------------------------- load_jsonl_state


def test_load_jsonl_state_missing_file_returns_empty(tmp_path):
    assert load_jsonl_state(tmp_path / "nope.jsonl") == {}


def test_load_jsonl_state_skips_blank_lines(tmp_path):
    path = tmp_path / "state.jsonl"
    path.write_text(
        '{"asset_id": "a1"}\n'
        "\n"
        "   \n"
        '{"asset_id": "a2"}\n',
        encoding="utf-8",
    )
    state = load_jsonl_state(path)
    assert set(state) == {"a1", "a2"}


def test_load_jsonl_state_dedup_last_wins(tmp_path):
    path = tmp_path / "state.jsonl"
    path.write_text(
        '{"asset_id": "a1", "decision": "skip"}\n'
        '{"asset_id": "a1", "decision": "match"}\n',
        encoding="utf-8",
    )
    state = load_jsonl_state(path)
    assert state["a1"]["decision"] == "match"


def test_load_jsonl_state_dedup_across_concatenated_records(tmp_path):
    path = tmp_path / "state.jsonl"
    # Two records on one physical line: the LAST occurrence still wins.
    path.write_text(
        '{"asset_id": "a1", "decision": "skip"}{"asset_id": "a1", "decision": "match"}\n',
        encoding="utf-8",
    )
    state = load_jsonl_state(path)
    assert state["a1"]["decision"] == "match"


def test_load_jsonl_state_skips_records_without_asset_id(tmp_path):
    path = tmp_path / "state.jsonl"
    path.write_text(
        '{"decision": "match"}\n'
        '{"asset_id": "", "decision": "match"}\n'
        '{"asset_id": "a1", "decision": "match"}\n',
        encoding="utf-8",
    )
    state = load_jsonl_state(path)
    assert set(state) == {"a1"}


# --------------------------------------------------------------------------- _ensure_trailing_newline


def test_ensure_trailing_newline_appends_when_missing(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text("hello", encoding="utf-8")
    _ensure_trailing_newline(path)
    assert path.read_bytes() == b"hello\n"


def test_ensure_trailing_newline_missing_file_untouched(tmp_path):
    path = tmp_path / "absent.jsonl"
    _ensure_trailing_newline(path)
    assert not path.exists()


def test_ensure_trailing_newline_empty_file_untouched(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_bytes(b"")
    _ensure_trailing_newline(path)
    assert path.read_bytes() == b""


def test_ensure_trailing_newline_already_terminated_unchanged(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text("hello\n", encoding="utf-8")
    _ensure_trailing_newline(path)
    assert path.read_bytes() == b"hello\n"


# --------------------------------------------------------------------------- JsonlAppender


def test_jsonl_appender_writes_sorted_keys_with_newline(tmp_path):
    path = tmp_path / "out.jsonl"
    with JsonlAppender(path) as appender:
        appender.write({"b": 2, "a": 1})
    assert path.read_text(encoding="utf-8") == '{"a": 1, "b": 2}\n'


def test_jsonl_appender_does_not_glue_records_when_no_trailing_newline(tmp_path):
    path = tmp_path / "out.jsonl"
    # Seed a file whose existing record has NO trailing newline.
    path.write_text('{"asset_id": "a1"}', encoding="utf-8")
    with JsonlAppender(path) as appender:
        appender.write({"asset_id": "a2"})
    # Both records must decode independently; gluing would corrupt the first line.
    state = load_jsonl_state(path)
    assert set(state) == {"a1", "a2"}


def test_jsonl_appender_write_outside_context_raises(tmp_path):
    appender = JsonlAppender(tmp_path / "out.jsonl")
    with pytest.raises(AssertionError):
        appender.write({"asset_id": "a1"})


def test_jsonl_appender_multiple_writes_each_on_own_line(tmp_path):
    path = tmp_path / "out.jsonl"
    with JsonlAppender(path) as appender:
        appender.write({"asset_id": "a1"})
        appender.write({"asset_id": "a2"})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == ['{"asset_id": "a1"}', '{"asset_id": "a2"}']


# --------------------------------------------------------------------------- CsvAppender


def test_csv_appender_writes_header_once_for_new_file(tmp_path):
    path = tmp_path / "m.csv"
    with CsvAppender(path, ["asset_id", "decision"]) as appender:
        appender.write({"asset_id": "a1", "decision": "match"})
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert rows == [{"asset_id": "a1", "decision": "match"}]
    # Header present exactly once.
    text = path.read_text(encoding="utf-8")
    assert text.count("asset_id,decision") == 1


def test_csv_appender_does_not_rewrite_header_on_reopen(tmp_path):
    path = tmp_path / "m.csv"
    with CsvAppender(path, ["asset_id", "decision"]) as appender:
        appender.write({"asset_id": "a1", "decision": "match"})
    with CsvAppender(path, ["asset_id", "decision"]) as appender:
        appender.write({"asset_id": "a2", "decision": "skip"})
    text = path.read_text(encoding="utf-8")
    assert text.count("asset_id,decision") == 1
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert [r["asset_id"] for r in rows] == ["a1", "a2"]


def test_csv_appender_projects_rows_missing_field_to_empty(tmp_path):
    path = tmp_path / "m.csv"
    with CsvAppender(path, ["asset_id", "decision", "extra"]) as appender:
        # "extra" omitted -> "", and unknown keys ignored.
        appender.write({"asset_id": "a1", "decision": "match", "ignored": "x"})
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert rows == [{"asset_id": "a1", "decision": "match", "extra": ""}]


def test_csv_appender_write_outside_context_raises(tmp_path):
    appender = CsvAppender(tmp_path / "m.csv", ["asset_id"])
    with pytest.raises(AssertionError):
        appender.write({"asset_id": "a1"})


# --------------------------------------------------------------------------- append_jsonl / append_csv


def test_append_jsonl_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "deep" / "out.jsonl"
    append_jsonl(path, {"asset_id": "a1"})
    assert path.exists()
    state = load_jsonl_state(path)
    assert set(state) == {"a1"}


def test_append_jsonl_sorts_keys_and_separates_records(tmp_path):
    path = tmp_path / "out.jsonl"
    append_jsonl(path, {"b": 2, "a": 1})
    append_jsonl(path, {"asset_id": "a2"})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == '{"a": 1, "b": 2}'
    assert json.loads(lines[1]) == {"asset_id": "a2"}


def test_append_jsonl_does_not_glue_when_seed_lacks_newline(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text('{"asset_id": "a1"}', encoding="utf-8")
    append_jsonl(path, {"asset_id": "a2"})
    state = load_jsonl_state(path)
    assert set(state) == {"a1", "a2"}


def test_append_csv_creates_parent_dirs_and_writes_header_once(tmp_path):
    path = tmp_path / "nested" / "deep" / "m.csv"
    append_csv(path, ["asset_id", "decision"], {"asset_id": "a1", "decision": "match"})
    append_csv(path, ["asset_id", "decision"], {"asset_id": "a2", "decision": "skip"})
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert text.count("asset_id,decision") == 1
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert [r["asset_id"] for r in rows] == ["a1", "a2"]


def test_append_csv_projects_missing_field_to_empty(tmp_path):
    path = tmp_path / "m.csv"
    append_csv(path, ["asset_id", "decision"], {"asset_id": "a1"})
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert rows == [{"asset_id": "a1", "decision": ""}]


# --------------------------------------------------------------------------- load_manifest_ids


def test_load_manifest_ids_missing_file_returns_empty_set(tmp_path):
    assert load_manifest_ids(tmp_path / "nope.csv") == set()


def test_load_manifest_ids_returns_id_set(tmp_path):
    path = tmp_path / "m.csv"
    append_csv(path, ["asset_id", "decision"], {"asset_id": "a1", "decision": "match"})
    append_csv(path, ["asset_id", "decision"], {"asset_id": "a2", "decision": "skip"})
    assert load_manifest_ids(path) == {"a1", "a2"}


def test_load_manifest_ids_skips_rows_missing_id(tmp_path):
    path = tmp_path / "m.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["asset_id", "decision"])
        writer.writeheader()
        writer.writerow({"asset_id": "", "decision": "match"})
        writer.writerow({"asset_id": "a1", "decision": "match"})
    assert load_manifest_ids(path) == {"a1"}


def test_load_manifest_ids_custom_id_field(tmp_path):
    path = tmp_path / "m.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "decision"])
        writer.writeheader()
        writer.writerow({"id": "x1", "decision": "match"})
        writer.writerow({"id": "x2", "decision": "skip"})
    assert load_manifest_ids(path, id_field="id") == {"x1", "x2"}
