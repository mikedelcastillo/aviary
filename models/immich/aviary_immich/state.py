"""State and manifest helpers for resumable Immich scans."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl_state(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records

    for line in path.read_text(encoding="utf-8").splitlines():
        raw_line = line.strip()
        if not raw_line:
            continue
        for record in _decode_jsonl_records(raw_line):
            asset_id = str(record.get("asset_id", ""))
            if asset_id:
                records[asset_id] = record
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_trailing_newline(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def append_csv(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


class JsonlAppender:
    """Keep a JSONL file open across many writes instead of reopening per record."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = None

    def __enter__(self) -> "JsonlAppender":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _ensure_trailing_newline(self.path)
        self._handle = self.path.open("a", encoding="utf-8")
        return self

    def write(self, record: dict[str, Any]) -> None:
        assert self._handle is not None, "JsonlAppender used outside of its context"
        self._handle.write(json.dumps(record, sort_keys=True))
        self._handle.write("\n")

    def __exit__(self, *exc: Any) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None


class CsvAppender:
    """Keep a CSV manifest open across many writes, writing the header once if new."""

    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self._handle = None
        self._writer = None

    def __enter__(self) -> "CsvAppender":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        self._handle = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames)
        if not exists:
            self._writer.writeheader()
        return self

    def write(self, row: dict[str, Any]) -> None:
        assert self._writer is not None, "CsvAppender used outside of its context"
        self._writer.writerow({field: row.get(field, "") for field in self.fieldnames})

    def __exit__(self, *exc: Any) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None
            self._writer = None


def load_manifest_ids(path: Path, id_field: str = "asset_id") -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row[id_field] for row in csv.DictReader(handle) if row.get(id_field)}


def _decode_jsonl_records(line: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    records: list[dict[str, Any]] = []
    index = 0
    while index < len(line):
        record, index = decoder.raw_decode(line, index)
        if isinstance(record, dict):
            records.append(record)
        while index < len(line) and line[index].isspace():
            index += 1
    return records


def _ensure_trailing_newline(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as handle:
        handle.seek(-1, 2)
        if handle.read(1) != b"\n":
            handle.write(b"\n")
