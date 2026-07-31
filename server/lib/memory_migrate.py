"""Backfill the day-memory journal to the v3 structured-per-bird format.

``uv run memory-migrate`` re-processes every saved memory photo through the SAME
pipeline live capture uses (:mod:`lib.memory_build`): YOLO for identity + boxes,
the VLM for per-bird activity on the boxed frame. It rewrites each day's ``.jsonl``
in place (after a ``.bak`` backup) so old prose-only entries gain structured
detections, activity, and model/confidence provenance — making the whole history
queryable the same way new memories are.

Safety + control:
  * ``--dry-run``      process + report, write nothing.
  * ``--limit N``      stop after N photos (sample a few save types first).
  * ``--days N``       only the N most recent day files.
  * ``--force``        re-process even observations already at v3.
  * a ``<day>.jsonl.bak`` is written before any file is overwritten.

YOLO runs under a lock (one GPU); the VLM calls fan across the Olla cluster, so a
big backfill is VLM-bound, not GPU-bound. Progress + a final summary are printed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from lib.ai.client import OllamaClient
from lib.ai.vlm import analyze_frame
from lib.config import _ollama_config, build_config
from lib.detector import ObjectDetector
from lib.journal import MemoryDetection, MemoryObservation, observation_record
from lib.memory_build import annotate, build_observation


def _iter_day_files(memories_dir: Path, days: int | None, skip_today: bool) -> list[Path]:
    files = sorted(memories_dir.glob("*.jsonl"))
    files = [f for f in files if not f.name.endswith(".bak")]
    if skip_today:
        # The live server appends to today's file; rewriting it concurrently would
        # drop entries. New entries there are already v3 anyway.
        today = datetime.now().date().isoformat()
        files = [f for f in files if f.stem != today]
    return files[-days:] if days else files


def _load_raw_entries(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


class Migrator:
    def __init__(self, detector, client, vlm_model, detector_model, timeout=90.0):
        self._detector = detector
        self._client = client
        self._vlm_model = vlm_model
        self._detector_model = detector_model
        self._timeout = timeout
        self._yolo_lock = threading.Lock()

    def _detect(self, image_bytes: bytes):
        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return []
        with self._yolo_lock:  # one GPU — serialize the detector
            return self._detector.predict(frame)

    def reprocess(self, photo: str, camera: str) -> MemoryObservation | None:
        """Re-derive a v3 observation from a saved photo (YOLO + boxed-frame VLM)."""
        path = Path(photo)
        if not path.exists():
            return None
        image_bytes = path.read_bytes()
        detections = self._detect(image_bytes)
        analysis = None
        if detections:
            boxed = annotate(image_bytes, detections)
            try:
                analysis = analyze_frame(
                    self._client, self._vlm_model, boxed,
                    [d.label for d in detections], timeout_seconds=self._timeout,
                )
            except Exception as exc:  # keep identity even if the VLM is down
                print(f"    VLM failed for {path.name}: {str(exc)[:60]}", flush=True)
        return build_observation(
            detections, camera=camera, photo=photo, analysis=analysis,
            detector_model=self._detector_model, vlm_model=self._vlm_model,
        )


def _raw_observations(raw: dict) -> list[dict]:
    """The entry's observation dicts, synthesised from bare photos for legacy rows."""
    obs = list(raw.get("observations") or [])
    if not obs and raw.get("photos"):
        obs = [
            {"photo": p, "camera": "", "birds": raw.get("birds", []), "note": raw.get("note", "")}
            for p in raw["photos"]
        ]
    return obs


def _raw_to_observation(ro: dict) -> MemoryObservation:
    """A MemoryObservation carrying the observation's ORIGINAL data verbatim —
    used to PRESERVE it when re-detection finds nothing, so a frame the CPU
    detector misses on a single pass never loses its logged history. That
    includes any STORED detections: an outage-era v3 record (identity saved,
    VLM missing) must keep its YOLO boxes even when this pass can't re-detect."""
    return MemoryObservation(
        camera=str(ro.get("camera", "")),
        birds=[str(b).lower() for b in (ro.get("birds") or [])],
        note=str(ro.get("note", "")),
        photo=str(ro.get("photo", "")),
        detections=[
            MemoryDetection(
                label=str(d.get("label", "")),
                confidence=float(d.get("confidence", 0.0) or 0.0),
                bbox=tuple(d.get("bbox") or (0, 0, 0, 0)),
                activity=str(d.get("activity", "")),
                tags=[str(t) for t in (d.get("tags") or []) if str(t).strip()],
                posture=str(d.get("posture", "")),
                health=str(d.get("health", "")),
            )
            for d in (ro.get("detections") or [])
            if isinstance(d, dict)
        ],
        detector_model=str(ro.get("detector_model", "")),
        vlm_model=str(ro.get("vlm_model", "")),
    )


def _needs_migration(raw: dict, force: bool) -> bool:
    if force:
        return True
    if raw.get("version") != 3:
        return True
    return any(not o.get("vlm_model") for o in (raw.get("observations") or []))


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill memory to the v3 structured format")
    parser.add_argument("--dry-run", action="store_true", help="process but write nothing")
    parser.add_argument("--limit", type=int, default=0, help="stop after N photos (0 = all)")
    parser.add_argument("--days", type=int, default=0, help="only the N most recent day files")
    parser.add_argument("--force", action="store_true", help="re-process even v3 entries")
    # Skipping today is the DEFAULT: the live server appends to today's file and
    # this is a separate process — a concurrent rewrite here would silently drop
    # those appends. (The live server auto-backfills recent outage-era entries
    # itself; migrate is for deeper history.)
    parser.add_argument("--include-today", action="store_true",
                        help="also rewrite today's file (ONLY safe with the server stopped)")
    parser.add_argument("--workers", type=int, default=4, help="concurrent VLM calls")
    parser.add_argument("--device", default="cpu",
                        help="detector device: cpu (default, no GPU contention with a live "
                             "server) / cuda / auto")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    config = build_config()
    ocfg = _ollama_config()
    memories_dir = config.collect.directory.parent / "memories"
    model_cfg = dataclasses.replace(config.model, device=args.device) if args.device else config.model
    detector = ObjectDetector(model_cfg)
    client = OllamaClient(ocfg.base_url, timeout_seconds=ocfg.timeout_seconds,
                          vision_concurrency=max(1, args.workers))
    detector_model = ",".join(p.name for p in config.model.paths)
    migrator = Migrator(detector, client, ocfg.vlm_model, detector_model)

    day_files = _iter_day_files(memories_dir, args.days or None, not args.include_today)
    print(f"memory dir: {memories_dir}")
    print(f"day files: {len(day_files)}  vlm={ocfg.vlm_model}  detector={detector_model}  workers={args.workers}")
    print(f"mode: {'DRY-RUN' if args.dry_run else 'WRITE'}{'  FORCE' if args.force else ''}"
          f"{f'  LIMIT={args.limit}' if args.limit else ''}\n")

    processed = migrated_entries = skipped = failed = 0
    started = time.time()
    stop = False

    def flush(day_file, done_lines, raw_entries, changed):
        # Atomically rewrite the day file with migrated-so-far + not-yet-reached
        # originals, so a kill/timeout mid-day never loses progress or corrupts the
        # file (resume with a normal, non-force run skips the v3 entries already done).
        if args.dry_run or not changed:
            return
        remaining = [json.dumps(r) for r in raw_entries[len(done_lines):]]
        # PID-unique tmp name: the live server's backfill worker rewrites recent
        # day files too — sharing one tmp path could promote a half-written file.
        tmp = day_file.with_name(f"{day_file.name}.{os.getpid()}.tmp")
        tmp.write_text("\n".join(done_lines + remaining) + "\n", encoding="utf-8")
        tmp.replace(day_file)

    for day_file in day_files:
        if stop:
            break
        raw_entries = _load_raw_entries(day_file)
        # Back up ONCE before the first write — the .bak holds the pristine original.
        bak = day_file.with_suffix(".jsonl.bak")
        if not args.dry_run and not bak.exists():
            shutil.copy2(day_file, bak)
        new_lines: list[str] = []
        changed = 0
        for raw in raw_entries:
            if not _needs_migration(raw, args.force):
                skipped += 1
                new_lines.append(json.dumps(raw))
                continue
            new_obs: list[MemoryObservation] = []
            truncated = False
            for ro in _raw_observations(raw):
                # A mixed entry (some observations decorated, some not) only
                # needs work on the undecorated ones — re-running the detector
                # and a ~30s VLM call on an observation that already has its
                # decoration would re-buy what we own. --force re-does all.
                if not args.force and ro.get("vlm_model") and ro.get("detections"):
                    new_obs.append(_raw_to_observation(ro))
                    continue
                photo = str(ro.get("photo", ""))
                if args.limit and processed >= args.limit:
                    # Limit hit MID-entry: writing the partial new_obs would
                    # silently drop this entry's unprocessed observations — keep
                    # the original line; a later run migrates it whole.
                    stop = True
                    truncated = True
                    break
                upgraded = migrator.reprocess(photo, str(ro.get("camera", ""))) if photo else None
                if photo:
                    processed += 1
                if upgraded is not None and upgraded.detections:
                    new_obs.append(upgraded)
                    acts = ",".join(f"{d.label}:{d.activity or '?'}" for d in upgraded.detections)
                    print(f"  [{processed}] {Path(photo).name}: {acts}", flush=True)
                else:
                    # No detections (or photo gone) — keep the original record
                    # (incl. any stored detections) so its history survives;
                    # never overwrite history with an empty entry.
                    new_obs.append(_raw_to_observation(ro))
                    if photo and upgraded is None:
                        failed += 1
            if truncated:
                # Truncated by --limit: leave the entry exactly as it was so a
                # later run migrates it whole.
                new_lines.append(json.dumps(raw))
                if stop:
                    break
                continue
            # When nothing was re-detected the observations in new_obs are the
            # ORIGINALS, preserved verbatim (_raw_to_observation) — stamping
            # the record v3 loses no history, it just stops the entry counting
            # as forever-unfinished migration work. --force re-processes it if
            # a better detector pass is ever wanted.
            # Swap ONLY the observations; the entry-level time/birds/note/photos
            # stay byte-identical — they form the key load_entries uses to merge
            # the JSONL with the Markdown day file, so rewriting them would
            # DUPLICATE the entry on read-back (md orphan + jsonl extra).
            record = dict(raw)
            record["version"] = 3
            record["observations"] = [observation_record(o) for o in new_obs]
            new_lines.append(json.dumps(record))
            migrated_entries += 1
            changed += 1
            if changed % 15 == 0:  # periodic crash-safe checkpoint
                flush(day_file, new_lines, raw_entries, changed)
            if stop:
                break
        flush(day_file, new_lines, raw_entries, changed)  # final flush for the day
        if changed and not args.dry_run:
            print(f"  -> wrote {day_file.name} ({changed} entries migrated)", flush=True)

    dt = time.time() - started
    print(f"\n=== done in {dt:.0f}s ===")
    print(f"photos processed: {processed}  entries migrated: {migrated_entries}  "
          f"skipped(v3): {skipped}  failed(no photo/decode): {failed}")
    if processed:
        print(f"avg {dt/processed:.2f}s/photo")


if __name__ == "__main__":
    main()
