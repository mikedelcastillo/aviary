"""Catch up on LLM work that was missed while the cluster was unreachable.

Two kinds of debt accumulate during an Ollama outage, and this worker pays both
back the moment the cluster returns (a recovery listener pokes it; a slow poll
is the belt-and-braces):

  1. **Queued messages** (:mod:`lib.pending`): every free-text message saved by
     the router is replayed — classified and answered — oldest first, with a
     "catching up" prefix. A message that keeps failing is given up on after
     :data:`MESSAGE_MAX_ATTEMPTS`, WITH a notice to the user (never silence).

  2. **Undecorated memories**: journal observations written during the outage
     carry YOLO identity + the saved photo but no ``vlm_model`` (the VLM never
     ran). The memory pass re-runs the SAME structured VLM analysis the live
     pipeline uses — on the saved raw photo with its stored boxes redrawn — and
     swaps the upgraded observation back into the day file. Recall over an
     outage window then reads real activity ("preening on the perch") instead
     of a bare "seen".

Messages drain first (a person is waiting); memories fill in behind them, a
bounded batch per cycle so a long backlog never hogs the cluster against live
traffic (vision calls also queue behind the client's shared semaphore).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import requests

from lib.clock import now_ph
from lib.journal import load_day_records, observation_record, update_day_observations
from lib.memory_build import analysis_has_content, annotate_for_vlm, build_observation


LOGGER = logging.getLogger("lib.backfill")

POLL_SECONDS = 30.0
# Replay attempts per queued message before giving up (with a notice).
MESSAGE_MAX_ATTEMPTS = 3
# How far back the memory scan looks. None = the whole journal history: the
# scan walks newest-first and remembers days it found clean, so old fully
# decorated days cost nothing per cycle, while any undecorated backlog keeps
# the otherwise-idle VLM busy until the history is fully tagged. (This used to
# stop at 3 days and leave anything older to a manual memory-migrate run.)
MEMORY_BACKFILL_DAYS: int | None = None
# A day scanned clean stays skipped this long before it's re-checked — belt
# and braces for out-of-band edits (memory-migrate, hand fixes).
EXHAUSTED_DAY_RESCAN_SECONDS = 6 * 3600.0
# VLM re-analyses per cycle — bounds how long one pass can occupy the cluster.
MEMORY_BATCH_PER_CYCLE = 12
# Re-analysis attempts per photo (per process run) before it's parked.
PHOTO_MAX_ATTEMPTS = 3

GAVE_UP_NOTICE = (
    '⚠️ I kept failing to answer your saved message: "{text}" — giving up on that '
    "one. Ask me again and I'll take a fresh run at it."
)


class LlmBackfillWorker:
    def __init__(
        self,
        store,
        health,
        stop_event: threading.Event,
        *,
        replay: Callable[[int, str, object], None] | None = None,
        notify: Callable[[int, str], None] | None = None,
        memory_backfill: Callable[[], int] | None = None,
        poll_seconds: float = POLL_SECONDS,
        llm_model: str = "",
    ) -> None:
        self._store = store
        self._health = health
        self._stop = stop_event
        self._replay = replay
        self._notify = notify
        self._memory_backfill = memory_backfill
        self._poll = poll_seconds
        # The classify model: replays are pointless (and would burn give-up
        # attempts on 404s) while a partially-recovered cluster doesn't serve it.
        self._llm_model = llm_model
        self._poke = threading.Event()

    def start(self) -> None:
        # Wake immediately on recovery — queued users shouldn't also wait out a
        # poll interval. The first cycle runs promptly too, draining anything
        # queued before a restart.
        self._health.on_change(lambda up: self._poke.set() if up else None)
        self._poke.set()
        threading.Thread(target=self._run, name="llm-backfill", daemon=True).start()

    def _run(self) -> None:
        LOGGER.info(
            "LLM backfill worker on (pending=%d, memory pass %s)",
            self._store.count() if self._store is not None else 0,
            "on" if self._memory_backfill is not None else "off",
        )
        while not self._stop.is_set():
            self._poke.wait(self._poll)
            self._poke.clear()
            if self._stop.is_set():
                return
            if not self._health.available:
                continue
            try:
                self._cycle()
            except Exception:
                LOGGER.exception("Backfill cycle failed")

    def _cycle(self) -> None:
        if self._store is not None and self._replay is not None:
            self._drain_messages()
        if self._memory_backfill is not None and self._health.available:
            done = self._memory_backfill()
            if done:
                LOGGER.info("Backfilled VLM analysis for %d memory observation(s)", done)
                # A non-empty pass means there is probably more backlog behind
                # it — go straight into the next cycle instead of sleeping out
                # the poll, so an idle machine drains the history back-to-back.
                # An empty pass (or an outage pause) falls back to the normal
                # poll cadence.
                self._poke.set()

    def _drain_messages(self) -> None:
        if self._llm_model and not self._health.has_model(self._llm_model):
            LOGGER.info(
                "Pending messages wait: %s not served by the cluster yet", self._llm_model
            )
            return
        for item in self._store.items():
            if self._stop.is_set() or not self._health.available:
                return
            try:
                self._replay(item.chat_id, item.text, item.created)
            except requests.exceptions.ConnectionError as exc:
                # Transport-level failure (outage, gated call, busy timeout) —
                # not this message's fault. Keep it WITHOUT burning a give-up
                # attempt; the attempts counter is reserved for failures that
                # happen while the backend is demonstrably up (poison messages).
                LOGGER.warning("Replay of %s hit an outage (%s); pausing drain", item.id, exc)
                return
            except Exception as exc:
                updated = self._store.bump_attempts(item)
                if updated.attempts >= MESSAGE_MAX_ATTEMPTS:
                    LOGGER.error(
                        "Giving up on pending message %s after %d attempts",
                        item.id, updated.attempts,
                    )
                    if self._notify is not None:
                        try:
                            self._notify(item.chat_id, GAVE_UP_NOTICE.format(text=item.text[:200]))
                        except Exception:
                            LOGGER.exception("Give-up notice failed")
                    self._store.remove(item.id)
                    continue  # a later message may still be answerable
                LOGGER.warning(
                    "Replay of %s failed (%s); will retry (attempt %d/%d)",
                    item.id, exc, updated.attempts, MESSAGE_MAX_ATTEMPTS,
                )
                return  # stop the cycle; retry next poll
            self._store.remove(item.id)


@dataclass(frozen=True)
class _StoredDetection:
    """Adapter: a journal detection record shaped like lib.detector.Detection,
    so annotate()/build_observation() accept it (they read .label/.confidence/
    .bbox_xyxy). YOLO is NOT re-run for backfill — identity and boxes were
    already stored by the live pass; only the VLM decoration was missed."""

    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


def _stored_detections(obs: dict) -> list[_StoredDetection]:
    out: list[_StoredDetection] = []
    for det in obs.get("detections") or []:
        if not isinstance(det, dict) or not str(det.get("label", "")).strip():
            continue
        bbox = det.get("bbox") or (0, 0, 0, 0)
        try:
            bbox = tuple(int(v) for v in bbox)[:4]
        except (TypeError, ValueError):
            bbox = (0, 0, 0, 0)
        out.append(
            _StoredDetection(
                label=str(det["label"]),
                confidence=float(det.get("confidence", 0.0) or 0.0),
                bbox_xyxy=bbox if len(bbox) == 4 else (0, 0, 0, 0),
            )
        )
    return out


def _journal_days(memories_dir: Path) -> list[date]:
    """Every day with a journal file, newest first (from the filenames)."""
    days: list[date] = []
    try:
        for path in memories_dir.glob("*.jsonl"):
            try:
                days.append(date.fromisoformat(path.stem))
            except ValueError:
                continue
    except OSError:
        return []
    return sorted(days, reverse=True)


def build_memory_backfill(
    memories_dir: Path,
    analyze: Callable[[bytes, list[str]], dict],
    vlm_model: str,
    *,
    days_back: int | None = MEMORY_BACKFILL_DAYS,
    batch: int = MEMORY_BATCH_PER_CYCLE,
    now=now_ph,
    exhausted_rescan_seconds: float = EXHAUSTED_DAY_RESCAN_SECONDS,
) -> Callable[[], int]:
    """One bounded memory-backfill pass over the journal day files.

    Scans v3 entries for observations missing ``vlm_model`` (the durable "the
    VLM never ran" marker) that still have their photo + stored detections,
    re-runs the structured analysis on the boxed photo, and swaps the upgraded
    observation into the day file. Returns how many were upgraded. Photos that
    keep failing are parked for this process run; pre-v3 records are left to
    ``memory-migrate`` (they need YOLO re-detection, not just decoration).

    ``days_back=None`` walks the WHOLE history newest-first. Days that scan
    clean are remembered and skipped for ``exhausted_rescan_seconds`` (today
    and yesterday are always scanned — live writes land there), so a fully
    tagged history costs almost nothing per cycle while any backlog keeps the
    otherwise-idle VLM working through it.
    """
    attempts: dict[str, int] = {}
    exhausted: dict[date, float] = {}

    def _reprocess(obs: dict) -> dict | None:
        photo = str(obs.get("photo", ""))
        path = Path(photo)
        if not path.exists():
            attempts[photo] = PHOTO_MAX_ATTEMPTS
            return None
        stored = _stored_detections(obs)
        if not stored:
            attempts[photo] = PHOTO_MAX_ATTEMPTS
            return None
        # Boxes rendered directly at the VLM input size — the analyze callable
        # must not downscale again (main wires the drain to max_dim=None).
        boxed = annotate_for_vlm(path.read_bytes(), stored)
        analysis = analyze(boxed, [d.label for d in stored])
        if not analysis_has_content(analysis):
            # The VLM ran but said nothing usable — a contentless result must not
            # stamp vlm_model (it would mark the observation "done" undecorated).
            attempts[photo] = attempts.get(photo, 0) + 1
            return None
        upgraded = build_observation(
            stored,
            camera=str(obs.get("camera", "")),
            photo=photo,
            analysis=analysis,
            detector_model=str(obs.get("detector_model", "")),
            vlm_model=vlm_model,
        )
        return observation_record(upgraded)

    def run() -> int:
        done = 0
        today = now().date()
        yesterday = today - timedelta(days=1)
        if days_back is None:
            days = _journal_days(memories_dir)
        else:
            days = [today - timedelta(days=offset) for offset in range(days_back)]
        for day in days:
            if day not in (today, yesterday):
                stamp = exhausted.get(day)
                if stamp is not None and time.monotonic() - stamp < exhausted_rescan_seconds:
                    continue
            updates: dict[str, dict[str, dict]] = {}
            candidates = 0
            scanned_whole_day = False
            try:
                for record in load_day_records(memories_dir, day):
                    if record.get("version") != 3:
                        continue
                    for obs in record.get("observations") or []:
                        if done >= batch:
                            break
                        if not isinstance(obs, dict) or obs.get("vlm_model"):
                            continue
                        photo = str(obs.get("photo", ""))
                        if not photo or attempts.get(photo, 0) >= PHOTO_MAX_ATTEMPTS:
                            continue
                        candidates += 1
                        try:
                            upgraded = _reprocess(obs)
                        except requests.exceptions.ConnectionError as exc:
                            # Outage / gated call (down OR model unserved) /
                            # vision-slot timeout — cluster trouble, not this
                            # photo's fault: pause WITHOUT burning one of the
                            # photo's attempts.
                            LOGGER.info("Memory backfill paused: %s", exc)
                            return done  # the finally flushes what we have
                        except Exception:
                            attempts[photo] = attempts.get(photo, 0) + 1
                            LOGGER.exception("Memory backfill failed for %s", photo)
                            continue
                        if upgraded is not None:
                            updates.setdefault(str(record.get("time", "")), {})[photo] = upgraded
                            done += 1
                    if done >= batch:
                        break
                else:
                    scanned_whole_day = True
            finally:
                replaced = update_day_observations(memories_dir, day, updates)
                if replaced:
                    LOGGER.info("Backfilled %d observation(s) into %s", replaced, day.isoformat())
            # Only a full, candidate-free scan proves the day needs no work;
            # a batch-limited pass says nothing about what's left in it.
            if scanned_whole_day and candidates == 0:
                exhausted[day] = time.monotonic()
            if done >= batch:
                break
        return done

    return run
