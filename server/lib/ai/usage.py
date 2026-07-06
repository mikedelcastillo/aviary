"""Persistent per-model LLM usage accounting (tokens, calls, busy time).

Every successful Ollama response carries its own metering — ``prompt_eval_count``
(input tokens), ``eval_count`` (output tokens) and ``total_duration`` — but the
server used to drop those fields on the floor. The recorder accumulates them
per model and persists to one small JSON file, so ``/tokens`` can answer "how
much work has each model actually done" across restarts.

Recording is wired into :class:`~lib.ai.client.OllamaClient` at the single
``_post_json`` seam, so chat, intent, recall and vision calls are all counted
with no per-caller plumbing. Failed calls count too (attempts that exhausted
retries) — health-GATED calls do not (they never reached the backend).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path

from lib.clock import now_ph


LOGGER = logging.getLogger("lib.ai.usage")

_ZERO = {"calls": 0, "failures": 0, "prompt_tokens": 0, "output_tokens": 0, "busy_seconds": 0.0}


def _coerce(value, default):
    """``value`` as ``default``'s numeric type, or ``default`` if it won't convert."""
    try:
        return type(default)(value)
    except (TypeError, ValueError):
        return default


class LlmUsageRecorder:
    def __init__(self, path: Path, *, now=now_ph) -> None:
        self._path = Path(path)
        self._now = now
        self._lock = threading.Lock()
        self._data = self._load()

    # -- recording -------------------------------------------------------------

    def record(self, model: str, response: dict) -> None:
        """Accumulate one successful call's metering from the response body.

        Token fields can be absent (e.g. a fully prompt-cached call) — count
        what's there. ``total_duration`` is nanoseconds.
        """
        model = (model or "").strip() or "(unknown)"
        try:
            prompt = int(response.get("prompt_eval_count") or 0)
            output = int(response.get("eval_count") or 0)
            seconds = float(response.get("total_duration") or 0) / 1e9
        except (TypeError, ValueError):
            prompt = output = 0
            seconds = 0.0
        with self._lock:
            usage = self._data["models"].setdefault(model, dict(_ZERO))
            usage["calls"] += 1
            usage["prompt_tokens"] += prompt
            usage["output_tokens"] += output
            usage["busy_seconds"] = round(usage["busy_seconds"] + seconds, 3)
            self._flush()

    def record_failure(self, model: str) -> None:
        """One call that reached the backend but failed (retries exhausted)."""
        model = (model or "").strip() or "(unknown)"
        with self._lock:
            usage = self._data["models"].setdefault(model, dict(_ZERO))
            usage["failures"] += 1
            self._flush()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "since": self._data.get("since", ""),
                "models": {m: dict(u) for m, u in self._data["models"].items()},
            }

    # -- persistence -----------------------------------------------------------

    def _load(self) -> dict:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            models = data.get("models")
            if isinstance(models, dict):
                clean = {}
                for model, usage in models.items():
                    if isinstance(usage, dict):
                        # Coerce every counter to its proper type: a hand-edited
                        # value like "12" would otherwise poison += and sorting
                        # forever (the flush faithfully re-serializes it).
                        clean[str(model)] = {
                            k: _coerce(usage.get(k, v), v) for k, v in _ZERO.items()
                        }
                return {"since": str(data.get("since", "")), "models": clean}
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            LOGGER.warning("Unreadable LLM usage file %s; starting fresh", self._path)
        return {"since": self._now().isoformat(timespec="seconds"), "models": {}}

    def _flush(self) -> None:
        # Called with the lock held. One LLM call takes seconds, so a tiny
        # atomic write per call is nothing — and the counters survive a crash.
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(f"{self._path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError:
            LOGGER.exception("Persisting LLM usage failed")


def _fmt_count(value: int) -> str:
    """Humanize a token count: 1234 -> '1,234', 1234567 -> '1.2M'."""
    if value >= 10_000_000:
        return f"{value / 1_000_000:.0f}M"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 100_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:,}"


def _fmt_busy(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{seconds:.0f}s"


def format_usage(snapshot: dict) -> str:
    """The /tokens reply: per-model calls, tokens in/out, busy time, avg latency."""
    models = snapshot.get("models") or {}
    since = snapshot.get("since", "")
    try:
        since_text = datetime.fromisoformat(since).strftime("%b %d %H:%M")
    except (TypeError, ValueError):
        since_text = since or "?"
    if not models:
        return f"🔤 No LLM calls recorded yet (counting since {since_text})."
    lines = [f"🔤 LLM usage since {since_text}"]
    total_calls = total_in = total_out = 0
    total_busy = 0.0
    for model in sorted(models, key=lambda m: -models[m]["busy_seconds"]):
        u = models[model]
        total_calls += u["calls"]
        total_in += u["prompt_tokens"]
        total_out += u["output_tokens"]
        total_busy += u["busy_seconds"]
        failed = f" (+{u['failures']} failed)" if u["failures"] else ""
        avg = f" · avg {u['busy_seconds'] / u['calls']:.1f}s" if u["calls"] else ""
        lines.append(
            f"\n{model}\n"
            f"  {u['calls']:,} call{'s' if u['calls'] != 1 else ''}{failed} · "
            f"in {_fmt_count(u['prompt_tokens'])} / out {_fmt_count(u['output_tokens'])} tok\n"
            f"  busy {_fmt_busy(u['busy_seconds'])}{avg}"
        )
    lines.append(
        f"\nTotal: {total_calls:,} calls · in {_fmt_count(total_in)} / "
        f"out {_fmt_count(total_out)} tok · busy {_fmt_busy(total_busy)}"
    )
    return "\n".join(lines)
