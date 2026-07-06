from __future__ import annotations

from datetime import datetime

from lib.ai.usage import LlmUsageRecorder, format_usage


NOW = datetime(2026, 7, 4, 12, 0)


def _recorder(tmp_path) -> LlmUsageRecorder:
    return LlmUsageRecorder(tmp_path / "llm_usage.json", now=lambda: NOW)


def test_record_accumulates_tokens_calls_and_busy_time(tmp_path) -> None:
    rec = _recorder(tmp_path)
    rec.record("gemma3:4b", {"prompt_eval_count": 100, "eval_count": 20, "total_duration": 2_000_000_000})
    rec.record("gemma3:4b", {"prompt_eval_count": 50, "eval_count": 10, "total_duration": 1_000_000_000})
    usage = rec.snapshot()["models"]["gemma3:4b"]
    assert usage["calls"] == 2
    assert usage["prompt_tokens"] == 150
    assert usage["output_tokens"] == 30
    assert usage["busy_seconds"] == 3.0


def test_usage_persists_across_recorder_instances(tmp_path) -> None:
    # The whole point: /tokens must survive a server restart.
    _recorder(tmp_path).record("gemma3:4b", {"prompt_eval_count": 7, "eval_count": 3})
    reloaded = _recorder(tmp_path)
    assert reloaded.snapshot()["models"]["gemma3:4b"]["prompt_tokens"] == 7
    # The original "since" stamp is kept, not reset on reload.
    assert reloaded.snapshot()["since"] == NOW.isoformat(timespec="seconds")


def test_missing_metering_fields_count_as_zero(tmp_path) -> None:
    # A fully prompt-cached call can omit prompt_eval_count — never crash.
    rec = _recorder(tmp_path)
    rec.record("gemma3:4b", {})
    usage = rec.snapshot()["models"]["gemma3:4b"]
    assert usage["calls"] == 1 and usage["prompt_tokens"] == 0


def test_failures_are_counted_separately(tmp_path) -> None:
    rec = _recorder(tmp_path)
    rec.record_failure("gemma3:4b")
    usage = rec.snapshot()["models"]["gemma3:4b"]
    assert usage["failures"] == 1 and usage["calls"] == 0


def test_corrupt_usage_file_starts_fresh(tmp_path) -> None:
    (tmp_path / "llm_usage.json").write_text("{not json")
    rec = _recorder(tmp_path)
    assert rec.snapshot()["models"] == {}
    rec.record("gemma3:4b", {"eval_count": 1})  # and it can persist again
    assert _recorder(tmp_path).snapshot()["models"]["gemma3:4b"]["calls"] == 1


def test_format_usage_lists_models_and_totals(tmp_path) -> None:
    rec = _recorder(tmp_path)
    rec.record("gemma3:4b", {"prompt_eval_count": 1200, "eval_count": 240, "total_duration": 3_000_000_000})
    rec.record("qwen2.5vl:7b", {"prompt_eval_count": 900_000, "eval_count": 41_000, "total_duration": 90_000_000_000})
    rec.record_failure("gemma3:4b")
    text = format_usage(rec.snapshot())
    assert "gemma3:4b" in text and "qwen2.5vl:7b" in text
    assert "(+1 failed)" in text
    assert "900k" in text  # humanized token count
    assert text.startswith("🔤 LLM usage since ")
    assert "Total:" in text
    # The busiest model is listed first.
    assert text.index("qwen2.5vl:7b") < text.index("gemma3:4b")


def test_format_usage_empty_says_so(tmp_path) -> None:
    text = format_usage(_recorder(tmp_path).snapshot())
    assert "No LLM calls recorded yet" in text


def test_wrong_typed_counters_in_file_are_coerced_on_load(tmp_path) -> None:
    # A hand-edited file with string counters must not poison += or the /tokens
    # sort forever — values are coerced back to numbers on load.
    (tmp_path / "llm_usage.json").write_text(
        '{"since": "2026-07-04T10:00:00", "models": {"gemma3:4b": '
        '{"calls": "5", "failures": "1", "prompt_tokens": "100", '
        '"output_tokens": null, "busy_seconds": "2.5"}}}'
    )
    rec = _recorder(tmp_path)
    rec.record_failure("gemma3:4b")  # would TypeError on a string counter
    usage = rec.snapshot()["models"]["gemma3:4b"]
    assert usage == {
        "calls": 5, "failures": 2, "prompt_tokens": 100,
        "output_tokens": 0, "busy_seconds": 2.5,
    }
    assert "gemma3:4b" in format_usage(rec.snapshot())  # sorting works again
