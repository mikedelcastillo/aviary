from pathlib import Path

from bench_speed import sample_frames, summarize_latencies


def test_summarize_empty():
    summary = summarize_latencies([])
    assert summary["count"] == 0
    assert summary["p50_ms"] is None
    assert summary["imgs_per_sec"] is None


def test_summarize_single_value():
    summary = summarize_latencies([0.05])
    assert summary["count"] == 1
    assert summary["p50_ms"] == 50.0
    assert summary["p95_ms"] == 50.0
    assert summary["mean_ms"] == 50.0
    assert summary["imgs_per_sec"] == 20.0


def test_summarize_percentiles_ignore_order():
    seconds = [0.100, 0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090]
    summary = summarize_latencies(seconds)
    assert summary["count"] == 10
    assert summary["p50_ms"] == 55.0
    assert summary["p95_ms"] == 100.0
    assert summary["mean_ms"] == 55.0


def test_sample_frames_deterministic(tmp_path):
    for index in range(10):
        (tmp_path / f"frame_{index}.jpg").write_bytes(b"jpg")
    first = sample_frames(tmp_path, 4, seed=7)
    second = sample_frames(tmp_path, 4, seed=7)
    assert first == second
    assert len(first) == 4
    assert all(isinstance(path, Path) for path in first)


def test_sample_frames_small_dir_returns_all(tmp_path):
    (tmp_path / "one.jpg").write_bytes(b"jpg")
    assert len(sample_frames(tmp_path, 40)) == 1


def test_sample_frames_empty_dir(tmp_path):
    assert sample_frames(tmp_path, 5) == []
