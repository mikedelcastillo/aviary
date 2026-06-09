"""Unit tests for ``aviary_immich.video`` — frame sampling logic and the video scan loop.

The cv2 I/O in ``sample_video_frames`` is thin glue (mirroring ``extract_frames.py``); the
selection/spacing decisions are factored into pure helpers tested here without cv2. The scan loop
is exercised with the dependency-free ``FakeImmichClient``/``FakeDetector`` and a monkeypatched
``sample_video_frames`` so no real video decode happens.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import aviary_immich.video as video
from aviary_immich.album_filing import AlbumTarget
from aviary_immich.records import aggregate_video_predictions
from aviary_immich.video import (
    predict_frames_adaptive,
    run_video_pipeline,
    sample_interval,
    select_frames,
    target_size,
)
from fakes import (
    FakeDetector,
    FakeImmichClient,
    OutOfMemoryError,
    make_asset,
    make_detection,
    make_prediction,
)


# --------------------------------------------------------------------------- sample_interval


def test_sample_interval_uses_every_seconds_floor_for_short_video():
    # 5s video, 30-frame budget -> duration/max = 0.167, but every=1.0 floor wins.
    assert sample_interval(5.0, 1.0, 30) == pytest.approx(1.0)


def test_sample_interval_spreads_budget_across_long_video():
    # 300s video, 30-frame budget -> 10s spacing beats the 1s floor so sampling reaches the end.
    assert sample_interval(300.0, 1.0, 30) == pytest.approx(10.0)


def test_sample_interval_unknown_duration_falls_back_to_every():
    assert sample_interval(0.0, 2.0, 30) == pytest.approx(2.0)


def test_sample_interval_zero_max_frames_falls_back_to_every():
    assert sample_interval(300.0, 2.0, 0) == pytest.approx(2.0)


# --------------------------------------------------------------------------- target_size


def test_target_size_none_when_within_cap():
    assert target_size(720, 1280, 1280) is None


def test_target_size_none_when_no_cap():
    assert target_size(2160, 3840, None) is None
    assert target_size(2160, 3840, 0) is None


def test_target_size_downscales_long_edge_to_cap():
    # 3840x2160 capped at 1280 -> scale 1/3 -> (1280, 720), returned as (width, height).
    assert target_size(2160, 3840, 1280) == (1280, 720)


def test_target_size_portrait_caps_height():
    assert target_size(3840, 2160, 1280) == (720, 1280)


# --------------------------------------------------------------------------- select_frames


def _reader(script):
    """Build a read_next() that yields (ok, frame, ts) from ``script`` then (False, ...)."""
    items = list(script)

    def read_next():
        if not items:
            return False, None, 0.0
        frame, ts = items.pop(0)
        return True, frame, ts

    return read_next


def test_select_frames_keeps_one_per_interval():
    # Frames at 0,0.5,1.0,1.5,2.0s, interval 1.0 -> keep t=0,1.0,2.0.
    script = [(f"f{i}", i * 0.5) for i in range(5)]
    frames = select_frames(_reader(script), max_frames=10, interval=1.0)
    assert frames == ["f0", "f2", "f4"]


def test_select_frames_respects_max_frames_cap():
    script = [(f"f{i}", float(i)) for i in range(10)]
    frames = select_frames(_reader(script), max_frames=3, interval=1.0)
    assert frames == ["f0", "f1", "f2"]


def test_select_frames_stops_at_stream_end():
    script = [("f0", 0.0), ("f1", 1.0)]
    frames = select_frames(_reader(script), max_frames=10, interval=1.0)
    assert frames == ["f0", "f1"]


def test_select_frames_zero_interval_keeps_every_frame_up_to_cap():
    script = [(f"f{i}", float(i)) for i in range(5)]
    frames = select_frames(_reader(script), max_frames=4, interval=0.0)
    assert frames == ["f0", "f1", "f2", "f3"]


# --------------------------------------------------------------------------- aggregate_video_predictions


def test_aggregate_video_unions_labels_across_frames(monkeypatch):
    import aviary_immich.state as state

    monkeypatch.setattr(state, "utc_now", lambda: "TS")
    asset = make_asset(id="v1", name="clip.mp4")
    predictions = [
        make_prediction(labels=["bird"], confidence=0.4),
        make_prediction(labels=["dog"], confidence=0.6),
    ]
    record = aggregate_video_predictions(asset, "acct", predictions, frames_scanned=2)
    assert record["decision"] == "match"
    assert record["labels"] == ["bird", "dog"]
    assert record["asset_type"] == "video"
    assert record["frames_scanned"] == 2
    assert record["asset_id"] == "v1"
    assert record["original_file_name"] == "clip.mp4"
    assert record["scanned_at"] == "TS"


def test_aggregate_video_keeps_highest_confidence_per_label(monkeypatch):
    import aviary_immich.state as state

    monkeypatch.setattr(state, "utc_now", lambda: "TS")
    asset = make_asset(id="v1")
    predictions = [
        make_prediction(detections=[make_detection("bird", confidence=0.3)]),
        make_prediction(detections=[make_detection("bird", confidence=0.9)]),
        make_prediction(detections=[make_detection("bird", confidence=0.5)]),
    ]
    record = aggregate_video_predictions(asset, "acct", predictions, frames_scanned=3)
    assert record["labels"] == ["bird"]
    assert len(record["detections"]) == 1
    assert record["detections"][0]["confidence"] == pytest.approx(0.9)
    assert record["max_confidence"] == pytest.approx(0.9)


def test_aggregate_video_no_detections_is_not_match(monkeypatch):
    import aviary_immich.state as state

    monkeypatch.setattr(state, "utc_now", lambda: "TS")
    record = aggregate_video_predictions(make_asset(id="v1"), "acct", [make_prediction(detections=[])], frames_scanned=1)
    assert record["decision"] == "not_match"
    assert record["labels"] == []
    assert record["max_confidence"] == 0.0


def test_aggregate_video_empty_predictions_is_not_match(monkeypatch):
    import aviary_immich.state as state

    monkeypatch.setattr(state, "utc_now", lambda: "TS")
    record = aggregate_video_predictions(make_asset(id="v1"), "acct", [], frames_scanned=0)
    assert record["decision"] == "not_match"
    assert record["frames_scanned"] == 0


# --------------------------------------------------------------------------- predict_frames_adaptive


def test_predict_frames_adaptive_empty_returns_empty():
    assert predict_frames_adaptive(FakeDetector(), [], 64) == []


def test_predict_frames_adaptive_single_pass_when_no_oom():
    detector = FakeDetector(responder=lambda items: [make_prediction(labels=["bird"]) for _ in items])
    predictions = predict_frames_adaptive(detector, ["a", "b", "c"], batch_size=64)
    assert len(predictions) == 3
    # One call, all three frames at once.
    assert detector.calls == [("arrays", 3, 3)]


def test_predict_frames_adaptive_chunks_by_batch_size():
    detector = FakeDetector(responder=lambda items: [make_prediction() for _ in items])
    predict_frames_adaptive(detector, ["a", "b", "c", "d", "e"], batch_size=2)
    # 5 frames, batch_size 2 -> chunks of 2, 2, 1.
    assert [n for _, n, _ in detector.calls] == [2, 2, 1]


def test_predict_frames_adaptive_halves_on_cuda_oom():
    def responder(items):
        if len(items) > 1:
            raise OutOfMemoryError("CUDA out of memory")
        return [make_prediction(labels=["bird"])]

    detector = FakeDetector(responder=responder)
    predictions = predict_frames_adaptive(detector, ["a", "b", "c", "d"], batch_size=64)
    assert len(predictions) == 4
    # Cap learned to 1 after halving 4 -> 2 -> 1.
    assert detector._infer_cap == 1


def test_predict_frames_adaptive_non_oom_propagates():
    def responder(items):
        raise ValueError("decode error")

    with pytest.raises(ValueError, match="decode error"):
        predict_frames_adaptive(FakeDetector(responder=responder), ["a", "b"], batch_size=64)


# --------------------------------------------------------------------------- run_video_pipeline


class _Appender:
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(record)


def _video_args(tmp_path, **overrides):
    base = dict(
        cache_dir=tmp_path / "cache",
        download_workers=4,
        inference_batch_size=8,
        video_transcoded=True,
        video_every_seconds=1.0,
        video_max_frames=4,
        video_max_edge=1280,
        dry_run=False,
        batch_size=100,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _targets():
    return [
        AlbumTarget(
            label="bird",
            album_name="Birds",
            album_id="alb-birds",
            album_ids=set(),
            manifested_ids=set(),
            write_manifest=lambda row: None,
        )
    ]


def _use_fake_download_client(monkeypatch, client):
    """Per-thread download clients are real ImmichClients by default; hand back the fake instead."""
    monkeypatch.setattr(video, "thread_local_client_factory", lambda base_url, api_key: (lambda: client))


def test_run_video_pipeline_files_matches_into_albums(monkeypatch, tmp_path):
    monkeypatch.setattr(video, "sample_video_frames", lambda *a, **k: ["frame0", "frame1"])
    client = FakeImmichClient(video_assets=[make_asset(id="v1", name="clip.mp4")])
    _use_fake_download_client(monkeypatch, client)
    detector = FakeDetector(responder=lambda items: [make_prediction(labels=["bird"]) for _ in items])
    appender = _Appender()
    state: dict = {}
    stats: dict = {}
    account = SimpleNamespace(slug="acct", api_key="key")
    targets = _targets()

    run_video_pipeline(
        [make_asset(id="v1", name="clip.mp4")],
        detector,
        client,
        "http://x/api",
        "key",
        account,
        targets,
        _video_args(tmp_path),
        state,
        appender,
        stats,
    )

    assert stats["videos"] == 1
    assert stats["birds"] == 1
    assert "v1" in state
    record = state["v1"]
    assert record["asset_type"] == "video"
    assert record["frames_scanned"] == 2
    # The video was filed into the Birds album pending batch.
    assert targets[0].pending == ["v1"]
    assert client.downloaded_videos == ["v1"]


def test_run_video_pipeline_records_error_when_sampling_fails(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise ValueError("bad codec")

    monkeypatch.setattr(video, "sample_video_frames", boom)
    client = FakeImmichClient(video_assets=[make_asset(id="v1")])
    _use_fake_download_client(monkeypatch, client)
    appender = _Appender()
    state: dict = {}
    stats: dict = {}
    account = SimpleNamespace(slug="acct", api_key="key")
    targets = _targets()

    run_video_pipeline(
        [make_asset(id="v1")],
        FakeDetector(),
        client,
        "http://x/api",
        "key",
        account,
        targets,
        _video_args(tmp_path),
        state,
        appender,
        stats,
    )

    assert stats["errors"] == 1
    assert stats["videos"] == 1
    assert state["v1"]["decision"] == "error"
    assert "bad codec" in state["v1"]["error"]
    assert targets[0].pending == []


def test_run_video_pipeline_deletes_temp_videos(monkeypatch, tmp_path):
    monkeypatch.setattr(video, "sample_video_frames", lambda *a, **k: ["frame0"])
    client = FakeImmichClient(video_assets=[make_asset(id="v1", name="clip.mov")])
    _use_fake_download_client(monkeypatch, client)
    account = SimpleNamespace(slug="acct", api_key="key")

    run_video_pipeline(
        [make_asset(id="v1", name="clip.mov")],
        FakeDetector(responder=lambda items: [make_prediction(detections=[]) for _ in items]),
        client,
        "http://x/api",
        "key",
        account,
        _targets(),
        _video_args(tmp_path),
        {},
        _Appender(),
        {},
    )

    video_dir = tmp_path / "cache" / "acct" / "videos"
    # Temp media must not linger after frames are decoded.
    assert not any(video_dir.glob("v1*")) if video_dir.exists() else True
