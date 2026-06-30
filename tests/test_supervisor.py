from __future__ import annotations

import threading
import time
from pathlib import Path

from lib.config import (
    AppConfig,
    CameraCredentials,
    CollectConfig,
    DiscoveryConfig,
    FilterConfig,
    ModelConfig,
    TelegramConfig,
)
from lib.discovery import DiscoveredCamera, DiscoveryResult
from lib.objects import ObjectRegistry
from lib.supervisor import CameraSupervisor, format_discovery_report


def _app_config() -> AppConfig:
    """A minimal AppConfig — the supervisor only reads filter/discovery/credentials."""
    return AppConfig(
        snapshot_dir=Path("./data/server/snapshots"),
        model=ModelConfig(paths=(Path("model.pt"),)),
        telegram=TelegramConfig(enabled=False, bot_token="", user_ids=[]),
        collect=CollectConfig(objects=frozenset()),
        filter=FilterConfig(objects=frozenset()),
        credentials=CameraCredentials(username="user", password="pass"),
        discovery=DiscoveryConfig(hosts=("10.0.0.5", "10.0.0.6")),
    )


def _supervisor(monkeypatch) -> tuple[CameraSupervisor, dict, list]:
    """Build a supervisor with monitor_camera stubbed to a no-op (no capture)."""
    started: list[str] = []

    def fake_monitor(camera, *_args, **_kwargs):
        # Record that the thread ran for this host; do NOT touch any network.
        started.append(camera.host)

    monkeypatch.setattr("lib.supervisor.monitor_camera", fake_monitor)

    stats: dict = {}
    supervisor = CameraSupervisor(
        app_config=_app_config(),
        detector=object(),
        alert_state=object(),
        dispatcher=object(),
        registry=ObjectRegistry(),
        stats=stats,
        stats_lock=threading.Lock(),
        stop_event=threading.Event(),
    )
    return supervisor, stats, started


def test_discover_and_apply_starts_cameras_and_registers_stats(monkeypatch) -> None:
    supervisor, stats, _started = _supervisor(monkeypatch)

    result = DiscoveryResult(
        cameras=[
            DiscoveredCamera(
                host="10.0.0.5",
                port=554,
                stream_path="/stream1",
                rtsp_url="rtsp://user:pass@10.0.0.5:554/stream1",
            ),
            DiscoveredCamera(
                host="10.0.0.6",
                port=554,
                stream_path="/stream1",
                rtsp_url="rtsp://user:pass@10.0.0.6:554/stream1",
            ),
        ],
        hosts_scanned=2,
        ports_open=2,
        auth_failures=0,
        elapsed_seconds=0.4,
    )
    monkeypatch.setattr("lib.supervisor.discover_cameras", lambda *_a, **_k: result)

    applied = supervisor.discover_and_apply()

    assert sorted(applied.added) == ["camera-10.0.0.5", "camera-10.0.0.6"]
    assert applied.already_active == 0
    # Stats registered for each new camera under stable per-IP names.
    assert set(stats) == {"camera-10.0.0.5", "camera-10.0.0.6"}
    assert supervisor.active_hosts() == {"10.0.0.5", "10.0.0.6"}


def test_discover_and_apply_dedups_by_host(monkeypatch) -> None:
    supervisor, stats, _started = _supervisor(monkeypatch)

    camera = DiscoveredCamera(
        host="10.0.0.5",
        port=554,
        stream_path="/stream1",
        rtsp_url="rtsp://user:pass@10.0.0.5:554/stream1",
    )
    result = DiscoveryResult(
        cameras=[camera],
        hosts_scanned=2,
        ports_open=1,
        auth_failures=0,
        elapsed_seconds=0.2,
    )
    monkeypatch.setattr("lib.supervisor.discover_cameras", lambda *_a, **_k: result)

    first = supervisor.discover_and_apply()
    assert first.added == ["camera-10.0.0.5"]

    # Re-running discovery with the same host must NOT start a second thread or
    # add a second stats entry — dedup is keyed on the IP.
    second = supervisor.discover_and_apply()
    assert second.added == []
    assert second.already_active == 1
    assert list(stats) == ["camera-10.0.0.5"]
    assert supervisor.active_hosts() == {"10.0.0.5"}


def test_discover_and_apply_reconciles_stale_hosts_after_scan(monkeypatch) -> None:
    supervisor, stats, _started = _supervisor(monkeypatch)
    first_result = DiscoveryResult(
        cameras=[
            DiscoveredCamera(
                host="10.0.0.5",
                port=554,
                stream_path="/stream1",
                rtsp_url="rtsp://user:pass@10.0.0.5:554/stream1",
            ),
            DiscoveredCamera(
                host="10.0.0.6",
                port=554,
                stream_path="/stream1",
                rtsp_url="rtsp://user:pass@10.0.0.6:554/stream1",
            ),
        ],
        hosts_scanned=2,
        ports_open=2,
        auth_failures=0,
        elapsed_seconds=0.2,
    )
    second_result = DiscoveryResult(
        cameras=[
            DiscoveredCamera(
                host="10.0.0.6",
                port=554,
                stream_path="/stream1",
                rtsp_url="rtsp://user:pass@10.0.0.6:554/stream1",
            ),
            DiscoveredCamera(
                host="10.0.0.7",
                port=554,
                stream_path="/stream1",
                rtsp_url="rtsp://user:pass@10.0.0.7:554/stream1",
            ),
        ],
        hosts_scanned=2,
        ports_open=2,
        auth_failures=0,
        elapsed_seconds=0.2,
    )
    calls = 0

    def fake_discover(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            assert supervisor.active_hosts() == {"10.0.0.5", "10.0.0.6"}
            assert set(stats) == {"camera-10.0.0.5", "camera-10.0.0.6"}
        return first_result if calls == 1 else second_result

    monkeypatch.setattr("lib.supervisor.discover_cameras", fake_discover)

    first = supervisor.discover_and_apply()
    second = supervisor.discover_and_apply()

    assert sorted(first.added) == ["camera-10.0.0.5", "camera-10.0.0.6"]
    assert second.added == ["camera-10.0.0.7"]
    assert second.already_active == 1
    assert second.removed == ["camera-10.0.0.5"]
    assert supervisor.active_hosts() == {"10.0.0.6", "10.0.0.7"}
    assert set(stats) == {"camera-10.0.0.6", "camera-10.0.0.7"}


def test_discover_and_apply_serializes_overlapping_sweeps(monkeypatch) -> None:
    supervisor, _stats, _started = _supervisor(monkeypatch)
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_discover(*_args, **_kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return DiscoveryResult(
            cameras=[],
            hosts_scanned=2,
            ports_open=0,
            auth_failures=0,
            elapsed_seconds=0.02,
        )

    monkeypatch.setattr("lib.supervisor.discover_cameras", fake_discover)

    threads = [
        threading.Thread(target=supervisor.discover_and_apply),
        threading.Thread(target=supervisor.discover_and_apply),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert max_active == 1


def test_format_discovery_report_mentions_auth_failures() -> None:
    result = DiscoveryResult(
        cameras=[],
        hosts_scanned=254,
        ports_open=1,
        auth_failures=1,
        elapsed_seconds=1.5,
    )
    from lib.supervisor import DiscoveryApplied

    report = format_discovery_report(
        DiscoveryApplied(result=result, added=[], already_active=0)
    )

    assert "254 hosts" in report
    assert "TAPO_CREDENTIALS" in report
    assert "No new cameras started." in report
