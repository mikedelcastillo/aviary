"""Camera watchlist: MAC-keyed registry persistence and supervisor enforcement
(only watchlisted cameras stream; allow/remove act immediately)."""

from __future__ import annotations

import json
import threading
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
from lib.watchlist import CameraRegistry

MAC_A = "cc:ba:bd:9a:ef:51"
MAC_B = "ac:a7:f1:34:3f:8c"


# --- CameraRegistry ----------------------------------------------------------


def test_registry_records_and_persists(tmp_path) -> None:
    path = tmp_path / "camera_registry.json"
    registry = CameraRegistry(cache_path=path)
    registry.record_sighting("CC-BA-BD-9A-EF-51", "10.0.0.5", now=100.0)
    registry.allow(MAC_A)

    # A fresh instance reloads both the camera cache and the allowlist.
    reloaded = CameraRegistry(cache_path=path)
    assert reloaded.ip_for_mac(MAC_A) == "10.0.0.5"
    assert reloaded.allowed_macs() == {MAC_A}
    assert reloaded.known()[MAC_A].last_seen == 100.0


def test_registry_empty_allowlist_permits_everything(tmp_path) -> None:
    registry = CameraRegistry(cache_path=tmp_path / "r.json")
    assert registry.permits(MAC_A) is True
    assert registry.permits(None) is True
    assert registry.enforcing() is False


def test_registry_nonempty_allowlist_permits_members_only(tmp_path) -> None:
    registry = CameraRegistry(cache_path=tmp_path / "r.json")
    registry.allow(MAC_A)
    assert registry.permits(MAC_A) is True
    assert registry.permits(MAC_B) is False
    # An unresolvable MAC can't prove membership while the list is enforcing.
    assert registry.permits(None) is False
    assert registry.enforcing() is True


def test_registry_allow_remove_roundtrip(tmp_path) -> None:
    registry = CameraRegistry(cache_path=tmp_path / "r.json")
    assert registry.allow(MAC_A) is True
    assert registry.allow(MAC_A) is False  # already present
    assert registry.remove(MAC_A) is True
    assert registry.remove(MAC_A) is False  # already gone


def test_registry_mac_for_host_reverse_lookup(tmp_path) -> None:
    registry = CameraRegistry(cache_path=tmp_path / "r.json")
    registry.record_sighting(MAC_A, "10.0.0.5")
    assert registry.mac_for_host("10.0.0.5") == MAC_A
    assert registry.mac_for_host("10.0.0.9") is None


def test_registry_tolerates_corrupt_file(tmp_path) -> None:
    path = tmp_path / "camera_registry.json"
    path.write_text("{not json")
    registry = CameraRegistry(cache_path=path)  # must not raise
    assert registry.known() == {}
    assert registry.allowed_macs() == set()


def test_registry_tolerates_wrong_shape_sections(tmp_path) -> None:
    # Valid JSON with the wrong shapes (a hand-edited file) must warn, not
    # crash the server at boot.
    path = tmp_path / "camera_registry.json"
    path.write_text(json.dumps({"allowed": "oops", "cameras": [MAC_A]}))
    registry = CameraRegistry(cache_path=path)
    assert registry.known() == {}
    assert registry.allowed_macs() == set()


def test_record_sighting_evicts_stale_ip_claim(tmp_path) -> None:
    # DHCP moved an IP from camera B to camera A: the new sighting must strip
    # B's claim so allow/remove/status never act on the wrong camera.
    registry = CameraRegistry(cache_path=tmp_path / "r.json")
    registry.record_sighting(MAC_B, "10.0.0.5", now=1.0)
    registry.record_sighting(MAC_A, "10.0.0.5", now=2.0)
    assert registry.ip_for_mac(MAC_A) == "10.0.0.5"
    assert registry.ip_for_mac(MAC_B) is None
    assert registry.mac_for_host("10.0.0.5") == MAC_A


def test_registry_skips_malformed_entries(tmp_path) -> None:
    path = tmp_path / "camera_registry.json"
    path.write_text(
        json.dumps(
            {
                "allowed": [MAC_A, "garbage"],
                "cameras": {MAC_B: {"ip": "10.0.0.6", "last_seen": 5.0}, "bad": {}},
            }
        )
    )
    registry = CameraRegistry(cache_path=path)
    assert registry.allowed_macs() == {MAC_A}
    assert set(registry.known()) == {MAC_B}


# --- supervisor enforcement ----------------------------------------------------


def _app_config() -> AppConfig:
    return AppConfig(
        snapshot_dir=Path("./data/server/snapshots"),
        model=ModelConfig(paths=(Path("model.pt"),)),
        telegram=TelegramConfig(enabled=False, bot_token="", user_ids=[]),
        collect=CollectConfig(objects=frozenset()),
        filter=FilterConfig(objects=frozenset()),
        credentials=CameraCredentials(username="user", password="pass"),
        discovery=DiscoveryConfig(hosts=("10.0.0.5", "10.0.0.6")),
    )


def _supervisor(monkeypatch, tmp_path, macs: dict[str, str]):
    """Supervisor with stubbed capture threads, a real registry, and a fake
    ARP resolver serving the ``ip -> mac`` mapping in ``macs``."""
    monkeypatch.setattr("lib.supervisor.monitor_camera", lambda *_a, **_k: None)
    registry = CameraRegistry(cache_path=tmp_path / "camera_registry.json")
    supervisor = CameraSupervisor(
        app_config=_app_config(),
        detector=object(),
        alert_state=object(),
        dispatcher=object(),
        registry=ObjectRegistry(),
        stats={},
        stats_lock=threading.Lock(),
        stop_event=threading.Event(),
        watchlist=registry,
        mac_resolver=macs.get,
    )
    return supervisor, registry


def _sweep_result(*hosts: str) -> DiscoveryResult:
    return DiscoveryResult(
        cameras=[
            DiscoveredCamera(
                host=host,
                port=554,
                stream_path="/stream1",
                rtsp_url=f"rtsp://user:pass@{host}:554/stream1",
            )
            for host in hosts
        ],
        hosts_scanned=len(hosts),
        ports_open=len(hosts),
        auth_failures=0,
        elapsed_seconds=0.1,
    )


def test_sweep_streams_only_watchlisted_cameras(monkeypatch, tmp_path) -> None:
    supervisor, registry = _supervisor(
        monkeypatch, tmp_path, {"10.0.0.5": MAC_A, "10.0.0.6": MAC_B}
    )
    registry.allow(MAC_A)
    monkeypatch.setattr(
        "lib.supervisor.discover_cameras",
        lambda *_a, **_k: _sweep_result("10.0.0.5", "10.0.0.6"),
    )

    applied = supervisor.discover_and_apply()

    assert applied.added == ["camera-10.0.0.5"]
    assert applied.blocked == ["10.0.0.6 (AC-A7-F1-34-3F-8C)"]
    assert supervisor.active_hosts() == {"10.0.0.5"}
    # The blocked camera is still CACHED (with its MAC) so /watchlist can offer it.
    assert registry.ip_for_mac(MAC_B) == "10.0.0.6"
    assert applied.macs == {"10.0.0.5": MAC_A, "10.0.0.6": MAC_B}


def test_sweep_blocks_unresolvable_mac_when_enforcing(monkeypatch, tmp_path) -> None:
    supervisor, registry = _supervisor(monkeypatch, tmp_path, {"10.0.0.5": MAC_A})
    registry.allow(MAC_A)
    monkeypatch.setattr(
        "lib.supervisor.discover_cameras",
        lambda *_a, **_k: _sweep_result("10.0.0.6"),  # MAC unknown for this host
    )

    applied = supervisor.discover_and_apply()

    assert applied.added == []
    assert applied.blocked == ["10.0.0.6 (MAC unknown)"]
    assert supervisor.active_hosts() == set()


def test_sweep_with_empty_watchlist_streams_everything(monkeypatch, tmp_path) -> None:
    supervisor, registry = _supervisor(monkeypatch, tmp_path, {"10.0.0.5": MAC_A})
    monkeypatch.setattr(
        "lib.supervisor.discover_cameras", lambda *_a, **_k: _sweep_result("10.0.0.5")
    )

    applied = supervisor.discover_and_apply()

    assert applied.added == ["camera-10.0.0.5"]
    assert applied.blocked == []
    # Sightings are recorded even when nothing is filtered.
    assert registry.ip_for_mac(MAC_A) == "10.0.0.5"


def test_sweep_stops_active_camera_that_lost_its_watchlist_spot(monkeypatch, tmp_path) -> None:
    # Boot with an empty list (permits all) -> camera streams; then the owner
    # allows a DIFFERENT camera. The next sweep confirms only the now-blocked
    # camera — it must be stopped THERE, since the miss counter never runs on
    # a sweep whose permitted set is empty.
    supervisor, registry = _supervisor(monkeypatch, tmp_path, {"10.0.0.5": MAC_A})
    monkeypatch.setattr(
        "lib.supervisor.discover_cameras", lambda *_a, **_k: _sweep_result("10.0.0.5")
    )
    supervisor.discover_and_apply()
    assert supervisor.active_hosts() == {"10.0.0.5"}

    registry.allow(MAC_B)  # enforcing on; MAC_A is not listed
    applied = supervisor.discover_and_apply()

    assert supervisor.active_hosts() == set()
    assert applied.blocked == ["10.0.0.5 (CC-BA-BD-9A-EF-51) — stream stopped"]


def test_allow_enforcement_stops_unlisted_active_cameras(monkeypatch, tmp_path) -> None:
    # /watchlist allow flips the list from "empty = allow all" to enforcing;
    # cameras streaming on that default must stop immediately, not next sweep.
    supervisor, registry = _supervisor(monkeypatch, tmp_path, {"10.0.0.5": MAC_A})
    monkeypatch.setattr(
        "lib.supervisor.discover_cameras", lambda *_a, **_k: _sweep_result("10.0.0.5")
    )
    supervisor.discover_and_apply()
    assert supervisor.active_hosts() == {"10.0.0.5"}

    reply = supervisor.allow_camera(MAC_B)

    assert supervisor.active_hosts() == set()
    assert "Stopped (not on the watchlist): camera-10.0.0.5" in reply


def test_allow_does_not_start_ip_that_moved_to_another_device(monkeypatch, tmp_path) -> None:
    # The cached IP now answers with a DIFFERENT MAC (DHCP churn): starting it
    # would stream the wrong camera under the allowed identity.
    supervisor, registry = _supervisor(monkeypatch, tmp_path, {"10.0.0.5": MAC_B})
    registry.record_sighting(MAC_A, "10.0.0.5")

    reply = supervisor.allow_camera(MAC_A)

    assert "belongs to a different device" in reply
    assert supervisor.active_hosts() == set()
    assert registry.permits(MAC_A) is True


def test_remove_does_not_stop_wrong_camera_after_dhcp_move(monkeypatch, tmp_path) -> None:
    # Stale cache: MAC_B's record still claims 10.0.0.5, but the device at that
    # IP is now MAC_A's (allowed, streaming) camera. Removing MAC_B must NOT
    # kill MAC_A's stream.
    supervisor, registry = _supervisor(monkeypatch, tmp_path, {"10.0.0.5": MAC_A})
    registry.record_sighting(MAC_B, "10.0.0.5")
    registry.allow(MAC_A)
    registry.allow(MAC_B)
    supervisor.start_host("10.0.0.5")
    assert supervisor.active_hosts() == {"10.0.0.5"}

    reply = supervisor.remove_camera(MAC_B)

    assert "Removed" in reply
    assert "Stopped" not in reply
    assert supervisor.active_hosts() == {"10.0.0.5"}


def test_allow_starts_cached_camera_immediately(monkeypatch, tmp_path) -> None:
    supervisor, registry = _supervisor(monkeypatch, tmp_path, {})
    registry.record_sighting(MAC_A, "10.0.0.5")
    registry.allow(MAC_B)  # enforcing; MAC_A currently blocked

    reply = supervisor.allow_camera("CC-BA-BD-9A-EF-51")

    assert "CC-BA-BD-9A-EF-51" in reply
    assert "10.0.0.5" in reply
    # The stream starts NOW, without waiting for the next discovery sweep.
    assert supervisor.active_hosts() == {"10.0.0.5"}
    assert registry.permits(MAC_A) is True


def test_allow_unseen_camera_is_stored_for_later(monkeypatch, tmp_path) -> None:
    supervisor, registry = _supervisor(monkeypatch, tmp_path, {})

    reply = supervisor.allow_camera(MAC_B)

    assert "hasn't been seen" in reply
    assert registry.permits(MAC_B) is True
    assert supervisor.active_hosts() == set()


def test_allow_rejects_malformed_mac(monkeypatch, tmp_path) -> None:
    supervisor, registry = _supervisor(monkeypatch, tmp_path, {})
    reply = supervisor.allow_camera("pizza")
    assert "doesn't look like a MAC" in reply
    assert registry.allowed_macs() == set()


def test_remove_stops_running_camera_immediately(monkeypatch, tmp_path) -> None:
    supervisor, registry = _supervisor(monkeypatch, tmp_path, {"10.0.0.5": MAC_A})
    registry.allow(MAC_A)
    registry.allow(MAC_B)
    monkeypatch.setattr(
        "lib.supervisor.discover_cameras", lambda *_a, **_k: _sweep_result("10.0.0.5")
    )
    supervisor.discover_and_apply()
    assert supervisor.active_hosts() == {"10.0.0.5"}

    reply = supervisor.remove_camera(MAC_A)

    assert "Removed" in reply
    assert "Stopped camera-10.0.0.5" in reply
    assert supervisor.active_hosts() == set()
    assert registry.permits(MAC_A) is False


def test_remove_last_mac_warns_filtering_is_off(monkeypatch, tmp_path) -> None:
    supervisor, registry = _supervisor(monkeypatch, tmp_path, {})
    registry.allow(MAC_A)
    reply = supervisor.remove_camera(MAC_A)
    assert "watchlist is now empty" in reply
    assert "EVERY discovered camera" in reply


def test_watchlist_text_groups_watched_and_unwatched(monkeypatch, tmp_path) -> None:
    supervisor, registry = _supervisor(
        monkeypatch, tmp_path, {"10.0.0.5": MAC_A, "10.0.0.6": MAC_B}
    )
    registry.allow(MAC_A)
    monkeypatch.setattr(
        "lib.supervisor.discover_cameras",
        lambda *_a, **_k: _sweep_result("10.0.0.5", "10.0.0.6"),
    )
    supervisor.discover_and_apply()

    text = supervisor.watchlist_text()

    watched, unwatched = text.split("**Discovered, not on the watchlist:**")
    assert "CC-BA-BD-9A-EF-51" in watched
    assert "10.0.0.5" in watched
    assert "streaming" in watched
    assert "AC-A7-F1-34-3F-8C" in unwatched
    assert "10.0.0.6" in unwatched


def test_watchlist_text_marks_unseen_allowed_camera(monkeypatch, tmp_path) -> None:
    supervisor, registry = _supervisor(monkeypatch, tmp_path, {})
    registry.allow(MAC_B)
    text = supervisor.watchlist_text()
    assert "AC-A7-F1-34-3F-8C" in text
    assert "not seen on the network yet" in text


def test_discovery_report_shows_macs_and_blocked(monkeypatch, tmp_path) -> None:
    supervisor, registry = _supervisor(
        monkeypatch, tmp_path, {"10.0.0.5": MAC_A, "10.0.0.6": MAC_B}
    )
    registry.allow(MAC_A)
    monkeypatch.setattr(
        "lib.supervisor.discover_cameras",
        lambda *_a, **_k: _sweep_result("10.0.0.5", "10.0.0.6"),
    )

    report = format_discovery_report(supervisor.discover_and_apply())

    assert "camera-10.0.0.5 (`CC-BA-BD-9A-EF-51`)" in report
    assert "Not on the watchlist" in report
    assert "10.0.0.6 (AC-A7-F1-34-3F-8C)" in report
