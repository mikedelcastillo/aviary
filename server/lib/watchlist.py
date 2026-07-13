"""MAC-keyed camera registry + watchlist: which cameras may stream.

Cameras used to be identified only by IP, which drifts with DHCP and gives the
owner no say over WHICH cameras the server consumes. This registry keys every
credential-confirmed camera on its MAC address (stable hardware identity):

  * every camera the discovery sweep confirms is CACHED here (MAC -> last-known
    IP + last-seen time), whether or not it is being streamed, and
  * an ALLOWLIST of MACs decides which cameras actually stream. An EMPTY list
    means no filtering (every discovered camera streams — the out-of-the-box
    behavior); once any MAC is listed, only listed cameras are consumed.

State persists to JSON (same tmp-write + atomic-rename pattern as
:mod:`lib.camera_names`) so the cache and the allowlist survive restarts, and
``/watchlist`` can offer cameras that are currently offline.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from lib.netid import format_mac, normalize_mac

LOGGER = logging.getLogger("lib.watchlist")


@dataclass(frozen=True)
class CameraRecord:
    """A camera the sweep has confirmed at least once, keyed by MAC upstream."""

    ip: str
    last_seen: float  # epoch seconds of the last credential-confirmed sighting


class CameraRegistry:
    """Thread-safe MAC -> camera cache plus the streaming allowlist."""

    def __init__(self, cache_path: Path | None = None) -> None:
        self._lock = threading.Lock()
        # Serializes whole save operations (snapshot + write + rename). The
        # discovery thread and the Telegram command thread both mutate-and-save;
        # without this a stale snapshot written last would silently revert the
        # other thread's just-persisted change (visible only after a restart).
        self._save_lock = threading.Lock()
        self._cameras: dict[str, CameraRecord] = {}
        self._allowed: set[str] = set()
        self._cache_path = Path(cache_path) if cache_path else None
        if self._cache_path is not None:
            self._load()

    # -- persistence ---------------------------------------------------------

    def _load(self) -> None:
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception:
            LOGGER.exception("Could not read camera registry %s", self._cache_path)
            return
        if not isinstance(data, dict):
            return
        # Shape-check each section too: a hand-edited file that still parses as
        # JSON must degrade to a warning, never crash the server at boot.
        cameras_raw = data.get("cameras")
        if not isinstance(cameras_raw, dict):
            if cameras_raw is not None:
                LOGGER.warning("Ignoring malformed 'cameras' section in %s", self._cache_path)
            cameras_raw = {}
        allowed_raw = data.get("allowed")
        if not isinstance(allowed_raw, list):
            if allowed_raw is not None:
                LOGGER.warning("Ignoring malformed 'allowed' section in %s", self._cache_path)
            allowed_raw = []
        cameras: dict[str, CameraRecord] = {}
        for mac, record in cameras_raw.items():
            try:
                cameras[normalize_mac(mac)] = CameraRecord(
                    ip=str(record.get("ip", "")),
                    last_seen=float(record.get("last_seen", 0.0)),
                )
            except (ValueError, AttributeError):
                LOGGER.warning("Skipping malformed registry entry %r", mac)
        allowed = set()
        for mac in allowed_raw:
            try:
                allowed.add(normalize_mac(mac))
            except (ValueError, TypeError):
                LOGGER.warning("Skipping malformed allowlist MAC %r", mac)
        with self._lock:
            self._cameras = cameras
            self._allowed = allowed
        LOGGER.info(
            "Loaded camera registry: %d camera(s), %d on the watchlist",
            len(cameras),
            len(allowed),
        )

    def _save(self) -> None:
        if self._cache_path is None:
            return
        # The save lock covers snapshot THROUGH rename, so the last writer to
        # land always wrote the newest state (no lost update between the
        # discovery and command threads) and the tmp file is never shared.
        with self._save_lock:
            with self._lock:
                snapshot = {
                    "allowed": sorted(self._allowed),
                    "cameras": {
                        mac: {"ip": record.ip, "last_seen": record.last_seen}
                        for mac, record in self._cameras.items()
                    },
                }
            try:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._cache_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
                tmp.replace(self._cache_path)
            except Exception:
                LOGGER.exception("Could not write camera registry %s", self._cache_path)

    # -- camera cache ----------------------------------------------------------

    def record_sighting(self, mac: str, ip: str, *, now: float | None = None) -> None:
        """Cache a credential-confirmed camera (called on every sweep hit).

        Any OTHER MAC still claiming this IP loses it: DHCP just proved the
        address belongs to this camera now, and a stale claim would make
        ``ip_for_mac``/``mac_for_host`` act on (or display) the wrong camera.
        """
        mac = normalize_mac(mac)
        with self._lock:
            for other, record in self._cameras.items():
                if other != mac and record.ip == ip:
                    self._cameras[other] = CameraRecord(ip="", last_seen=record.last_seen)
            self._cameras[mac] = CameraRecord(
                ip=ip, last_seen=time.time() if now is None else now
            )
        self._save()

    def known(self) -> dict[str, CameraRecord]:
        with self._lock:
            return dict(self._cameras)

    def mac_for_host(self, host: str) -> str | None:
        """The cached MAC for an IP, or None (reverse lookup for display)."""
        with self._lock:
            for mac, record in self._cameras.items():
                if record.ip == host:
                    return mac
        return None

    def ip_for_mac(self, mac: str) -> str | None:
        with self._lock:
            record = self._cameras.get(mac)
        return record.ip if record and record.ip else None

    # -- allowlist -------------------------------------------------------------

    def permits(self, mac: str | None) -> bool:
        """May a camera with this MAC stream?

        An empty allowlist permits everything (filtering off). A non-empty list
        permits only its members — including denying a camera whose MAC could
        not be resolved, since membership can't be proven.
        """
        with self._lock:
            if not self._allowed:
                return True
            return mac is not None and mac in self._allowed

    def enforcing(self) -> bool:
        """True when a non-empty allowlist is actively filtering cameras."""
        with self._lock:
            return bool(self._allowed)

    def allowed_macs(self) -> set[str]:
        with self._lock:
            return set(self._allowed)

    def allow(self, mac: str) -> bool:
        """Add a MAC to the watchlist. True if it was newly added."""
        mac = normalize_mac(mac)
        with self._lock:
            if mac in self._allowed:
                return False
            self._allowed.add(mac)
        self._save()
        LOGGER.info("Watchlist: allowed %s", format_mac(mac))
        return True

    def remove(self, mac: str) -> bool:
        """Drop a MAC from the watchlist. True if it was present."""
        mac = normalize_mac(mac)
        with self._lock:
            if mac not in self._allowed:
                return False
            self._allowed.discard(mac)
        self._save()
        LOGGER.info("Watchlist: removed %s", format_mac(mac))
        return True
