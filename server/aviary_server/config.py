"""Application configuration.

Camera definitions are hardcoded below. Secrets and host-specific paths come
from the environment; RTSP URLs come from ``TAPO_RSTP`` (comma-separated, ordered
to match ``CAMERA_SPECS``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ZoneConfig:
    name: str
    polygon: list[tuple[int, int]]
    alert: bool = True


@dataclass(frozen=True)
class CameraConfig:
    name: str
    enabled: bool
    rtsp_url: str
    sample_fps: float = 1.0
    reconnect_seconds: float = 5.0
    alert_zones: list[ZoneConfig] = field(default_factory=list)


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    confidence: float = 0.45
    iou: float = 0.5
    image_size: int = 960
    device: str = "auto"


@dataclass(frozen=True)
class TelegramConfig:
    enabled: bool
    bot_token: str
    user_ids: list[str]
    cooldown_seconds: int = 600
    include_snapshot: bool = True


@dataclass(frozen=True)
class AppConfig:
    snapshot_dir: Path
    model: ModelConfig
    telegram: TelegramConfig
    cameras: list[CameraConfig]


# Hardcoded camera definitions. Each entry maps positionally to a URL in the
# comma-separated ``TAPO_RSTP`` env var. Edit this list to add, rename, enable,
# or zone cameras.
CAMERA_SPECS: list[dict[str, Any]] = [
    {"name": "room-main", "enabled": True, "sample_fps": 1.0, "reconnect_seconds": 5.0, "alert_zones": []},
    {"name": "room-side", "enabled": False, "sample_fps": 1.0, "reconnect_seconds": 5.0, "alert_zones": []},
    {"name": "room-perch", "enabled": False, "sample_fps": 1.0, "reconnect_seconds": 5.0, "alert_zones": []},
]


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _as_user_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _rtsp_urls() -> list[str]:
    raw = os.environ.get("TAPO_RSTP", "")
    return [url.strip() for url in raw.split(",") if url.strip()]


def _build_cameras() -> list[CameraConfig]:
    urls = _rtsp_urls()
    cameras: list[CameraConfig] = []
    for index, spec in enumerate(CAMERA_SPECS):
        name = spec["name"]
        enabled = bool(spec.get("enabled", True))
        if index >= len(urls):
            if enabled:
                raise ValueError(
                    f"Camera {name!r} is enabled but TAPO_RSTP has no URL at index {index}"
                )
            continue
        cameras.append(
            CameraConfig(
                name=name,
                enabled=enabled,
                rtsp_url=urls[index],
                sample_fps=float(spec.get("sample_fps", 1.0)),
                reconnect_seconds=float(spec.get("reconnect_seconds", 5.0)),
                alert_zones=list(spec.get("alert_zones", [])),
            )
        )
    return cameras


def build_config() -> AppConfig:
    cameras = _build_cameras()
    if not cameras:
        raise ValueError("No cameras configured; set TAPO_RSTP")

    model = ModelConfig(path=Path(_require_env("AVIARY_MODEL_PATH")))

    telegram = TelegramConfig(
        enabled=True,
        bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        user_ids=_as_user_ids(os.environ.get("TELEGRAM_USER_IDS", "")),
        cooldown_seconds=600,
        include_snapshot=True,
    )

    return AppConfig(
        snapshot_dir=Path(os.environ.get("AVIARY_SNAPSHOT_DIR", "server/snapshots")),
        model=model,
        telegram=telegram,
        cameras=cameras,
    )
