"""Application configuration.

Camera definitions are hardcoded below. Secrets and host-specific paths come
from the environment; RTSP URLs come from ``TAPO_RSTP`` (comma-separated, ordered
to match ``CAMERA_SPECS``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CameraConfig:
    name: str
    enabled: bool
    rtsp_url: str
    sample_fps: float = 1.0
    # Base reconnect delay; grows exponentially up to ``max_reconnect_seconds``
    # while a camera stays unreachable, then resets once frames flow again.
    reconnect_seconds: float = 5.0
    max_reconnect_seconds: float = 60.0
    # Hard caps so a missing camera can't block a worker thread forever. Tapo
    # streams are flaky, so opening and reading must be able to give up.
    open_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 15.0
    # RTSP over TCP is far more reliable than the UDP default on lossy WiFi.
    rtsp_transport: str = "tcp"


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
    last_seen_alert_seconds: float = 60.0
    bbox_movement_alert_ratio: float = 0.10
    include_snapshot: bool = True


@dataclass(frozen=True)
class AppConfig:
    snapshot_dir: Path
    model: ModelConfig
    telegram: TelegramConfig
    cameras: list[CameraConfig]


# Shared camera defaults. One camera is built per URL in the comma-separated
# ``TAPO_RSTP`` env var, each using these settings.
SAMPLE_FPS = 0.25
RECONNECT_SECONDS = 5
MAX_RECONNECT_SECONDS = 60
OPEN_TIMEOUT_SECONDS = 10
READ_TIMEOUT_SECONDS = 15
RTSP_TRANSPORT = "tcp"


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
    return [
        CameraConfig(
            name=f"camera-{index + 1}",
            enabled=True,
            rtsp_url=url,
            sample_fps=SAMPLE_FPS,
            reconnect_seconds=RECONNECT_SECONDS,
        )
        for index, url in enumerate(_rtsp_urls())
    ]


def build_config() -> AppConfig:
    cameras = _build_cameras()
    if not cameras:
        raise ValueError("No cameras configured; set TAPO_RSTP")

    model = ModelConfig(path=Path(_require_env("AVIARY_MODEL_PATH")))

    telegram = TelegramConfig(
        enabled=True,
        bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        user_ids=_as_user_ids(os.environ.get("TELEGRAM_USER_IDS", "")),
        last_seen_alert_seconds=15.0,
        bbox_movement_alert_ratio=0.10,
        include_snapshot=True,
    )

    return AppConfig(
        snapshot_dir=Path("./snapshots"),
        model=model,
        telegram=telegram,
        cameras=cameras,
    )
