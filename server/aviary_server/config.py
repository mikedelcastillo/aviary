"""Configuration loading and validation."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


UNRESOLVED_ENV_PATTERN = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")


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


def _expand_env(raw_text: str) -> str:
    expanded = os.path.expandvars(raw_text)
    unresolved = sorted(set(UNRESOLVED_ENV_PATTERN.findall(expanded)))
    if unresolved:
        joined = ", ".join(unresolved)
        raise ValueError(f"Missing required environment variables in config: {joined}")
    return expanded


def _as_user_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _as_polygon(value: Any) -> list[tuple[int, int]]:
    polygon: list[tuple[int, int]] = []
    for point in value or []:
        if len(point) != 2:
            raise ValueError(f"Invalid polygon point: {point}")
        polygon.append((int(point[0]), int(point[1])))
    if len(polygon) < 3:
        raise ValueError("Alert zone polygons need at least 3 points")
    return polygon


def _load_zone(raw: dict[str, Any]) -> ZoneConfig:
    return ZoneConfig(
        name=str(raw["name"]),
        polygon=_as_polygon(raw.get("polygon")),
        alert=bool(raw.get("alert", True)),
    )


def _load_camera(raw: dict[str, Any]) -> CameraConfig:
    zones = [_load_zone(zone) for zone in raw.get("alert_zones", []) or []]
    sample_fps = float(raw.get("sample_fps", 1.0))
    if sample_fps <= 0:
        raise ValueError(f"Camera {raw.get('name')} sample_fps must be greater than 0")

    return CameraConfig(
        name=str(raw["name"]),
        enabled=bool(raw.get("enabled", True)),
        rtsp_url=str(raw["rtsp_url"]),
        sample_fps=sample_fps,
        reconnect_seconds=float(raw.get("reconnect_seconds", 5.0)),
        alert_zones=zones,
    )


def load_config(path: Path) -> AppConfig:
    raw_text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(_expand_env(raw_text)) or {}

    model_raw = raw.get("model", {})
    telegram_raw = raw.get("telegram", {})
    cameras_raw = raw.get("cameras", [])

    if not cameras_raw:
        raise ValueError("At least one camera must be configured")

    model = ModelConfig(
        path=Path(model_raw["path"]),
        confidence=float(model_raw.get("confidence", 0.45)),
        iou=float(model_raw.get("iou", 0.5)),
        image_size=int(model_raw.get("image_size", 960)),
        device=str(model_raw.get("device", "auto")),
    )

    telegram = TelegramConfig(
        enabled=bool(telegram_raw.get("enabled", True)),
        bot_token=str(telegram_raw.get("bot_token", "")),
        user_ids=_as_user_ids(telegram_raw.get("user_ids", "")),
        cooldown_seconds=int(telegram_raw.get("cooldown_seconds", 600)),
        include_snapshot=bool(telegram_raw.get("include_snapshot", True)),
    )

    cameras = [_load_camera(camera) for camera in cameras_raw]

    return AppConfig(
        snapshot_dir=Path(raw.get("snapshot_dir", "server/snapshots")),
        model=model,
        telegram=telegram,
        cameras=cameras,
    )
