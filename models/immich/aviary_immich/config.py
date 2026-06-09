"""Config and .env loading for Immich scripts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BIRD_ALBUM_NAME = "Birds"


@dataclass(frozen=True)
class AccountConfig:
    slug: str
    api_key_env: str
    api_key: str
    enabled: bool = True


@dataclass(frozen=True)
class ImmichConfig:
    base_url: str
    accounts: list[AccountConfig]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def normalize_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if not normalized:
        raise ValueError("IMMICH_BASE_URL is empty")
    if not normalized.endswith("/api"):
        normalized = f"{normalized}/api"
    return normalized


def load_accounts_config(path: Path, env_file: Path = Path(".env")) -> ImmichConfig:
    load_env_file(env_file)

    import yaml

    if not path.exists():
        raise FileNotFoundError(
            f"Missing account config: {path}."
        )

    base_url = os.getenv("IMMICH_BASE_URL")
    if not base_url:
        raise ValueError("IMMICH_BASE_URL must be set in .env, for example http://192.168.1.168:2283/api")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    accounts = [_load_account(item) for item in raw.get("accounts", [])]
    enabled_accounts = [account for account in accounts if account.enabled]
    if not enabled_accounts:
        raise ValueError(f"No enabled accounts found in {path}")

    return ImmichConfig(base_url=normalize_base_url(base_url), accounts=enabled_accounts)


def _load_account(raw: dict[str, Any]) -> AccountConfig:
    slug = str(raw["slug"]).strip()
    api_key_env = str(raw["api_key_env"]).strip()
    api_key = os.getenv(api_key_env, "").strip()

    if not slug:
        raise ValueError("Account slug cannot be empty")
    if not api_key_env:
        raise ValueError(f"Account {slug} is missing api_key_env")
    if not api_key:
        raise ValueError(f"Environment variable {api_key_env} is required for account {slug}")

    return AccountConfig(
        slug=slug,
        api_key_env=api_key_env,
        api_key=api_key,
        enabled=bool(raw.get("enabled", True)),
    )
