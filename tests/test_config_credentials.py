"""Tests for TAPO_CREDENTIALS parsing in lib.config._credentials.

These import only lib.config (no ultralytics), so they stay fast and isolated.
"""

from __future__ import annotations

import pytest

from lib.config import CameraCredentials, _credentials


def test_credentials_basic(monkeypatch) -> None:
    monkeypatch.setenv("TAPO_CREDENTIALS", "admin:secret")
    creds = _credentials()
    assert creds == CameraCredentials(username="admin", password="secret")


def test_credentials_password_with_colon(monkeypatch) -> None:
    # partition() splits on the FIRST colon only, so colons in the password are
    # preserved verbatim.
    monkeypatch.setenv("TAPO_CREDENTIALS", "admin:p:a:ss")
    creds = _credentials()
    assert creds.username == "admin"
    assert creds.password == "p:a:ss"


def test_credentials_missing_colon(monkeypatch) -> None:
    monkeypatch.setenv("TAPO_CREDENTIALS", "adminonly")
    with pytest.raises(ValueError):
        _credentials()


def test_credentials_empty_username(monkeypatch) -> None:
    monkeypatch.setenv("TAPO_CREDENTIALS", ":secret")
    with pytest.raises(ValueError):
        _credentials()


def test_credentials_unset(monkeypatch) -> None:
    monkeypatch.delenv("TAPO_CREDENTIALS", raising=False)
    with pytest.raises(ValueError):
        _credentials()
