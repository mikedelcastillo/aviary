"""Tests for aviary_immich.config: URL normalization, .env loading, and account config."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from aviary_immich.config import (
    ALBUM_RULES,
    ANIMAL_ALBUMS,
    ANIMAL_LABELS,
    MODELS,
    AccountConfig,
    ImmichConfig,
    _load_account,
    album_names,
    load_accounts_config,
    load_env_file,
    normalize_base_url,
    yolo_labels,
)


# --------------------------------------------------------------------------- constants


def test_animal_albums_mapping():
    # Back-compat shim, now derived from the YOLO signals in ALBUM_RULES (includes the racket).
    assert ANIMAL_ALBUMS == {
        "bird": "Birds",
        "dog": "Dogs",
        "cat": "Cats",
        "tennis racket": "Tennis",
    }


def test_animal_labels_tuple():
    assert ANIMAL_LABELS == ("bird", "dog", "cat", "tennis racket")


def test_album_names_in_declaration_order():
    assert album_names() == ("Birds", "Dogs", "Cats", "Tennis")


def test_yolo_labels_union_across_specs():
    assert yolo_labels() == ("bird", "dog", "cat", "tennis racket")


def test_clip_spec_present_but_disabled_by_default(monkeypatch):
    # CLIP is opt-in (IMMICH_CLIP); the spec exists but ships disabled so a stock run needs no
    # extra dependency. The Tennis rule still unions the YOLO racket with the CLIP court signal.
    clip = next(spec for spec in MODELS if spec.kind == "clip")
    assert clip.enabled is False
    tennis = next(rule for rule in ALBUM_RULES if rule.album_name == "Tennis")
    assert {(s.model, s.tag) for s in tennis.signals} == {
        ("yolo", "tennis racket"),
        ("clip", "tennis court"),
    }
    assert tennis.mode == "union"


# --------------------------------------------------------------------------- normalize_base_url


def test_normalize_strips_trailing_slash_and_appends_api():
    assert normalize_base_url("http://host:2283/") == "http://host:2283/api"


def test_normalize_appends_api_when_missing():
    assert normalize_base_url("http://host:2283") == "http://host:2283/api"


def test_normalize_leaves_existing_api():
    assert normalize_base_url("http://host:2283/api") == "http://host:2283/api"


def test_normalize_strips_trailing_slash_after_api():
    assert normalize_base_url("http://host:2283/api/") == "http://host:2283/api"


def test_normalize_empty_raises():
    with pytest.raises(ValueError):
        normalize_base_url("")


def test_normalize_whitespace_only_raises():
    with pytest.raises(ValueError):
        normalize_base_url("   ")


# --------------------------------------------------------------------------- load_env_file


def test_load_env_file_parses_key_value(tmp_path, monkeypatch):
    monkeypatch.delenv("CFG_TEST_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("CFG_TEST_KEY=plainvalue\n", encoding="utf-8")
    load_env_file(env)
    import os

    assert os.environ["CFG_TEST_KEY"] == "plainvalue"


def test_load_env_file_honors_export_prefix(tmp_path, monkeypatch):
    monkeypatch.delenv("CFG_EXPORT_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("export CFG_EXPORT_KEY=exported\n", encoding="utf-8")
    load_env_file(env)
    import os

    assert os.environ["CFG_EXPORT_KEY"] == "exported"


def test_load_env_file_strips_double_quotes(tmp_path, monkeypatch):
    monkeypatch.delenv("CFG_DQ_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('CFG_DQ_KEY="quoted value"\n', encoding="utf-8")
    load_env_file(env)
    import os

    assert os.environ["CFG_DQ_KEY"] == "quoted value"


def test_load_env_file_strips_single_quotes(tmp_path, monkeypatch):
    monkeypatch.delenv("CFG_SQ_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("CFG_SQ_KEY='single quoted'\n", encoding="utf-8")
    load_env_file(env)
    import os

    assert os.environ["CFG_SQ_KEY"] == "single quoted"


def test_load_env_file_skips_blank_and_comment_lines(tmp_path, monkeypatch):
    monkeypatch.delenv("CFG_REAL_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        textwrap.dedent(
            """\

            # this is a comment
            CFG_REAL_KEY=real

            """
        ),
        encoding="utf-8",
    )
    load_env_file(env)
    import os

    assert os.environ["CFG_REAL_KEY"] == "real"


def test_load_env_file_skips_lines_without_equals(tmp_path, monkeypatch):
    monkeypatch.delenv("CFG_NOEQ_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("NOTANASSIGNMENT\nCFG_NOEQ_KEY=ok\n", encoding="utf-8")
    load_env_file(env)
    import os

    assert os.environ["CFG_NOEQ_KEY"] == "ok"


def test_load_env_file_setdefault_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("CFG_EXISTING_KEY", "preexisting")
    env = tmp_path / ".env"
    env.write_text("CFG_EXISTING_KEY=fromfile\n", encoding="utf-8")
    load_env_file(env)
    import os

    assert os.environ["CFG_EXISTING_KEY"] == "preexisting"


def test_load_env_file_missing_file_is_noop(tmp_path):
    # Should not raise even though the file does not exist.
    load_env_file(tmp_path / "does-not-exist.env")


# --------------------------------------------------------------------------- _load_account


def test_load_account_valid(monkeypatch):
    monkeypatch.setenv("ACCT_KEY_ENV", "secret-key")
    account = _load_account({"slug": "myacct", "api_key_env": "ACCT_KEY_ENV"})
    assert isinstance(account, AccountConfig)
    assert account.slug == "myacct"
    assert account.api_key_env == "ACCT_KEY_ENV"
    assert account.api_key == "secret-key"
    assert account.enabled is True


def test_load_account_enabled_defaults_true(monkeypatch):
    monkeypatch.setenv("ACCT_KEY_ENV", "secret-key")
    account = _load_account({"slug": "myacct", "api_key_env": "ACCT_KEY_ENV"})
    assert account.enabled is True


def test_load_account_enabled_can_be_false(monkeypatch):
    monkeypatch.setenv("ACCT_KEY_ENV", "secret-key")
    account = _load_account(
        {"slug": "myacct", "api_key_env": "ACCT_KEY_ENV", "enabled": False}
    )
    assert account.enabled is False


def test_load_account_missing_api_key_env_value_raises(monkeypatch):
    monkeypatch.delenv("ACCT_MISSING_ENV", raising=False)
    with pytest.raises(ValueError):
        _load_account({"slug": "myacct", "api_key_env": "ACCT_MISSING_ENV"})


def test_load_account_empty_slug_raises(monkeypatch):
    monkeypatch.setenv("ACCT_KEY_ENV", "secret-key")
    with pytest.raises(ValueError):
        _load_account({"slug": "   ", "api_key_env": "ACCT_KEY_ENV"})


# --------------------------------------------------------------------------- load_accounts_config


def _write_config(tmp_path: Path, body: str) -> Path:
    config = tmp_path / "accounts.yaml"
    config.write_text(textwrap.dedent(body), encoding="utf-8")
    return config


def test_load_accounts_config_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("IMMICH_BASE_URL", "http://host:2283")
    missing = tmp_path / "nope.yaml"
    env_file = tmp_path / ".env"  # does not exist -> no-op
    with pytest.raises(FileNotFoundError):
        load_accounts_config(missing, env_file=env_file)


def test_load_accounts_config_missing_base_url_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("IMMICH_BASE_URL", raising=False)
    config = _write_config(
        tmp_path,
        """\
        accounts:
          - slug: a
            api_key_env: ACCT_A_KEY
        """,
    )
    env_file = tmp_path / ".env"
    with pytest.raises(ValueError):
        load_accounts_config(config, env_file=env_file)


def test_load_accounts_config_returns_normalized_config(tmp_path, monkeypatch):
    monkeypatch.setenv("IMMICH_BASE_URL", "http://host:2283/")
    monkeypatch.setenv("ACCT_A_KEY", "key-a")
    config = _write_config(
        tmp_path,
        """\
        accounts:
          - slug: a
            api_key_env: ACCT_A_KEY
        """,
    )
    env_file = tmp_path / ".env"
    result = load_accounts_config(config, env_file=env_file)
    assert isinstance(result, ImmichConfig)
    assert result.base_url == "http://host:2283/api"
    assert len(result.accounts) == 1
    assert result.accounts[0].slug == "a"
    assert result.accounts[0].api_key == "key-a"


def test_load_accounts_config_filters_disabled_accounts(tmp_path, monkeypatch):
    monkeypatch.setenv("IMMICH_BASE_URL", "http://host:2283")
    monkeypatch.setenv("ACCT_A_KEY", "key-a")
    monkeypatch.setenv("ACCT_B_KEY", "key-b")
    config = _write_config(
        tmp_path,
        """\
        accounts:
          - slug: a
            api_key_env: ACCT_A_KEY
            enabled: true
          - slug: b
            api_key_env: ACCT_B_KEY
            enabled: false
        """,
    )
    env_file = tmp_path / ".env"
    result = load_accounts_config(config, env_file=env_file)
    slugs = [account.slug for account in result.accounts]
    assert slugs == ["a"]


def test_load_accounts_config_no_enabled_accounts_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("IMMICH_BASE_URL", "http://host:2283")
    monkeypatch.setenv("ACCT_A_KEY", "key-a")
    config = _write_config(
        tmp_path,
        """\
        accounts:
          - slug: a
            api_key_env: ACCT_A_KEY
            enabled: false
        """,
    )
    env_file = tmp_path / ".env"
    with pytest.raises(ValueError):
        load_accounts_config(config, env_file=env_file)


def test_load_accounts_config_reads_env_file(tmp_path, monkeypatch):
    # base_url and the api key come purely from the .env file via setdefault.
    monkeypatch.delenv("IMMICH_BASE_URL", raising=False)
    monkeypatch.delenv("ACCT_ENVFILE_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "IMMICH_BASE_URL=http://envhost:2283\nACCT_ENVFILE_KEY=key-from-env\n",
        encoding="utf-8",
    )
    config = _write_config(
        tmp_path,
        """\
        accounts:
          - slug: envacct
            api_key_env: ACCT_ENVFILE_KEY
        """,
    )
    result = load_accounts_config(config, env_file=env_file)
    assert result.base_url == "http://envhost:2283/api"
    assert result.accounts[0].api_key == "key-from-env"
