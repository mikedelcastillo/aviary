"""Headless startup arguments must preserve the two interactive modes."""
import pytest

from lib.main import parse_args


def test_headless_is_explicit_and_excludes_terminal_chat(monkeypatch):
    monkeypatch.setattr("sys.argv", ["server", "--headless"])
    args = parse_args()
    assert args.headless and not args.chat
    monkeypatch.setattr("sys.argv", ["server"])
    args = parse_args()
    assert not args.headless and not args.chat
    monkeypatch.setattr("sys.argv", ["server", "--headless", "--chat"])
    with pytest.raises(SystemExit):
        parse_args()
