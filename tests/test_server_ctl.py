from __future__ import annotations

import sys
import types
from pathlib import Path

import lib.server_ctl as sc


def test_tmux_conf_rebinds_ctrl_c_and_escape_to_detach():
    conf = sc.render_tmux_conf()
    assert "bind-key -n C-c detach-client" in conf
    assert "bind-key -n Escape detach-client" in conf
    # escape-time must be short so a lone Esc fires the binding without swallowing
    # arrow-key escape sequences.
    assert "escape-time 50" in conf


def test_launcher_sets_inner_env_and_cds_into_repo():
    repo = Path("/srv/aviary repo")  # space proves the path is quoted
    out = sc.render_launcher(repo, "/home/u/.local/bin/uv")
    assert f"export {sc.INNER_ENV}=1" in out
    assert "cd '/srv/aviary repo'" in out
    assert "/home/u/.local/bin/uv run server" in out  # absolute uv, plain path needs no quoting
    # A clean exit (code 0) must stop the respawn loop, not restart forever.
    assert '[ "$code" -eq 0 ] && exit 0' in out


def test_unit_has_absolute_paths_and_lifecycle():
    repo = Path("/srv/aviary")
    unit = sc.render_unit(repo, "/usr/bin/tmux", "/home/u/.local/bin/uv")
    assert "Type=oneshot" in unit
    assert "RemainAfterExit=yes" in unit
    assert f"WorkingDirectory={repo}" in unit
    assert "WantedBy=default.target" in unit  # boot autostart target
    # systemd PATH must include uv's dir so `uv` resolves under the minimal env.
    assert "/home/u/.local/bin" in unit
    assert "ExecStart=/usr/bin/tmux -L aviary" in unit
    assert "new-session -d -s aviary" in unit
    assert "ExecStop=/usr/bin/tmux -L aviary kill-session -t aviary" in unit


def test_session_running_reads_returncode(monkeypatch):
    monkeypatch.setattr(sc.shutil, "which", lambda _name: "/usr/bin/tmux")
    calls = {}

    class _R:
        def __init__(self, rc):
            self.returncode = rc

    def fake_run(cmd, **kw):
        calls["cmd"] = cmd
        return _R(0)

    monkeypatch.setattr(sc.subprocess, "run", fake_run)
    assert sc._session_running() is True
    assert calls["cmd"][:5] == ["/usr/bin/tmux", "-L", "aviary", "has-session", "-t"]

    monkeypatch.setattr(sc.subprocess, "run", lambda cmd, **kw: _R(1))
    assert sc._session_running() is False


def test_server_skips_attach_inside_the_pane(monkeypatch):
    """With INNER_ENV set (we ARE the bg server), server() must run main, not attach."""
    monkeypatch.setenv(sc.INNER_ENV, "1")

    attached = {"v": False}
    monkeypatch.setattr(sc, "_attach", lambda: attached.__setitem__("v", True))

    ran = {"v": False}
    fake_main = types.ModuleType("lib.main")
    fake_main.main = lambda: ran.__setitem__("v", True)
    monkeypatch.setitem(sys.modules, "lib.main", fake_main)

    sc.server()
    assert ran["v"] is True
    assert attached["v"] is False


def test_server_attaches_when_session_up(monkeypatch):
    monkeypatch.delenv(sc.INNER_ENV, raising=False)
    monkeypatch.setattr(sc.sys, "platform", "linux")
    monkeypatch.setattr(sc.shutil, "which", lambda _name: "/usr/bin/tmux")
    monkeypatch.setattr(sc, "_session_running", lambda: True)

    attached = {"v": False}
    monkeypatch.setattr(sc, "_attach", lambda: attached.__setitem__("v", True))

    # main must NOT be imported/run when we attach.
    fake_main = types.ModuleType("lib.main")
    fake_main.main = lambda: (_ for _ in ()).throw(AssertionError("main ran"))
    monkeypatch.setitem(sys.modules, "lib.main", fake_main)

    sc.server()
    assert attached["v"] is True
