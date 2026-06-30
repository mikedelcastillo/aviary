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


def test_socket_path_is_absolute_and_env_pinned(monkeypatch):
    # An absolute `-S` socket (not `-L name`) must not depend on TMUX_TMPDIR, so the
    # systemd unit and interactive commands always resolve the same socket.
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/4242")
    assert sc._socket_path() == "/run/user/4242/aviary-tmux.sock"


def test_unit_supervises_session_and_uses_absolute_socket(monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/4242")
    repo = Path("/srv/aviary")
    unit = sc.render_unit(repo, "/usr/bin/tmux", "/home/u/.local/bin/uv")
    sock = "/run/user/4242/aviary-tmux.sock"
    # forking so systemd tracks the real tmux server and can restart a dead session,
    # instead of oneshot+RemainAfterExit which sits at a false `active (exited)`.
    assert "Type=forking" in unit
    assert "RemainAfterExit" not in unit
    assert "Restart=on-failure" in unit
    assert f"WorkingDirectory={repo}" in unit
    assert "WantedBy=default.target" in unit  # boot autostart target
    # systemd PATH must include uv's dir so `uv` resolves under the minimal env.
    assert "/home/u/.local/bin" in unit
    assert f"ExecStart=/usr/bin/tmux -S {sock}" in unit
    assert "new-session -d -s aviary" in unit
    # `-` prefix tolerates an already-gone server so a clean stop isn't marked failed.
    assert f"ExecStop=-/usr/bin/tmux -S {sock} kill-server" in unit


def test_session_running_reads_returncode(monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/4242")
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
    assert calls["cmd"][:3] == ["/usr/bin/tmux", "-S", "/run/user/4242/aviary-tmux.sock"]
    assert calls["cmd"][3:6] == ["has-session", "-t", "aviary"]

    monkeypatch.setattr(sc.subprocess, "run", lambda cmd, **kw: _R(1))
    assert sc._session_running() is False


def test_server_skips_attach_inside_the_pane(monkeypatch):
    """With INNER_ENV set (we ARE the bg server), server() must run main, not attach."""
    monkeypatch.setenv(sc.INNER_ENV, "1")
    monkeypatch.setattr(sc, "_acquire_foreground_lock", lambda: True)

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


def test_server_refuses_second_foreground_when_lock_held(monkeypatch):
    """A direct `uv run server` must not start when another foreground server holds
    the lock (it would fight over the cameras)."""
    monkeypatch.delenv(sc.INNER_ENV, raising=False)
    monkeypatch.setattr(sc.sys, "platform", "linux")
    monkeypatch.setattr(sc.shutil, "which", lambda _name: "/usr/bin/tmux")
    monkeypatch.setattr(sc, "_session_running", lambda: False)
    monkeypatch.setattr(sc, "_acquire_foreground_lock", lambda: False)

    fake_main = types.ModuleType("lib.main")
    fake_main.main = lambda: (_ for _ in ()).throw(AssertionError("main ran"))
    monkeypatch.setitem(sys.modules, "lib.main", fake_main)

    try:
        sc.server()
    except SystemExit as exc:
        assert "already running" in str(exc.code)
    else:
        raise AssertionError("server() should have exited when the lock was held")


def test_foreground_lock_detects_another_holder(monkeypatch, tmp_path):
    """_foreground_server_running is True only while another fd holds the flock."""
    lock = tmp_path / "aviary-server.lock"
    monkeypatch.setattr(sc, "_lock_path", lambda: lock)

    # Nobody holds it yet.
    assert sc._foreground_server_running() is False

    # Simulate a live foreground server holding the lock, then releasing it.
    import fcntl

    holder = open(lock, "a+")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert sc._foreground_server_running() is True
    finally:
        holder.close()
    assert sc._foreground_server_running() is False


def test_up_relaunches_with_restart_and_clears_legacy(monkeypatch, tmp_path):
    """up() must use `systemctl restart` (NOT the no-op `enable --now` on an
    already-active unit) and tear down any legacy `-L aviary` session."""
    monkeypatch.setattr(sc, "_require_systemd", lambda: None)
    monkeypatch.setattr(sc, "_tmux", lambda: "/usr/bin/tmux")
    monkeypatch.setattr(sc, "_uv", lambda: "/home/u/.local/bin/uv")
    monkeypatch.setattr(sc, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(sc, "_session_running", lambda: False)
    monkeypatch.setattr(sc, "_foreground_server_running", lambda: False)
    monkeypatch.setattr(sc, "_write_generated_files", lambda *a, **k: None)
    monkeypatch.setattr(sc, "_ensure_linger", lambda: None)
    monkeypatch.setattr(sc, "_wait_for_session", lambda *a, **k: True)
    monkeypatch.setattr(sc.shutil, "which", lambda _name: "/usr/bin/tmux")

    cmds = []

    class _R:
        returncode = 0

    monkeypatch.setattr(sc.subprocess, "run", lambda cmd, **kw: cmds.append(cmd) or _R())

    sc.up()

    assert ["/usr/bin/tmux", "-L", sc.LEGACY_SOCKET, "kill-server"] in cmds
    assert ["systemctl", "--user", "restart", sc.SERVICE] in cmds
    # The original no-op path must be gone.
    assert ["systemctl", "--user", "enable", "--now", sc.SERVICE] not in cmds


def test_up_refuses_when_foreground_server_running(monkeypatch, tmp_path):
    """A live foreground `uv run server` must block server-up (no second camera war)."""
    monkeypatch.setattr(sc, "_require_systemd", lambda: None)
    monkeypatch.setattr(sc, "_tmux", lambda: "/usr/bin/tmux")
    monkeypatch.setattr(sc, "_uv", lambda: "/home/u/.local/bin/uv")
    monkeypatch.setattr(sc, "_session_running", lambda: False)
    monkeypatch.setattr(sc, "_foreground_server_running", lambda: True)

    started = []
    monkeypatch.setattr(sc.subprocess, "run", lambda cmd, **kw: started.append(cmd))

    try:
        sc.up()
    except SystemExit as exc:
        assert "foreground" in str(exc.code)
    else:
        raise AssertionError("up() should refuse when a foreground server is running")
    assert started == []  # nothing was enabled/started


def test_down_kills_legacy_socket_when_tmux_present(monkeypatch, tmp_path):
    monkeypatch.setattr(sc, "_require_systemd", lambda: None)
    monkeypatch.setattr(sc.shutil, "which", lambda _name: "/usr/bin/tmux")
    monkeypatch.setattr(sc, "_tmux", lambda: "/usr/bin/tmux")
    monkeypatch.setattr(sc, "_session_running", lambda: False)
    monkeypatch.setattr(sc, "_unit_path", lambda: tmp_path / "aviary.service")
    monkeypatch.setattr(sc, "_launcher_path", lambda: tmp_path / "run-server.sh")
    monkeypatch.setattr(sc, "_tmux_conf_path", lambda: tmp_path / "tmux.conf")
    monkeypatch.setattr(sc, "_gen_dir", lambda: tmp_path / "nope")

    cmds = []
    monkeypatch.setattr(sc.subprocess, "run", lambda cmd, **kw: cmds.append(cmd))

    sc.down()
    assert ["/usr/bin/tmux", "-L", sc.LEGACY_SOCKET, "kill-server"] in cmds


def test_down_tears_down_unit_even_when_tmux_missing(monkeypatch, tmp_path):
    """server-down must remove the autostart unit even if tmux is no longer installed
    — `had` must not call into tmux when it is absent."""
    monkeypatch.setattr(sc, "_require_systemd", lambda: None)
    monkeypatch.setattr(sc.shutil, "which", lambda _name: None)  # tmux + systemctl gone from PATH

    unit = tmp_path / "aviary.service"
    unit.write_text("[Service]\n")
    monkeypatch.setattr(sc, "_unit_path", lambda: unit)
    monkeypatch.setattr(sc, "_launcher_path", lambda: tmp_path / "run-server.sh")
    monkeypatch.setattr(sc, "_tmux_conf_path", lambda: tmp_path / "tmux.conf")
    monkeypatch.setattr(sc, "_gen_dir", lambda: tmp_path / "does-not-exist")

    # _session_running must NOT be reached (it would sys.exit via _tmux()).
    monkeypatch.setattr(
        sc, "_session_running", lambda: (_ for _ in ()).throw(AssertionError("tmux probed"))
    )

    ran = []
    monkeypatch.setattr(sc.subprocess, "run", lambda cmd, **kw: ran.append(cmd))

    sc.down()  # must not raise
    # The systemctl disable/daemon-reload/reset-failed teardown still ran.
    assert ["systemctl", "--user", "disable", "--now", sc.SERVICE] in ran
    assert not unit.exists()  # unit removed
