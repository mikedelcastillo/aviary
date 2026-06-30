"""Background lifecycle for the camera server: ``server``, ``server-up``, ``server-down``.

``uv run server`` runs the dashboard in the foreground exactly as before — UNLESS a
background instance is already running, in which case it attaches to that instead of
starting a second server that would fight over the cameras.

``uv run server-up`` runs the server in the background and registers it to start on
boot. The dashboard is a full-screen Rich TUI that needs a real terminal, so the
background server runs inside a dedicated tmux session supervised by a systemd *user*
service: ``Type=forking`` lets systemd track the long-lived tmux server as the unit's
main process, so a crashed or killed session is reported honestly and restarted
(``Restart=on-failure``) rather than leaving the unit stuck at a false ``active
(exited)``. App-level crashes are absorbed in place by the launcher's respawn loop;
systemd restarts the whole session if the tmux server itself dies abnormally (e.g. a
crash or OOM-kill). A clean shutdown (server exit 0) stops and is not respawned. Boot
autostart works because the user already has linger enabled. Re-running ``server-up`` while it is
up just attaches to the live dashboard. Inside the attached view, Esc or Ctrl-C
detaches and leaves the server running (the dashboard is output-only, so those keys are
repurposed to detach instead of being delivered to the process).

``uv run server-down`` stops the background server and removes the autostart unit.

Only this module is imported to decide *whether* to attach, so the heavy
``lib.main`` import (cv2/torch/ultralytics) is deferred until we actually run the
server in the foreground.
"""

from __future__ import annotations

import getpass
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Dedicated tmux session name, kept separate from any tmux the user runs
# interactively so our key rebindings and teardown never touch their sessions.
SESSION = "aviary"
SERVICE = "aviary.service"

# Pre-upgrade builds ran the session on `tmux -L aviary` (a socket NAME resolved
# relative to $TMUX_TMPDIR) instead of today's pinned absolute socket. Kept only so an
# in-place upgrade can tear that legacy session down — otherwise it would be orphaned
# and run a second camera server alongside the new one.
LEGACY_SOCKET = "aviary"

# Set inside the background tmux pane so the in-pane `uv run server` runs the real
# server instead of seeing its own session and recursively attaching.
INNER_ENV = "AVIARY_SERVER_INNER"

# Held open for the lifetime of a foreground server so a second camera-using server
# can detect us via flock (see _acquire_foreground_lock). Module-global so the fd is
# never garbage-collected — closing it would release the lock.
_FOREGROUND_LOCK = None  # type: ignore[var-annotated]


def _config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def _gen_dir() -> Path:
    return _config_home() / "aviary"


def _launcher_path() -> Path:
    return _gen_dir() / "run-server.sh"


def _tmux_conf_path() -> Path:
    return _gen_dir() / "tmux.conf"


def _unit_path() -> Path:
    return _config_home() / "systemd" / "user" / SERVICE


def _runtime_dir() -> Path:
    # The systemd user manager and an interactive logind session both set
    # XDG_RUNTIME_DIR to /run/user/<uid>, so resolving it here keeps every entry
    # point (the unit, `server`, `server-up`, `server-down`) in agreement.
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if not runtime:
        runtime = f"/run/user/{os.getuid()}"
    return Path(runtime)


def _socket_path() -> str:
    # Absolute tmux socket (NOT `-L name`, which is resolved relative to $TMUX_TMPDIR).
    # Pinning it absolutely guarantees the systemd unit and every interactive command
    # resolve the SAME socket even if a shell exports TMUX_TMPDIR.
    return str(_runtime_dir() / "aviary-tmux.sock")


def _lock_path() -> Path:
    return _runtime_dir() / "aviary-server.lock"


# --------------------------------------------------------------------------- #
# Tool discovery
# --------------------------------------------------------------------------- #
def _tmux() -> str:
    path = shutil.which("tmux")
    if not path:
        sys.exit(
            "tmux is required for the background server. Install it "
            "(e.g. `sudo apt install tmux`) or use `uv run server` in the foreground."
        )
    return path


def _uv() -> str:
    path = shutil.which("uv")
    if not path:
        sys.exit("Could not locate the `uv` binary on PATH.")
    return path


def _repo_root() -> Path:
    # server-up is launched via `uv run` from the repo, so cwd is the repo root.
    root = Path.cwd().resolve()
    if not (root / "pyproject.toml").is_file():
        sys.exit(f"Run this from the aviary repo root (no pyproject.toml in {root}).")
    return root


def _require_systemd() -> None:
    if sys.platform != "linux" or not shutil.which("systemctl"):
        sys.exit(
            "Background mode + boot autostart need systemd (Linux). "
            "Use `uv run server` to run in the foreground."
        )
    if not os.environ.get("XDG_RUNTIME_DIR"):
        sys.exit(
            "No systemd user session (XDG_RUNTIME_DIR unset). "
            "Log in on a normal desktop/SSH session and retry, or use `uv run server`."
        )


# --------------------------------------------------------------------------- #
# tmux session helpers
# --------------------------------------------------------------------------- #
def _session_running() -> bool:
    """True iff our dedicated tmux session exists (cheap, no heavy imports)."""
    return (
        subprocess.run(
            [_tmux(), "-S", _socket_path(), "has-session", "-t", SESSION],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _attach() -> None:
    """Replace this process with a tmux client attached to the background dashboard."""
    tmux = _tmux()
    print(
        "Attached to the background server — press Esc or Ctrl-C to detach "
        "(the server keeps running).",
        file=sys.stderr,
    )
    os.execv(tmux, [tmux, "-S", _socket_path(), "attach", "-t", SESSION])


def _wait_for_session(timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _session_running():
            return True
        time.sleep(0.25)
    return _session_running()


def _kill_legacy_session() -> None:
    """Tear down a pre-upgrade ``tmux -L aviary`` session (older builds used a
    TMUX_TMPDIR-relative socket name instead of the pinned absolute socket). Without
    this, an in-place upgrade would orphan that session and run two camera servers at
    once. Best-effort — harmless when no legacy session exists."""
    if not shutil.which("tmux"):
        return
    subprocess.run(
        [_tmux(), "-L", LEGACY_SOCKET, "kill-server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# --------------------------------------------------------------------------- #
# Foreground single-instance lock
# --------------------------------------------------------------------------- #
# A direct `uv run server` (no tmux session) is invisible to _session_running, so
# without this two camera servers could run at once — fighting over the RTSP streams
# and the single Telegram getUpdates consumer. The foreground server holds an
# exclusive flock for its whole lifetime; `server-up`/`server` test it before starting
# another. flock is released automatically when the holder exits (even on SIGKILL), so
# there is no stale-lock problem.
def _acquire_foreground_lock() -> bool:
    """Hold the foreground-server lock for this process. Returns False only when
    another live server already holds it; True (proceed) when acquired, or when the
    platform has no flock / the runtime dir is unavailable (best-effort)."""
    global _FOREGROUND_LOCK
    try:
        import fcntl
    except ImportError:
        return True
    try:
        fh = open(_lock_path(), "a+")
    except OSError:
        return True
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return False
    fh.seek(0)
    fh.truncate()
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    _FOREGROUND_LOCK = fh  # kept open -> released only when this process exits
    return True


def _foreground_server_running() -> bool:
    """True iff another process holds the foreground-server lock."""
    try:
        import fcntl
    except ImportError:
        return False
    try:
        fh = open(_lock_path(), "a")
    except OSError:
        return False
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    else:
        fcntl.flock(fh, fcntl.LOCK_UN)
        return False
    finally:
        fh.close()


# --------------------------------------------------------------------------- #
# Generated file contents (pure — unit tested)
# --------------------------------------------------------------------------- #
def render_tmux_conf() -> str:
    return (
        "# Generated by `uv run server-up` — dedicated tmux server for the aviary\n"
        "# monitor. The dashboard is output-only, so Ctrl-C and Escape are repurposed\n"
        "# to DETACH (leave the background server running) instead of being delivered\n"
        "# to the process.\n"
        "set -g escape-time 50\n"
        "set -g mouse on\n"
        "set -g history-limit 20000\n"
        "set -g window-size latest\n"
        "set -g status on\n"
        "set -g status-style 'bg=colour236 fg=colour108'\n"
        "set -g status-left ''\n"
        "set -g status-right ' aviary · Esc / Ctrl-C: detach (server keeps running) '\n"
        "set -g status-right-length 60\n"
        "set -g window-status-format ''\n"
        "set -g window-status-current-format ''\n"
        "bind-key -n C-c detach-client\n"
        "bind-key -n Escape detach-client\n"
    )


def render_launcher(repo: Path, uv: str) -> str:
    return (
        "#!/usr/bin/env bash\n"
        "# Generated by `uv run server-up`. Runs the aviary dashboard in the foreground\n"
        "# inside the tmux session and restarts it if it exits unexpectedly.\n"
        "set -u\n"
        f"cd {shlex.quote(str(repo))} || exit 1\n"
        f"export {INNER_ENV}=1\n"
        "while true; do\n"
        f"  {shlex.quote(uv)} run server\n"
        "  code=$?\n"
        '  [ "$code" -eq 0 ] && exit 0\n'
        "  printf '\\n[aviary] server exited (%s); restarting in 3s...\\n' \"$code\" >&2\n"
        "  sleep 3\n"
        "done\n"
    )


def render_unit(repo: Path, tmux: str, uv: str) -> str:
    # systemd user services get a minimal PATH; bake in the dir holding uv plus the
    # usual locations so the server (and anything it shells out to) resolves.
    dirs = [
        str(Path(uv).parent),
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
        f"{Path.home()}/.local/bin",
    ]
    path = os.pathsep.join(dict.fromkeys(dirs))  # de-dup, order preserved
    sock = _socket_path()
    return (
        "[Unit]\n"
        "Description=Aviary camera monitor (background tmux session)\n"
        # Never give up: an unattended monitor must not silently stay dead after a
        # transient crash loop, so disable systemd's start-rate limit.
        "StartLimitIntervalSec=0\n"
        "\n"
        "[Service]\n"
        # `tmux new-session -d` forks the long-lived tmux server (the cgroup root),
        # which systemd then tracks as the main PID. So the unit reflects the real
        # session state and `Restart=on-failure` resurrects a crashed/killed session
        # instead of leaving a false `active (exited)`.
        "Type=forking\n"
        f"WorkingDirectory={repo}\n"
        f"Environment=PATH={path}\n"
        f"ExecStart={tmux} -S {sock} -f {_tmux_conf_path()} new-session -d "
        f"-s {SESSION} -x 250 -y 60 {_launcher_path()}\n"
        # `-` prefix: after a clean shutdown the session is already gone, so
        # kill-server exits non-zero ("no server running"); ignore it so a clean
        # `/quit` is not mis-recorded as a failure (which would wrongly Restart).
        f"ExecStop=-{tmux} -S {sock} kill-server\n"
        "Restart=on-failure\n"
        "RestartSec=5\n"
        "TimeoutStartSec=60\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def _write_generated_files(repo: Path, uv: str, tmux: str) -> None:
    _gen_dir().mkdir(parents=True, exist_ok=True)
    _unit_path().parent.mkdir(parents=True, exist_ok=True)

    launcher = _launcher_path()
    launcher.write_text(render_launcher(repo, uv))
    launcher.chmod(0o755)

    _tmux_conf_path().write_text(render_tmux_conf())
    _unit_path().write_text(render_unit(repo, tmux, uv))


def _ensure_linger() -> None:
    user = getpass.getuser()
    warning = (
        "⚠ Could not enable linger; the server may not auto-start on boot. "
        f"Run manually:  sudo loginctl enable-linger {user}"
    )
    try:
        cur = subprocess.run(
            ["loginctl", "show-user", user, "-p", "Linger"],
            capture_output=True,
            text=True,
        )
    except OSError:
        # loginctl missing (some minimal/container setups) — warn, don't crash.
        print(warning, file=sys.stderr)
        return
    if "Linger=yes" in cur.stdout:
        return
    try:
        rc = subprocess.run(["loginctl", "enable-linger", user]).returncode
    except OSError:
        rc = 1
    if rc != 0:
        print(warning, file=sys.stderr)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def server() -> None:
    """`uv run server`: attach to the background server if one is up, else run it here."""
    if (
        not os.environ.get(INNER_ENV)
        and sys.platform == "linux"
        and shutil.which("tmux")
        and _session_running()
    ):
        _attach()  # replaces this process; never returns
        return

    if not _acquire_foreground_lock():
        sys.exit(
            "An aviary server is already running here (it holds the camera lock).\n"
            "Attach with `uv run server-up`, or stop it before starting another."
        )

    from lib.main import main

    main()


def up() -> None:
    """`uv run server-up`: start the server in the background + enable boot autostart."""
    _require_systemd()
    tmux = _tmux()
    uv = _uv()

    if _session_running():
        _attach()  # already up — just show the dashboard; never returns
        return

    if _foreground_server_running():
        sys.exit(
            "A foreground `uv run server` is already running in this session.\n"
            "Stop it first (Ctrl-C in that terminal) — otherwise two servers would "
            "fight over the cameras and the Telegram bot.\n"
            "Then re-run `uv run server-up`."
        )

    repo = _repo_root()
    _write_generated_files(repo, uv, tmux)
    _ensure_linger()

    # An older build ran the session on `tmux -L aviary`; the new unit only knows the
    # pinned absolute socket, so kill the legacy session here or an upgrade leaves it
    # orphaned and two camera servers fight over the cameras.
    _kill_legacy_session()

    try:
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", SERVICE], check=True)
        # A prior crash-loop can leave the unit `failed`; clear it so the restart
        # below is not blocked. Best-effort — ignore if there was nothing to reset.
        subprocess.run(
            ["systemctl", "--user", "reset-failed", SERVICE],
            stderr=subprocess.DEVNULL,
        )
        # `restart` (not `enable --now`): starting an already-active unit is a no-op
        # that would NOT re-run ExecStart, so a stale session never relaunches.
        # `restart` always re-runs ExecStart and recreates the tmux session.
        subprocess.run(["systemctl", "--user", "restart", SERVICE], check=True)
    except subprocess.CalledProcessError as exc:
        sys.exit(
            f"Failed to start the background service "
            f"(`{' '.join(exc.cmd)}` exited {exc.returncode}).\n"
            "Check it with:\n"
            "  systemctl --user status aviary.service\n"
            "  journalctl --user -u aviary.service -e"
        )

    if not _wait_for_session():
        sys.exit(
            "Background server did not come up in time. Check the service with:\n"
            "  systemctl --user status aviary.service\n"
            "  journalctl --user -u aviary.service -e"
        )

    print(
        "✓ Aviary server started in the background and enabled on boot.\n"
        f"  Logs:    {repo / 'aviary.log'}\n"
        "  View:    uv run server-up   (or `uv run server`) to attach; "
        "Esc / Ctrl-C detaches\n"
        "  Stop:    uv run server-down"
    )


def down() -> None:
    """`uv run server-down`: stop the background server and remove boot autostart."""
    _require_systemd()
    # Probe the session only when tmux is present — teardown of the systemd unit must
    # not depend on tmux still being installed (otherwise the autostart can't be
    # removed once tmux is gone).
    had = (bool(shutil.which("tmux")) and _session_running()) or _unit_path().exists()

    subprocess.run(
        ["systemctl", "--user", "disable", "--now", SERVICE],
        stderr=subprocess.DEVNULL,
    )
    # Tear down the dedicated tmux server (session + panes) in case it was started
    # outside systemd; this only touches our private socket, never the user's.
    if shutil.which("tmux"):
        subprocess.run(
            [_tmux(), "-S", _socket_path(), "kill-server"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    # Also clear a pre-upgrade `-L aviary` session so it can't linger after teardown.
    _kill_legacy_session()

    _unit_path().unlink(missing_ok=True)
    _launcher_path().unlink(missing_ok=True)
    _tmux_conf_path().unlink(missing_ok=True)
    try:
        _gen_dir().rmdir()
    except OSError:
        pass

    subprocess.run(["systemctl", "--user", "daemon-reload"])
    subprocess.run(
        ["systemctl", "--user", "reset-failed", SERVICE],
        stderr=subprocess.DEVNULL,
    )

    print(
        "✓ Background server stopped and boot autostart removed."
        if had
        else "Nothing to stop — no background server was running."
    )
