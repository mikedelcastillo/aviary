"""Per-child recorder supervision. All deadlines use a monotonic clock."""
from dataclasses import dataclass
from pathlib import Path
import os
import signal
import subprocess
import time


@dataclass(frozen=True)
class RecorderLimits:
    startup_seconds: float = 15
    stale_seconds: float = 12
    poll_seconds: float = .5
    shutdown_seconds: float = 3
    private_bytes: int = 384 * 1024**2
    emergency_bytes: int = 512 * 1024**2

    @classmethod
    def from_config(cls, config):
        values = config.get('recorder_limits', {})
        limits = cls(**{key: value for key, value in values.items() if key in cls.__dataclass_fields__})
        if min(limits.startup_seconds, limits.stale_seconds, limits.poll_seconds, limits.shutdown_seconds, limits.private_bytes) <= 0:
            raise ValueError('Recorder limits must be positive')
        if limits.emergency_bytes < limits.private_bytes:
            raise ValueError('Recorder emergency memory must exceed its graceful limit')
        return limits


def child_memory(pid):
    """Shared driver mappings inflate RSS; private anonymous plus swap is the guard."""
    try:
        fields = {}
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            key, _, value = line.partition(':')
            if key in {'VmRSS', 'RssAnon', 'RssFile', 'RssShmem', 'VmSwap'}:
                fields[key] = int(value.split()[0]) * 1024
        if 'VmRSS' not in fields:
            return None
        anonymous = fields.get('RssAnon', max(0, fields['VmRSS'] - fields.get('RssFile', 0) - fields.get('RssShmem', 0)))
        return {'rss_bytes': fields['VmRSS'], 'anonymous_bytes': anonymous,
                'swap_bytes': fields.get('VmSwap', 0), 'private_bytes': anonymous + fields.get('VmSwap', 0)}
    except (OSError, ValueError, IndexError):
        return None


def playlist_version(path):
    try:
        stat = Path(path).stat()
        return stat.st_mtime_ns, stat.st_size
    except FileNotFoundError:
        return None


class RecorderWatchdog:
    def __init__(self, limits, started, prior_playlist):
        self.limits = limits
        self.started = started
        self.previous_playlist = prior_playlist
        self.last_output = started
        self.output_started = False

    def observe(self, now, playlist, memory):
        if playlist is not None and playlist != self.previous_playlist:
            self.previous_playlist = playlist
            self.last_output = now
            self.output_started = True
        private = (memory or {}).get('private_bytes', 0)
        if private >= self.limits.emergency_bytes:
            return 'memory-emergency'
        if private >= self.limits.private_bytes:
            return 'memory-limit'
        if not self.output_started and now - self.started >= self.limits.startup_seconds:
            return 'startup-timeout'
        if self.output_started and now - self.last_output >= self.limits.stale_seconds:
            return 'output-stalled'
        return None


def stop_child(proc, limits, emergency=False):
    """Stop only this child's new process group, escalating within a bounded grace."""
    if proc.poll() is not None:
        return proc.returncode

    def send(sig):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass

    send(signal.SIGKILL if emergency else signal.SIGINT)
    deadline = time.monotonic() + (0 if emergency else limits.shutdown_seconds)
    while proc.poll() is None and time.monotonic() < deadline:
        memory = child_memory(proc.pid)
        if memory and memory['private_bytes'] >= limits.emergency_bytes:
            break
        time.sleep(min(.1, limits.poll_seconds))
    if proc.poll() is None:
        send(signal.SIGKILL)
    try:
        return proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        # A kernel-uninterruptible child must not be replaced with another decoder.
        raise RuntimeError('Recorder child did not exit after SIGKILL') from None
