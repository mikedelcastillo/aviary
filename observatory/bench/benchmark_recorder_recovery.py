#!/usr/bin/env python3
"""Offline child recovery experiment. Never reads a camera or changes a service."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.recorder import command_for, supervise
from backend.recorder_health import RecorderLimits, child_memory, playlist_version, stop_child


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.fixture.is_file(): parser.error('An existing offline video fixture is required')
    args.output.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite another run or point the experiment at runtime recordings.
    if any(args.output.iterdir()): parser.error('Output must be an empty isolated directory')
    limits = RecorderLimits()
    config = {'runtime': str(args.output), 'relay_rtsp': ''}
    processes = []
    traces = []
    errors = []
    began = time.monotonic()

    def launch(name):
        command = command_for(config, {'id': name}, 'libx264', str(args.fixture.resolve()), ['-re', '-stream_loop', '-1'], wallclock=False)
        log = (args.output/f'{name}.stderr').open('wb')
        errors.append(log)
        process = subprocess.Popen(command, start_new_session=True, stdout=subprocess.DEVNULL, stderr=log,
                                   env={**os.environ, 'TZ': 'UTC'})
        processes.append(process)
        return process

    def wait_output(name, process, timeout=15):
        playlist = args.output/'hls'/name/'live.m3u8'
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if process.poll() is not None: raise RuntimeError(f'{name} exited during startup')
            if playlist.exists(): return playlist
            time.sleep(.1)
        raise RuntimeError(f'{name} did not publish HLS in {timeout}s')

    try:
        healthy = launch('healthy')
        healthy_playlist = wait_output('healthy', healthy)
        first = launch('restarting')
        playlist = wait_output('restarting', first)
        initial = playlist_version(healthy_playlist)
        # Freeze a real encoder rather than touching a camera or relay.
        os.kill(first.pid, __import__('signal').SIGSTOP)
        stalled_started = time.monotonic()
        reason = supervise(first, playlist, RecorderLimits(startup_seconds=2, shutdown_seconds=.4, poll_seconds=.05),
                           lambda row: traces.append({'elapsed': round(time.monotonic() - began, 3), **row}))
        recovery_started = time.monotonic()
        old_version = playlist_version(playlist)
        replacement = launch('restarting')
        end = time.monotonic() + 15
        while time.monotonic() < end and playlist_version(playlist) == old_version:
            if replacement.poll() is not None: raise RuntimeError('Replacement exited before new HLS')
            time.sleep(.1)
        if playlist_version(playlist) == old_version: raise RuntimeError('Replacement never published fresh HLS')
        recovery_seconds = time.monotonic() - recovery_started
        if healthy.poll() is not None or playlist_version(healthy_playlist) == initial:
            raise RuntimeError('Healthy sibling stopped advancing during recovery')
        # Allow one complete 20s MP4 segment before verifying finalized output.
        remaining = max(0, 24 - (time.monotonic() - began))
        time.sleep(remaining)
        memory = {'healthy': child_memory(healthy.pid), 'replacement': child_memory(replacement.pid)}
        stop_child(replacement, limits); stop_child(healthy, limits)
        clips = []
        for path in (args.output/'recordings').glob('*/*.mp4'):
            check = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(path)], capture_output=True, text=True)
            if check.returncode:
                # Only the forcibly stopped encoder's incomplete MP4 is expected to be unreadable.
                clips.append({'camera': path.parent.name, 'readable': False})
            else:
                clips.append({'camera': path.parent.name, 'readable': True, 'duration': float(json.loads(check.stdout)['format']['duration'])})
        if not any(row['camera'] == 'healthy' and row['readable'] and row['duration'] >= 19 for row in clips):
            raise RuntimeError('Healthy camera did not retain its complete 20s recording')
        if not any(row['camera'] == 'restarting' and row['readable'] for row in clips):
            raise RuntimeError('Replacement did not finalize a readable recording')
        result = {'stop_reason': reason, 'stop_seconds': round(recovery_started - stalled_started, 3),
                  'fresh_hls_after_restart_seconds': round(recovery_seconds, 3), 'healthy_sibling_continued': True,
                  'memory': memory, 'clips': clips, 'trace': traces}
        (args.output/'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({key: value for key, value in result.items() if key != 'trace'}, indent=2))
    finally:
        for process in processes:
            if process.poll() is None: stop_child(process, limits, emergency=True)
        for log in errors: log.close()


if __name__ == '__main__': main()
