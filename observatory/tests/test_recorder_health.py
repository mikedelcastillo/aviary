"""Fault injection for one recorder child, independent of cameras and services."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.recorder import command_for, supervise
from backend.recorder_health import RecorderLimits, RecorderWatchdog, child_memory, stop_child


class RecorderPolicyTests(unittest.TestCase):
    def test_old_playlist_does_not_count_as_child_startup(self):
        policy = RecorderWatchdog(RecorderLimits(), 100, (123, 400))
        self.assertIsNone(policy.observe(114.9, (123, 400), None))
        self.assertEqual('startup-timeout', policy.observe(115, (123, 400), None))
        self.assertFalse(policy.output_started)

    def test_new_output_stall_uses_monotonic_deadline(self):
        policy = RecorderWatchdog(RecorderLimits(), 100, (123, 400))
        self.assertIsNone(policy.observe(105, (124, 420), None))
        self.assertTrue(policy.output_started)
        self.assertIsNone(policy.observe(116.9, (124, 420), None))
        self.assertEqual('output-stalled', policy.observe(117, (124, 420), None))

    def test_private_and_emergency_limits_ignore_shared_rss(self):
        limits = RecorderLimits()
        policy = RecorderWatchdog(limits, 100, None)
        self.assertIsNone(policy.observe(101, None, {'private_bytes': 250 * 1024**2, 'rss_bytes': 600 * 1024**2}))
        self.assertEqual('memory-limit', policy.observe(101, None, {'private_bytes': limits.private_bytes}))
        self.assertEqual('memory-emergency', policy.observe(101, None, {'private_bytes': limits.emergency_bytes}))

    def test_limits_validate_before_starting_a_child(self):
        with self.assertRaises(ValueError): RecorderLimits.from_config({'recorder_limits': {'poll_seconds': 0}})
        with self.assertRaises(ValueError): RecorderLimits.from_config({'recorder_limits': {'private_bytes': 100, 'emergency_bytes': 50}})

    def test_discovery_and_output_queues_are_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            command = command_for({'runtime': tmp, 'relay_rtsp': 'rtsp://localhost:8554'}, {'id': 'fixture'}, 'libx264')
            for flag, value in {'-filter_threads': '2', '-probesize': '1048576', '-analyzeduration': '2000000',
                                '-max_muxing_queue_size': '128', '-muxing_queue_data_threshold': '1048576'}.items():
                self.assertEqual(value, command[command.index(flag) + 1])
            self.assertLess(command.index('-probesize'), command.index('-i'))
            self.assertGreater(command.index('-max_muxing_queue_size'), command.index('-i'))
            self.assertIn('-y', command)


@unittest.skipUnless(sys.platform == 'linux', 'Linux process and /proc isolation checks')
class RecorderFaultTests(unittest.TestCase):
    def child(self, source):
        process = subprocess.Popen([sys.executable, '-c', source], start_new_session=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: stop_child(process, RecorderLimits(shutdown_seconds=.2), emergency=True) if process.poll() is None else None)
        return process

    def test_stalled_start_stops_only_its_own_group(self):
        healthy = self.child('import time; time.sleep(30)')
        stalled = self.child('import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); time.sleep(30)')
        samples = []
        with tempfile.TemporaryDirectory() as tmp:
            playlist = Path(tmp)/'live.m3u8'; playlist.write_text('old playlist')
            began = time.monotonic()
            reason = supervise(stalled, playlist, RecorderLimits(startup_seconds=.6, shutdown_seconds=.3, poll_seconds=.05), samples.append)
        self.assertEqual('startup-timeout', reason)
        self.assertLess(time.monotonic() - began, 2)
        self.assertEqual(-signal.SIGKILL, stalled.returncode)
        self.assertIsNone(healthy.poll())
        self.assertTrue(samples[-1]['memory']['private_bytes'] > 0)
        self.assertFalse(samples[-1]['output_started'])

    def test_memory_growth_is_reclaimed_before_an_unbounded_restart(self):
        healthy = self.child('import time; time.sleep(30)')
        growing = self.child('import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); blocks=[]\nwhile True:\n blocks.append(bytearray(1024*1024)); time.sleep(.01)')
        samples = []
        limits = RecorderLimits(startup_seconds=10, private_bytes=32 * 1024**2, emergency_bytes=48 * 1024**2,
                                shutdown_seconds=.5, poll_seconds=.02)
        with tempfile.TemporaryDirectory() as tmp:
            reason = supervise(growing, Path(tmp)/'missing.m3u8', limits, samples.append)
        self.assertEqual('memory-limit', reason)
        self.assertEqual(-signal.SIGKILL, growing.returncode)
        self.assertIsNone(healthy.poll())
        self.assertLess(max(row['memory']['private_bytes'] for row in samples), 40 * 1024**2)
        self.assertIsNone(child_memory(growing.pid))

    def test_graceful_shutdown_gives_child_time_to_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp)/'closed'
            source = ('import signal,time,pathlib,sys\n'
                      f'def done(*_): pathlib.Path({str(marker)!r}).write_text("flushed"); sys.exit(0)\n'
                      'signal.signal(signal.SIGINT,done)\nprint("ready",flush=True)\ntime.sleep(30)')
            process = subprocess.Popen([sys.executable, '-c', source], start_new_session=True, stdout=subprocess.PIPE)
            self.assertEqual(b'ready\n', process.stdout.readline())
            self.assertEqual(0, stop_child(process, RecorderLimits(shutdown_seconds=1)))
            process.stdout.close()
            self.assertEqual('flushed', marker.read_text())


if __name__ == '__main__': unittest.main()
