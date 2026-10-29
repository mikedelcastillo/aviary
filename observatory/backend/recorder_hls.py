"""Atomic, bounded HLS publication across independently restarted encoders.

FFmpeg 6.1 does not maintain EXT-X-DISCONTINUITY-SEQUENCE. Keep its incoming
playlist private and publish a durable sliding window with stable segment IDs.
"""
from dataclasses import dataclass, replace
from datetime import datetime
import math
import logging
import os
from pathlib import Path
import re
import time


@dataclass(frozen=True)
class Segment:
    sequence: int
    uri: str
    duration: float
    date: str
    discontinuity: int

    @property
    def timestamp(self):
        return datetime.fromisoformat(self.date.replace('Z', '+00:00')).timestamp()


def read_playlist(path):
    with Path(path).open() as stream:
        text = stream.read(64 * 1024 + 1)
    if len(text) > 64 * 1024 or not text.startswith('#EXTM3U\n'):
        raise ValueError('Invalid recorder playlist')
    sequence, discontinuity, duration, date, target = 0, 0, None, None, 2
    segments = []
    for line in text.splitlines():
        if line.startswith('#EXT-X-MEDIA-SEQUENCE:'):
            sequence = int(line.split(':', 1)[1])
        elif line.startswith('#EXT-X-DISCONTINUITY-SEQUENCE:'):
            discontinuity = int(line.split(':', 1)[1])
        elif line == '#EXT-X-DISCONTINUITY':
            discontinuity += 1
        elif line.startswith('#EXT-X-TARGETDURATION:'):
            target = int(line.split(':', 1)[1])
        elif line.startswith('#EXT-X-PROGRAM-DATE-TIME:'):
            date = line.split(':', 1)[1]
        elif line.startswith('#EXTINF:'):
            duration = float(line.split(':', 1)[1].split(',')[0])
        elif line and not line.startswith('#'):
            if (not re.fullmatch(r's\d+\.ts', line) or duration is None or not date
                    or not math.isfinite(duration) or not 0 < duration <= 10
                    or sequence < 0 or discontinuity < 0):
                raise ValueError('Invalid recorder segment')
            segment = Segment(sequence, line, duration, date, discontinuity)
            if not math.isfinite(segment.timestamp):
                raise ValueError('Invalid recorder date')
            segments.append(segment)
            sequence += 1
            duration, date = None, None
    if not 1 <= target <= 10 or len(segments) > 100:
        raise ValueError('Invalid recorder window')
    return segments, target


class HLSPublisher:
    def __init__(self, playlist, *, window=10, stale_history_seconds=20):
        self.playlist = Path(playlist)
        self.folder = self.playlist.parent
        self.folder.mkdir(parents=True, exist_ok=True)
        self.incoming = self.folder/'incoming.m3u8'
        self.checkpoint = self.folder/'continuity.m3u8'
        self.window = window
        self.stale_history_seconds = stale_history_seconds
        self.segments, self.target = [], 2
        for candidate in (self.playlist, self.checkpoint):
            try:
                segments, target = read_playlist(candidate)
                if not segments:
                    raise ValueError('Empty recorder playlist')
                self.segments, self.target = segments, target
                break
            except FileNotFoundError:
                continue
            except (ValueError, OverflowError, UnicodeError):
                if candidate == self.playlist:
                    self.playlist.replace(self.folder/'live.m3u8.invalid')
                    logging.getLogger(__name__).warning('Invalid persisted HLS playlist quarantined; recovering bounded continuity')
        if self.segments:
            self.start = self.segments[-1].sequence + 1
            self.generation = self.segments[-1].discontinuity + 1
        else:
            existing = [int(match[1]) for path in self.folder.iterdir()
                        if (match := re.fullmatch(r's(\d+)\.ts(?:\.tmp)?', path.name))]
            # Media numbers represent consecutive segments, not wall-clock
            # microseconds. Large jumps can create enormous client timeline
            # offsets. Existing digit-named TS files give a bounded next ID
            # even when a legacy encoder zeroed its playlist on shutdown.
            self.start = max(existing) + 1 if existing else int(time.time())
            self.generation = 1 if existing else 0
        self.last = self.start - 1
        self.last_version = None
        self.last_cleanup = 0
        # Only this per-camera recorder writes this private file. The public
        # playlist and completed TS files remain readable throughout startup.
        self.incoming.unlink(missing_ok=True)

    def publish(self):
        try:
            stat = self.incoming.stat()
        except FileNotFoundError:
            return False
        version = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        if version == self.last_version:
            return False
        incoming, target = read_playlist(self.incoming)
        fresh = [replace(segment, discontinuity=self.generation) for segment in incoming if segment.sequence > self.last]
        if not fresh:
            self.last_version = version
            return False
        # temp_file publishes each complete TS before its incoming manifest.
        # Fail closed if that invariant is ever broken.
        if any(not (self.folder/segment.uri).is_file() for segment in fresh):
            return False
        history = self.segments
        if fresh[0].sequence != self.last + 1:
            history = []  # A suspended publisher missed the preceding window.
        # Do not resurrect minutes of stale video following an upstream outage.
        # Retained entries keep their original sequence, PDT and discontinuity.
        while history and (history[0].timestamp < fresh[-1].timestamp - self.stale_history_seconds
                           or not (self.folder/history[0].uri).is_file()):
            history = history[1:]
        combined = (history + fresh)[-self.window:]
        # Start the removal grace period when a URI leaves the *public*
        # playlist, including after a long outage; file creation time alone
        # would delete old-but-just-advertised segments too early.
        retained = {segment.uri for segment in combined}
        for old in self.segments:
            if old.uri not in retained:
                try:
                    os.utime(self.folder/old.uri, None)
                except FileNotFoundError:
                    pass
        # TARGETDURATION is immutable for a live playlist (RFC 8216 6.2.1).
        # A short final segment may lower FFmpeg's private value; it must not
        # lower the public value. Unexpected oversized segments fail closed.
        if any(math.floor(segment.duration + .5) > self.target for segment in combined):
            raise ValueError('Recorder segment exceeds the fixed HLS target duration')
        target = self.target
        first = combined[0]
        lines = ['#EXTM3U', '#EXT-X-VERSION:6', f'#EXT-X-TARGETDURATION:{target}',
                 f'#EXT-X-MEDIA-SEQUENCE:{first.sequence}',
                 f'#EXT-X-DISCONTINUITY-SEQUENCE:{first.discontinuity}', '#EXT-X-INDEPENDENT-SEGMENTS']
        previous = first.discontinuity
        for segment in combined:
            lines.extend(['#EXT-X-DISCONTINUITY'] * (segment.discontinuity - previous))
            lines.extend([f'#EXTINF:{segment.duration:.6f},', f'#EXT-X-PROGRAM-DATE-TIME:{segment.date}', segment.uri])
            previous = segment.discontinuity
        temporary = self.playlist.with_suffix('.m3u8.publishing')
        body = '\n'.join(lines) + '\n'
        # Keep an independent atomic checkpoint, including discontinuity
        # state. Write it first so a lost public file can recover without an
        # invented epoch or URI reuse. A valid public playlist stays primary.
        checkpoint_temp = self.checkpoint.with_suffix('.m3u8.publishing')
        with checkpoint_temp.open('w') as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        checkpoint_temp.replace(self.checkpoint)
        temporary.write_text(body)
        temporary.replace(self.playlist)
        self.segments, self.target = combined, target
        self.last, self.last_version = fresh[-1].sequence, version
        self.cleanup()
        return True

    def cleanup(self):
        # FFmpeg does not own retention because its private window omits the
        # previous generation. Keep removed TS files for > playlist + segment
        # duration (RFC 8216), including old generations and abandoned .tmps.
        now = time.time()
        if now - self.last_cleanup < 10:
            return
        keep = {segment.uri for segment in self.segments}
        for path in self.folder.iterdir():
            if re.fullmatch(r's\d+\.ts(?:\.tmp)?', path.name) and path.name not in keep:
                try:
                    if now - path.stat().st_mtime > 90:
                        path.unlink()
                except FileNotFoundError:
                    pass
        self.last_cleanup = now
