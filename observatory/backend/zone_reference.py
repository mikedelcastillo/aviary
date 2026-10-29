"""Short-lived, bounded drafts of the exact screened image used to draw a zone.

Drafts stay in this API process. A restart deliberately requires a new picture.
Nothing here contacts or controls a camera.
"""
from dataclasses import dataclass
from io import BytesIO
import json
import math
import os
from pathlib import Path
import secrets
import threading
import time

from PIL import Image

MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_STATE_BYTES = 64 * 1024
FRAME_MAX_AGE = 10
REFERENCE_TTL = 300


class ReferenceError(ValueError):
    """A safe, user-facing reference failure."""


class FramePending(ReferenceError):
    """A new screened frame has not finished publication yet."""


def _fresh(stamp, now):
    return (isinstance(stamp, (int, float)) and not isinstance(stamp, bool)
            and math.isfinite(stamp) and -1 <= now - stamp < FRAME_MAX_AGE)


def _state(path, now):
    with path.open('rb') as source:
        metadata = os.fstat(source.fileno())
        raw = source.read(MAX_STATE_BYTES + 1)
    if len(raw) > MAX_STATE_BYTES:
        raise ValueError('state size')
    data = json.loads(raw)
    if (not isinstance(data, dict) or data.get('privacy') != 'clear'
            or not _fresh(data.get('timestamp'), now) or not _fresh(metadata.st_mtime, now)):
        raise ValueError('screening unavailable')
    return data, raw, metadata


def require_screened(runtime, camera, *, now=None):
    """Reject stale, missing, malformed or person-paused publications."""
    now = time.time() if now is None else now
    try:
        state, _, metadata = _state(Path(runtime) / 'frames' / (camera + '.json'), now)
        image = (Path(runtime) / 'frames' / (camera + '.jpg')).stat()
        if not 0 < image.st_size <= MAX_IMAGE_BYTES or not _fresh(image.st_mtime, now):
            raise ValueError('image unavailable')
        # The worker publishes screened JPEG first, then its status. Do not
        # accept the gap where a new image still has the preceding status.
        if image.st_mtime_ns > metadata.st_mtime_ns or image.st_mtime < state['timestamp']:
            raise FramePending('Waiting for a newly screened picture. Please refresh.')
        return state
    except FramePending:
        raise
    except (OSError, ValueError, TypeError):
        raise ReferenceError('A fresh screened view is required. Wait for the camera, then refresh the picture.') from None


def read_screened(runtime, camera, *, after=None, now=None):
    """Read a stable screened publication with bounded allocation; retain bytes."""
    now = time.time() if now is None else now
    folder = Path(runtime) / 'frames'
    try:
        state, raw, metadata = _state(folder / (camera + '.json'), now)
        if after is not None and state['timestamp'] <= after:
            raise FramePending('Waiting for a newly screened picture. Please refresh.')
        path = folder / (camera + '.jpg')
        with path.open('rb') as source:
            image_stat = os.fstat(source.fileno())
            if not 0 < image_stat.st_size <= MAX_IMAGE_BYTES or not _fresh(image_stat.st_mtime, now):
                raise ValueError('image unavailable')
            image = source.read(MAX_IMAGE_BYTES + 1)
        if len(image) > MAX_IMAGE_BYTES:
            raise ValueError('image size')
        _, latest_raw, latest_metadata = _state(folder / (camera + '.json'), now)
        latest_image = path.stat()
        if (raw != latest_raw or metadata.st_mtime_ns != latest_metadata.st_mtime_ns
                or image_stat.st_ino != latest_image.st_ino
                or image_stat.st_mtime_ns != latest_image.st_mtime_ns
                or image_stat.st_size != latest_image.st_size
                or image_stat.st_mtime < state['timestamp']
                or image_stat.st_mtime_ns > metadata.st_mtime_ns):
            raise FramePending('The picture changed while loading. Please refresh.')
        with Image.open(BytesIO(image)) as picture:
            if picture.format != 'JPEG' or not 0 < picture.width * picture.height <= MAX_IMAGE_PIXELS:
                raise ValueError('image format')
            picture.verify()
        return image
    except FramePending:
        raise
    except (OSError, ValueError, TypeError, Image.DecompressionBombError):
        raise ReferenceError('A fresh screened view is required. Wait for the camera, then refresh the picture.') from None


def stationary_pose(pose):
    if not isinstance(pose, dict) or pose.get('moving') is not False:
        raise ReferenceError('Wait for the camera to stop, then refresh the picture.')
    coordinates = [pose.get('pan'), pose.get('tilt')]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not -1 <= v <= 1 for v in coordinates):
        raise ReferenceError('Camera position is unavailable. Please refresh the picture.')
    return {'pan': coordinates[0], 'tilt': coordinates[1], 'moving': False}


def matching_pose(before, after):
    after = stationary_pose(after)
    if any(abs(before[key] - after[key]) > .002 for key in ('pan', 'tilt')):
        raise ReferenceError('The camera moved while loading. Please refresh the picture.')
    return before


@dataclass(frozen=True)
class Draft:
    owner: str
    camera: str
    expires: float
    image: bytes = b''
    anchor: str = ''


class ReferenceStore:
    """Reservations bound in-flight reads as well as retained image memory."""
    def __init__(self, *, max_drafts=16, per_owner=4, ttl=REFERENCE_TTL, clock=time.monotonic):
        self.max_drafts, self.per_owner, self.ttl, self.clock = max_drafts, per_owner, ttl, clock
        self._drafts = {}
        self._lock = threading.Lock()

    def _clean(self):
        now = self.clock()
        self._drafts = {key: value for key, value in self._drafts.items() if value.expires > now}

    def reserve(self, owner, camera):
        with self._lock:
            self._clean()
            # A refresh retires the preceding picture for this owner/camera.
            self._drafts = {key: value for key, value in self._drafts.items()
                            if not (value.owner == owner and value.camera == camera and value.image)}
            if (len(self._drafts) >= self.max_drafts
                    or sum(v.owner == owner for v in self._drafts.values()) >= self.per_owner):
                raise ReferenceError('Too many pictures are open. Wait five minutes, then refresh.')
            token = secrets.token_urlsafe(32)
            self._drafts[token] = Draft(owner, camera, self.clock() + self.ttl)
            return token

    def finish(self, token, image, anchor=''):
        if not 0 < len(image) <= MAX_IMAGE_BYTES:
            raise ReferenceError('The reference picture is too large. Please refresh.')
        with self._lock:
            self._clean()
            prior = self._drafts.get(token)
            if prior is None or prior.image:
                raise ReferenceError('This picture has expired. Please refresh and draw the zone again.')
            self._drafts[token] = Draft(prior.owner, prior.camera, self.clock() + self.ttl, bytes(image), anchor)

    def discard(self, token):
        with self._lock:
            self._drafts.pop(token, None)

    def claim(self, token, owner, camera):
        with self._lock:
            self._clean()
            draft = self._drafts.get(token)
            if not draft or not draft.image or draft.owner != owner or draft.camera != camera:
                raise ReferenceError('This picture has expired or belongs to another view. Please refresh and draw the zone again.')
            del self._drafts[token]
            return draft
