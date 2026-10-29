"""Small, diverse annotation queue; zero detections never excludes a frame.

Reuse frames and predictions already decoded by the curator. Whole-frame and
tile hashes suppress repeats without depending on the detector's boxes.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from .store import database


def phash(gray):
    values = cv2.dct(cv2.resize(gray, (32, 32)).astype(np.float32))[:8, :8].flatten()[1:]
    bits = values > np.median(values)
    return f'{sum(int(v) << i for i, v in enumerate(bits)):016x}'


def fingerprint(image):
    if image is None or image.ndim != 3 or min(image.shape[:2]) < 12:
        raise ValueError('Learning frames must be color images at least 12 pixels wide and tall.')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    tiles = [phash(gray[y*h//3:(y+1)*h//3, x*w//4:(x+1)*w//4]) for y in range(3) for x in range(4)]
    chroma = np.mean(np.max(image, axis=2).astype(float)-np.min(image, axis=2))
    return {'global': phash(gray), 'tiles': tiles, 'lighting': 'ir' if chroma < 5 else 'day'}


def distance(a, b):
    return (int(a, 16)^int(b, 16)).bit_count()


def near_duplicate(a, b):
    if a['lighting'] != b['lighting']:
        return False
    tiles = [distance(x, y) for x, y in zip(a['tiles'], b['tiles'])]
    return distance(a['global'], b['global']) <= 6 and max(tiles, default=64) <= 8 and np.mean(tiles) <= 3


def retain(config, sample, image, context_path=None):
    """Atomically enforce the queue budget/window/dedup rules for either source."""
    root = Path(config['runtime']).resolve()
    key, signature = sample['id'], sample['signature']
    if context_path is not None and not context_path.is_file():
        return 'expired'
    with database(config) as db:
        # A backfill and the curator can offer frames at the same time.
        db.execute('BEGIN IMMEDIATE')
        if db.execute('SELECT 1 FROM learning_samples WHERE id=?', (key,)).fetchone():
            return 'known'
        if db.execute("SELECT count(*) FROM learning_samples WHERE status='pending'").fetchone()[0] >= config.get('learning_queue_limit', 180):
            return 'full'
        if db.execute('SELECT 1 FROM learning_samples WHERE camera=? AND group_id=?', (sample['camera'], sample['group'])).fetchone():
            return 'same_window'
        previous = db.execute('SELECT signature FROM learning_samples WHERE camera=? ORDER BY time DESC LIMIT 300', (sample['camera'],)).fetchall()
        if any(near_duplicate(signature, json.loads(p[0])) for p in previous):
            return 'duplicate'
        folder = root/'learning'/key
        folder.mkdir(parents=True, exist_ok=True)
        # Never write to the source corpus or its sidecars. Media is retained
        # before publishing a queue row, so readers cannot see partial files.
        if context_path is not None:
            temporary = folder/'context.tmp'
            shutil.copy2(context_path, temporary)
            temporary.replace(folder/'context.mp4')
        temporary = folder/'frame.tmp.jpg'
        if not cv2.imwrite(str(temporary), image, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise OSError('Could not save learning frame')
        temporary.replace(folder/'frame.jpg')
        db.execute('INSERT INTO learning_samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
            (key, sample['clip'], sample['camera'], sample['time'], sample['frame_seconds'],
             signature['lighting'], sample['reason'], json.dumps(sample['suggestions']), json.dumps(signature), sample['group'],
             'pending', datetime.now(timezone.utc).isoformat()))
    return 'added'


def offer(config, row, image, suggestions=None, frame_seconds=None):
    """Keep a representative with original video context, never modify source."""
    root = Path(config['runtime']).resolve()
    source = (root/row['path']).resolve()
    if not source.is_relative_to(root):
        return 'expired'
    key = hashlib.sha256(('learning:'+row['id']).encode()).hexdigest()[:24]
    timestamp = datetime.fromisoformat(row['time']).astimezone(timezone.utc)
    # All views in the same capture window stay together in training splits.
    group = f"capture:{timestamp.date()}:{(timestamp.hour*60+timestamp.minute)//10}"
    suggestions = suggestions or []
    audit = int(key[:8], 16) % 4 == 0
    reason = 'audit' if audit or not suggestions else 'uncertain' if any(s.get('confidence', 0) < .7 for s in suggestions) else 'diverse'
    return retain(config, {'id': key, 'clip': row['id'], 'camera': row['camera'], 'time': row['time'],
        'frame_seconds': frame_seconds if frame_seconds is not None else row['duration']*.5,
        'signature': fingerprint(image), 'group': group, 'reason': reason, 'suggestions': suggestions}, image, source)
