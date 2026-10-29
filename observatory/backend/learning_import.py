"""Offer a few full frames from a local COPY of the old annotation corpus.

Run on the app host with its ML environment, from observatory/:
python -m backend.learning_import --source /path/to/copied/raw --limit 15

Only runtime/learning and the app database are written. Images, JSON annotations,
and detector TXT sidecars in the source are never modified or treated as truth.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
from zoneinfo import ZoneInfo

import cv2

from .learning_sampling import fingerprint, retain
from .store import configuration, initialize


CAPTURE = re.compile(r'^(?P<camera>camera-\d+)_(?:day|ir)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{6})_')


def import_corpus(config, source, limit=15, seed=0, max_scan=500, include_reviewed=False):
    source = Path(source).resolve(strict=True)
    runtime = Path(config['runtime']).resolve()
    if not source.is_dir() or source.is_relative_to(runtime) or runtime.is_relative_to(source):
        raise ValueError('Use a separate local corpus copy, outside the app runtime.')
    if limit < 1 or max_scan < 1:
        raise ValueError('limit and max_scan must be positive.')
    files = []
    for category in ('tapo/day', 'tapo/ir', 'phone'):
        folder = source/category
        if folder.is_dir():
            files.extend((path, category) for path in sorted(folder.iterdir())
                         if path.suffix.lower() in {'.jpg', '.jpeg', '.png'} and path.is_file())
    if not files:
        raise ValueError('No images found in tapo/day, tapo/ir, or phone under --source.')
    random.Random(seed).shuffle(files)
    initialize(config)
    counts = Counter()
    for path, category in files[:max_scan]:
        if counts['added'] >= limit:
            break
        if not re.fullmatch(r'[A-Za-z0-9._-]+\.(?:jpe?g|png)', path.name, re.I) or '..' in path.name:
            counts['unsupported_filename'] += 1
            continue
        # A symlink in a copy must not silently lead back into remote originals.
        if not path.resolve().is_relative_to(source):
            counts['outside_source'] += 1
            continue
        sidecar = path.with_suffix('.json')
        if sidecar.exists() and not include_reviewed:
            if not sidecar.resolve().is_relative_to(source):
                counts['outside_source'] += 1
                continue
            try:
                annotation = json.loads(sidecar.read_text(encoding='utf-8'))
                if annotation.get('boxed') and all(b.get('label') for b in annotation.get('boxes', [])):
                    counts['already_reviewed'] += 1
                    continue
            except (ValueError, TypeError, AttributeError, OSError):
                counts['invalid_sidecar'] += 1
                continue
        image = cv2.imread(str(path))
        if image is None or min(image.shape[:2]) < 12:
            counts['unreadable'] += 1
            continue
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        key = hashlib.sha256(('corpus:'+content_hash).encode()).hexdigest()[:24]
        signature = fingerprint(image)
        signature['lighting'] = 'ir' if category == 'tapo/ir' else 'day'
        camera, group = 'imported-'+category.replace('/', '-'), 'corpus:'+content_hash
        timestamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        timestamp_basis = 'file_mtime_not_capture_time'
        match = CAPTURE.match(path.stem)
        if match:
            try:
                timestamp = datetime.strptime(match['date']+match['time'], '%Y-%m-%d%H%M%S').replace(tzinfo=ZoneInfo(config['timezone']))
            except ValueError:
                counts['invalid_capture_time'] += 1
                continue
            camera = match['camera']
            group = f"{camera}|{match['date']}|{(timestamp.hour*60+timestamp.minute)//10}"
            timestamp_basis = 'capture_filename'
        signature['source'] = {'category': category, 'name': path.name, 'sha256': content_hash,
                               'timestamp_basis': timestamp_basis}
        result = retain(config, {'id': key, 'clip': None, 'camera': camera,
            'time': timestamp.astimezone(timezone.utc).isoformat(), 'frame_seconds': 0,
            'group': group, 'signature': signature, 'reason': 'corpus', 'suggestions': []}, image)
        counts[result] += 1
        if result == 'full':
            break
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Local copy containing tapo/day, tapo/ir and/or phone')
    parser.add_argument('--limit', type=int, default=15)
    parser.add_argument('--max-scan', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--include-reviewed', action='store_true', help='Also offer already completed canonical JSON annotations for a fresh review')
    args = parser.parse_args()
    try:
        result = import_corpus(configuration(), args.source, args.limit, args.seed, args.max_scan, args.include_reviewed)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
