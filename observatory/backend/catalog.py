"""Index finished recordings independently of the local VLM's latency."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

from .store import configuration, database, initialize


def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
        'format=duration:stream=codec_type', '-of', 'json', str(path)],
        capture_output=True, timeout=10, check=True)
    data = json.loads(result.stdout)
    if not any(s.get('codec_type') == 'video' for s in data.get('streams', [])):
        raise ValueError('No video stream')
    return float(data['format']['duration'])


def scan(config):
    root = Path(config['runtime'])
    with database(config) as db:
        known = {r[0] for r in db.execute('SELECT id FROM clips')}
        failed = dict(db.execute('SELECT path,signature FROM recording_failures'))
    counts = {'indexed': 0, 'invalid': 0, 'probed': 0}
    for path in sorted((root/'recordings').glob('*/*.mp4')):
        relative = str(path.relative_to(root))
        key = hashlib.sha256(relative.encode()).hexdigest()[:24]
        if key in known:
            continue
        try:
            stat = path.stat()
            if time.time()-stat.st_mtime < 25 or stat.st_size < 10000:
                continue
            signature = f'{stat.st_size}:{stat.st_mtime_ns}'
            if failed.get(relative) == signature:
                continue
            counts['probed'] += 1
            try:
                duration = probe(path)
                if not 5 < duration < 90:
                    raise ValueError('Unsupported segment duration')
                when = datetime.strptime(path.stem, '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).isoformat(timespec='seconds')
            except (ValueError, KeyError, subprocess.SubprocessError) as error:
                # Incomplete files are retried only if their size or timestamp changes.
                with database(config) as db:
                    db.execute('INSERT OR REPLACE INTO recording_failures VALUES(?,?,?)',
                               (relative, signature, type(error).__name__))
                counts['invalid'] += 1
                continue
            current = path.stat()
            if (current.st_size, current.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                continue
            # Do not hold a database write transaction during an external probe.
            with database(config) as db:
                db.execute('INSERT OR IGNORE INTO clips(id,time,camera,path,duration) VALUES(?,?,?,?,?)',
                           (key, when, path.parent.name, relative, duration))
                db.execute('DELETE FROM recording_failures WHERE path=?', (relative,))
            counts['indexed'] += 1
        except OSError:
            continue  # Rolling retention may remove a segment during the scan.
    return counts


def run():
    config = configuration()
    initialize(config)
    root = Path(config['runtime'])
    while True:
        started = time.monotonic()
        try:
            state = {**scan(config), 'timestamp': time.time(),
                     'scan_seconds': round(time.monotonic()-started, 3)}
            temporary = root/'catalog.tmp'
            temporary.write_text(json.dumps(state))
            temporary.replace(root/'catalog.json')
            print(json.dumps(state), flush=True)
        except Exception as error:
            print(json.dumps({'error': type(error).__name__}), flush=True)
        time.sleep(10)


if __name__ == '__main__':
    run()
