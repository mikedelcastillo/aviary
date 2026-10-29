"""Small, indexed evidence store. Deployment data never belongs in source control."""
import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def configuration():
    path = Path(os.environ.get('AVIARY_HOME_CONFIG', 'runtime/config.json'))
    return json.loads(path.read_text())


@contextmanager
def database(config):
    conn = sqlite3.connect(Path(config['runtime']) / 'aviary.sqlite', timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize(config):
    Path(config['runtime']).mkdir(parents=True, exist_ok=True)
    with database(config) as db:
        db.executescript('''
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS events(
          id TEXT PRIMARY KEY, time TEXT NOT NULL, camera TEXT NOT NULL,
          bird TEXT NOT NULL, confidence REAL, note TEXT NOT NULL DEFAULT '',
          photo TEXT NOT NULL DEFAULT '', source TEXT NOT NULL,
          activity TEXT NOT NULL DEFAULT '', bbox TEXT NOT NULL DEFAULT '[]');
        CREATE INDEX IF NOT EXISTS events_time ON events(time DESC,id DESC);
        CREATE INDEX IF NOT EXISTS events_bird_time ON events(bird,time DESC,id DESC);
        CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(id UNINDEXED,note,bird);
        CREATE TABLE IF NOT EXISTS imports(path TEXT PRIMARY KEY,signature TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY,expires REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS chat(id INTEGER PRIMARY KEY,session TEXT,role TEXT,
          text TEXT,time TEXT,evidence TEXT NOT NULL DEFAULT '[]');
        CREATE INDEX IF NOT EXISTS chat_session ON chat(session,id);
        CREATE TABLE IF NOT EXISTS zones(id TEXT PRIMARY KEY,camera TEXT,name TEXT,
          kind TEXT,rect TEXT,enabled INTEGER NOT NULL DEFAULT 1);
        CREATE TABLE IF NOT EXISTS visits(id INTEGER PRIMARY KEY,time TEXT,camera TEXT,
          bird TEXT,zone TEXT,seconds REAL,motion REAL,track TEXT);
        CREATE INDEX IF NOT EXISTS visits_bird_time ON visits(bird,time);
        CREATE TABLE IF NOT EXISTS favorites(event TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS preferences(key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE IF NOT EXISTS clips(
          id TEXT PRIMARY KEY,time TEXT NOT NULL,camera TEXT NOT NULL,path TEXT NOT NULL,
          duration REAL NOT NULL,poster TEXT NOT NULL DEFAULT '',status TEXT NOT NULL DEFAULT 'recorded',
          caption TEXT NOT NULL DEFAULT '',behavior TEXT NOT NULL DEFAULT '',score REAL,
          analysis TEXT NOT NULL DEFAULT '{}',birds TEXT NOT NULL DEFAULT '[]',favorite INTEGER NOT NULL DEFAULT 0);
        CREATE INDEX IF NOT EXISTS clips_status_time ON clips(status,time DESC,id DESC);
        CREATE TABLE IF NOT EXISTS clip_views(clip TEXT,viewer TEXT,seen_at TEXT,
          watched_seconds REAL,PRIMARY KEY(clip,viewer));
        CREATE INDEX IF NOT EXISTS clip_views_clip ON clip_views(clip);
        CREATE TABLE IF NOT EXISTS recording_failures(path TEXT PRIMARY KEY,signature TEXT,error TEXT);
        CREATE TABLE IF NOT EXISTS learning_samples(
          id TEXT PRIMARY KEY,clip TEXT,camera TEXT,time TEXT,frame_seconds REAL,
          lighting TEXT,reason TEXT,suggestions TEXT,signature TEXT,group_id TEXT,
          status TEXT NOT NULL DEFAULT 'pending',created TEXT);
        CREATE INDEX IF NOT EXISTS learning_queue ON learning_samples(status,time DESC);
        CREATE TABLE IF NOT EXISTS learning_answers(
          id TEXT PRIMARY KEY,sample TEXT,reviewer TEXT,time TEXT,outcome TEXT,
          annotation TEXT,lighting TEXT,undone INTEGER NOT NULL DEFAULT 0);
        CREATE UNIQUE INDEX IF NOT EXISTS learning_active_answer ON learning_answers(sample) WHERE undone=0;
        CREATE INDEX IF NOT EXISTS learning_answer_time ON learning_answers(time);
        CREATE TABLE IF NOT EXISTS playback(id INTEGER PRIMARY KEY,time TEXT,camera TEXT,event TEXT,detail TEXT);
        ''')
        columns = {r[1] for r in db.execute('PRAGMA table_info(zones)')}
        if 'anchor' not in columns:
            db.execute("ALTER TABLE zones ADD COLUMN anchor TEXT NOT NULL DEFAULT ''")


def utc_time(value, timezone):
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(timezone))
    return dt.astimezone(ZoneInfo('UTC')).isoformat(timespec='seconds')


def add_event(db, row):
    old = db.execute('SELECT note FROM events WHERE id=?', (row['id'],)).fetchone()
    if old and old['note'] == row['note']:
        return False
    db.execute('INSERT OR REPLACE INTO events VALUES(:id,:time,:camera,:bird,:confidence,:note,:photo,:source,:activity,:bbox)', row)
    if old:
        db.execute('DELETE FROM search WHERE id=?', (row['id'],))
    db.execute('INSERT INTO search(id,note,bird) VALUES(?,?,?)', (row['id'], row['note'], row['bird']))
    return True


def import_journal(config, path):
    path = Path(path)
    signature = hashlib.sha256(path.read_bytes()).hexdigest()
    count = 0
    with database(config) as db:
        old = db.execute('SELECT signature FROM imports WHERE path=?', (str(path),)).fetchone()
        if old and old['signature'] == signature:
            return 0
        for line in path.read_text().splitlines():
            try:
                entry = json.loads(line)
                when = utc_time(entry['time'], config['timezone'])
            except (ValueError, KeyError):
                continue
            observations = entry.get('observations') or [{
                'birds': entry.get('birds', []), 'note': entry.get('note', ''),
                'photo': next(iter(entry.get('photos', [])), ''), 'camera': 'Archive'}]
            for index, obs in enumerate(observations):
                labels = sorted({str(b).lower().strip() for b in obs.get('birds', [])})
                for bird in labels:
                    detection = next((d for d in obs.get('detections', []) if str(d.get('label', '')).lower() == bird), {})
                    # Identifiers come only from structured detector fields, never generated prose.
                    key = hashlib.sha256(f'{when}:{index}:{bird}:{obs.get("photo", "")}'.encode()).hexdigest()[:24]
                    row = dict(id=key, time=when, camera=str(obs.get('camera', 'Archive'))[:100],
                        bird=bird[:60], confidence=detection.get('confidence'), note=str(obs.get('note', ''))[:4000],
                        photo=str(obs.get('photo', '')), source='archive',
                        activity=str(detection.get('activity', ''))[:80], bbox=json.dumps(detection.get('bbox', [])))
                    count += add_event(db, row)
        db.execute('INSERT OR REPLACE INTO imports VALUES(?,?)', (str(path), signature))
    return count


if __name__ == '__main__':
    import sys
    config = configuration()
    initialize(config)
    print(json.dumps({'imported': sum(import_journal(config, p) for p in sorted(Path(sys.argv[1]).glob('*.jsonl')))}))
