"""Owner-reviewed annotations with reversible answers and compatible exports."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import secrets
import tempfile
from typing import Literal
from zoneinfo import ZoneInfo
import zipfile

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .store import database


class Box(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    cx: float = Field(ge=0, le=1)
    cy: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)
    label: str | None = Field(default=None, max_length=80)


class Answer(BaseModel):
    outcome: Literal['annotated', 'empty', 'uncertain', 'skip']
    lighting: Literal['day', 'ir']
    complete: bool = False
    boxes: list[Box] = Field(default_factory=list, max_length=40)


def labels_for(config, lighting, category=None):
    labels = config.get('annotation_labels') or [
        *[{'name': b['id'], 'kind': 'individual', 'models': ['live']} for b in config['birds']],
        {'name': 'unknown_bird', 'kind': 'unknown', 'models': ['live']}]
    return [item for item in labels if item.get('kind') == 'unknown' or
            (lighting == 'ir' and item.get('kind') == 'species') or
            (lighting == 'day' and item.get('kind') == 'individual' and
             ('archive' if category == 'phone' else 'live') in item.get('models', []))]


def validate_answer(config, answer, category=None):
    if answer.outcome == 'empty' and (answer.boxes or not answer.complete):
        raise HTTPException(422, 'Confirm that the full frame contains no birds.')
    if answer.outcome == 'annotated' and (not answer.complete or not answer.boxes):
        raise HTTPException(422, 'Account for every visible bird before completing this frame.')
    names = {item['name'] for item in labels_for(config, answer.lighting, category)}
    if len({box.id for box in answer.boxes}) != len(answer.boxes):
        raise HTTPException(422, 'Box IDs must be unique.')
    for box in answer.boxes:
        if not all(math.isfinite(v) for v in [box.cx, box.cy, box.w, box.h]) or \
           box.cx-box.w/2 < -.00001 or box.cy-box.h/2 < -.00001 or \
           box.cx+box.w/2 > 1.00001 or box.cy+box.h/2 > 1.00001:
            raise HTTPException(422, 'Keep each bird box within the frame.')
        if answer.outcome == 'annotated' and box.label not in names:
            raise HTTPException(422, 'Choose a valid label for each bird. Infrared frames use species labels.')


def select_diverse(rows, limit=5):
    """Reserve an audit slot, then balance camera and sampling reason."""
    selected, groups, cameras, reasons = [], set(), {}, {}
    remaining = list(rows)
    while remaining and len(selected) < limit:
        eligible = [r for r in remaining if r['group_id'] not in groups]
        if not eligible:
            break
        def rank(row):
            audit_first = 0 if selected or row['reason'] == 'audit' else 1
            return (audit_first, cameras.get(row['camera'], 0), reasons.get(row['reason'], 0))
        chosen = min(eligible, key=rank)
        selected.append(chosen)
        groups.add(chosen['group_id'])
        cameras[chosen['camera']] = cameras.get(chosen['camera'], 0)+1
        reasons[chosen['reason']] = reasons.get(chosen['reason'], 0)+1
        remaining.remove(chosen)
    return selected


def create_router(config):
    router = APIRouter(prefix='/api/learn')
    root = Path(config['runtime']).resolve()

    def sample_path(key, filename):
        if len(key) != 24 or any(c not in '0123456789abcdef' for c in key):
            raise HTTPException(404, 'Sample not found')
        path = (root/'learning'/key/filename).resolve()
        if not path.is_relative_to(root/'learning') or not path.is_file():
            raise HTTPException(404, 'Sample not found')
        return path

    @router.get('')
    def queue(more: bool = False):
        day = datetime.now(ZoneInfo(config['timezone'])).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc).isoformat()
        with database(config) as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM learning_samples WHERE status='pending' ORDER BY time DESC LIMIT 300")]
            today = db.execute("SELECT count(*) FROM learning_answers WHERE undone=0 AND outcome!='skip' AND time>=?", (day,)).fetchone()[0]
            total = db.execute("SELECT count(*) FROM learning_answers WHERE undone=0 AND outcome IN ('annotated','empty')").fetchone()[0]
        cards = []
        for row in select_diverse(rows, 5 if more else max(0, 5-today)):
            source = json.loads(row['signature']).get('source', {})
            cards.append({key: row[key] for key in ['id', 'camera', 'time', 'frame_seconds', 'lighting', 'reason']} | {
                'image': f"/api/learn/{row['id']}/image",
                'video': f"/api/learn/{row['id']}/video" if (root/'learning'/row['id']/'context.mp4').is_file() else None,
                'category': source.get('category', 'tapo/'+row['lighting']),
                'imported': bool(source),
                'suggestions': json.loads(row['suggestions'])})
        return {'cards': cards, 'today': today, 'goal': 5, 'confirmed_frames': total,
                'pending': len(rows), 'labels': {**{mode: labels_for(config, mode) for mode in ['day', 'ir']},
                                               'phone': labels_for(config, 'day', 'phone')}}

    @router.get('/export')
    def export():
        roster = config.get('annotation_labels', [])
        if not roster or any(type(r.get('index')) is not int or r['index'] < 0 for r in roster):
            raise HTTPException(409, 'Configure the original annotation roster before exporting global YOLO class indices.')
        indices = {r['name']: r['index'] for r in roster}
        if len(indices) != len(roster) or len(set(indices.values())) != len(indices):
            raise HTTPException(409, 'Roster names and indices must be unique.')
        with database(config) as db:
            rows = db.execute("SELECT a.*,s.camera,s.clip,s.time source_time,s.group_id,s.frame_seconds,s.signature FROM learning_answers a JOIN learning_samples s ON s.id=a.sample WHERE a.undone=0 AND a.outcome IN ('annotated','empty') ORDER BY a.time DESC").fetchall()
        folder = root/'exports'; folder.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder, suffix='.zip', delete=False) as handle:
            path = Path(handle.name)
        manifest, exported = [], set()
        try:
            with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
                for row in rows:
                    annotation = json.loads(row['annotation'])
                    if any(b['label'] not in indices for b in annotation['boxes']):
                        raise HTTPException(409, 'An annotation label no longer matches the roster.')
                    source_time = datetime.fromisoformat(row['source_time']).astimezone(timezone.utc)
                    bucket = (source_time.hour*60+source_time.minute)//10
                    prefix = f"tapo/{row['lighting']}/learn_{source_time:%Y%m%d}_{bucket:03d}_{row['sample']}"
                    source = json.loads(row['signature']).get('source', {})
                    if source:
                        # Keep old corpus names so its existing split grouping survives.
                        name = source.get('name', '')
                        category = 'phone' if source.get('category') == 'phone' else 'tapo/'+row['lighting']
                        if not re.fullmatch(r'[A-Za-z0-9._-]+\.(?:jpe?g|png)', name, re.I) or '..' in name:
                            raise HTTPException(409, 'An imported source filename is invalid.')
                        prefix = category+'/'+Path(name).stem
                    if prefix in exported:
                        raise HTTPException(409, 'Two imported frames have the same export filename. Export them separately after resolving their source names.')
                    exported.add(prefix)
                    provenance = {'source_group': row['group_id'], 'clip_id': row['clip'],
                        'source_time': row['source_time'], 'camera': row['camera'],
                        'frame_seconds': row['frame_seconds'], 'reviewed_at': row['time'], 'answer_id': row['id']}
                    if source:
                        provenance['import'] = source
                    archive.write(sample_path(row['sample'], 'frame.jpg'), prefix+'.jpg')
                    archive.writestr(prefix+'.json', json.dumps({'v': 1, **annotation, 'provenance': provenance}, indent=2))
                    archive.writestr(prefix+'.txt', ''.join(f"{indices[b['label']]} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}\n" for b in annotation['boxes']))
                    manifest.append({'image': prefix+'.jpg', **provenance})
                archive.writestr('manifest.json', json.dumps({'version': 1, 'roster': roster, 'frames': manifest}, indent=2))
                archive.writestr('README.txt', 'Owner-reviewed frames only. Empty TXT means a reviewed empty frame.\nKeep source_group together across train/validation/test. The detector proposals are not labels.\nImport into a separate annotation data root; do not overwrite original images.\n')
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return FileResponse(path, media_type='application/zip', filename='aviary-reviewed-frames.zip', background=BackgroundTask(path.unlink, missing_ok=True))

    @router.get('/{key}/image')
    def image(key: str):
        return FileResponse(sample_path(key, 'frame.jpg'), media_type='image/jpeg')

    @router.get('/{key}/video')
    def video(key: str):
        return FileResponse(sample_path(key, 'context.mp4'), media_type='video/mp4')

    @router.post('/{key}/answer')
    def answer(key: str, body: Answer, request: Request):
        sample_path(key, 'frame.jpg')
        annotation = {'boxed': body.complete and body.outcome in ('annotated', 'empty'),
                      'boxes': [box.model_dump() for box in body.boxes]}
        answer_id = secrets.token_hex(12)
        with database(config) as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT status,signature FROM learning_samples WHERE id=?', (key,)).fetchone()
            if not row or row['status'] != 'pending':
                raise HTTPException(409, 'This frame was already reviewed. Refresh the queue.')
            validate_answer(config, body, json.loads(row['signature']).get('source', {}).get('category'))
            db.execute('INSERT INTO learning_answers VALUES(?,?,?,?,?,?,?,0)', (answer_id, key,
                request.state.session, datetime.now(timezone.utc).isoformat(), body.outcome,
                json.dumps(annotation), body.lighting))
            db.execute('UPDATE learning_samples SET status=? WHERE id=?', ('deferred' if body.outcome == 'skip' else 'reviewed', key))
        return {'ok': True, 'answer_id': answer_id}

    @router.post('/answers/{answer_id}/undo')
    def undo(answer_id: str, request: Request):
        with database(config) as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM learning_answers WHERE id=? AND reviewer=? AND undone=0', (answer_id, request.state.session)).fetchone()
            if not row:
                raise HTTPException(409, 'This answer is no longer available to undo.')
            db.execute('UPDATE learning_answers SET undone=1 WHERE id=?', (answer_id,))
            db.execute("UPDATE learning_samples SET status='pending' WHERE id=?", (row['sample'],))
        return {'ok': True}

    return router
