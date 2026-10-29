"""Bounded latest-frame readers, local bird tracking and fail-closed previews.

Capture never waits for inference. Every camera has one replaceable frame slot.
An occupied feeder/water rectangle is a visit, never proof of eating/drinking.
"""
import hashlib
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
import cv2
import numpy as np
import torch
import supervision as sv
from trackers import ByteTrackTracker
from ultralytics import YOLO

from .store import add_event, configuration, database, initialize
from .view_guard import ViewGuard


class Reader:
    def __init__(self, url):
        self.url = url
        self.latest = None
        self.lock = threading.Lock()
        self.reconnects = 0
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        delay = 1
        while True:
            # Decoder threads bounded independently of the machine's core count.
            proc = subprocess.Popen(['ffmpeg', '-nostdin', '-loglevel', 'error', '-rtsp_transport', 'tcp',
                '-timeout', '10000000', '-use_wallclock_as_timestamps', '1', '-threads', '2', '-i', self.url, '-an',
                '-vf', 'fps=2,scale=1280:720', '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            try:
                while True:
                    raw = proc.stdout.read(1280 * 720 * 3)
                    if len(raw) != 1280 * 720 * 3:
                        break
                    frame = np.frombuffer(raw, np.uint8).reshape(720, 1280, 3).copy()
                    with self.lock:
                        self.latest = (time.time(), frame)
                    delay = 1
            finally:
                proc.kill(); proc.wait(); proc.stdout.close()
            self.reconnects += 1
            time.sleep(delay)
            delay = min(delay * 2, 30)

    def get(self):
        with self.lock:
            return self.latest


class SharedFrameReader:
    """The recorder already decoded this camera; reuse its atomic latest JPEG."""
    def __init__(self, path):
        self.path=path
        self.latest=None
        self.reconnects=0
    def get(self):
        try:
            stamp=self.path.stat().st_mtime
            if self.latest is None or stamp != self.latest[0]:
                frame=cv2.imread(str(self.path))
                if frame is not None:self.latest=(stamp,frame)
        except OSError:pass
        return self.latest


def atomic(path, content):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes(content)
    temp.replace(path)


def run():
    config = configuration()
    initialize(config)
    runtime = Path(config['runtime'])
    frames, photos = runtime / 'frames', runtime / 'photos'
    frames.mkdir(parents=True, exist_ok=True); photos.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    detector = YOLO(config['bird_model'])
    person = YOLO(config['person_model'])
    readers = {c['id']: SharedFrameReader(runtime/'capture'/(c['id']+'.jpg')) if config.get('shared_capture',True)
               else Reader(config['relay_rtsp'] + '/' + c['id']) for c in config['cameras']}
    trackers = {c['id']: ByteTrackTracker(frame_rate=1, minimum_consecutive_frames=2) for c in config['cameras']}
    roster = {b['id'] for b in config['birds']}
    last_frame, last_event, last_position, visits = {}, {}, {}, {}
    backgrounds, zone_pause = {}, {}
    last_zone_read = 0
    zones = []
    view_guard = ViewGuard(runtime)
    while True:
        cycle = time.monotonic()
        if time.monotonic() - last_zone_read > 5:
            with database(config) as db:
                zones = [dict(r) for r in db.execute('SELECT * FROM zones WHERE enabled=1')]
            last_zone_read = time.monotonic()
        for camera, reader in readers.items():
            sample = reader.get()
            if sample is None or time.time() - sample[0] > 8 or sample[0] == last_frame.get(camera):
                continue
            stamp, image = sample
            last_frame[camera] = stamp
            started = time.monotonic()
            state = {'timestamp': stamp, 'privacy': 'unavailable', 'detections': [], 'reconnects': reader.reconnects}
            try:
                screened = person.predict(image, imgsz=640, conf=.2, classes=[0], device='cpu', verbose=False)[0]
                if len(screened.boxes):
                    state['privacy'] = 'person'
                    atomic(frames / f'{camera}.json', json.dumps(state).encode())
                    continue
                state['privacy'] = 'clear'
                result = detector.predict(image, imgsz=960, conf=.4, iou=.45, device='cpu', verbose=False)[0]
                detections = sv.Detections.from_ultralytics(result)
                tracked = trackers[camera].update(detections)
                small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (32, 18))
                if camera in backgrounds and np.median(cv2.absdiff(small, backgrounds[camera])) > 20:
                    zone_pause[camera] = time.time() + 30
                backgrounds[camera] = small
                state['zone_motion_pause'] = zone_pause.get(camera, 0) > time.time()
                state['zone_views'] = {z['id']: view_guard.check({**z, 'rect':json.loads(z['rect'])}, image, tracked.xyxy)
                    for z in zones if z['camera'] == camera}
                records = []
                # Tracking is temporal assistance; named identity still requires the bird classifier.
                for i, box in enumerate(tracked.xyxy):
                    confidence = float(tracked.confidence[i])
                    label = str(result.names[int(tracked.class_id[i])]).lower()
                    track = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                    bird = label if confidence >= .7 else 'uncertain bird'
                    coords = [round(float(v), 1) for v in box]
                    records.append({'bird': bird, 'confidence': round(confidence, 3), 'bbox': coords,
                                    'track': track, 'identified': bird in roster})
                    center = ((box[0]+box[2])/2560, (box[1]+box[3])/1440)
                    pos_key = (camera, track, bird)
                    old = last_position.get(pos_key)
                    motion = float(np.linalg.norm(np.array(center) - np.array(old[1]))) if old else 0
                    last_position[pos_key] = (stamp, center)
                    interval = min(2, max(0, stamp-old[0])) if old else 0
                    if bird in roster and track >= 0 and not state['zone_motion_pause']:
                        for z in zones:
                            if z['camera'] != camera:
                                continue
                            if not state['zone_views'].get(z['id'], {}).get('valid'):
                                continue
                            if z.get('anchor'):
                                try:
                                    pose=json.loads((runtime/'poses'/f'{camera}.json').read_text())
                                    anchor=json.loads(z['anchor'])
                                    if time.time()-pose['timestamp']>15 or pose.get('moving') or abs(pose['pan']-anchor['pan'])>.01 or abs(pose['tilt']-anchor['tilt'])>.01:
                                        continue
                                except (OSError,KeyError,ValueError):
                                    continue
                            x,y,w,h = json.loads(z['rect'])
                            if x <= center[0] <= x+w and y <= center[1] <= y+h:
                                key = (camera, bird, z['id'], track)
                                value = visits.setdefault(key, {'seconds': 0, 'motion': 0, 'start': stamp})
                                value['seconds'] += interval
                                value['motion'] += motion
                    if confidence >= .7 and stamp - last_event.get((camera, bird), 0) >= 60:
                        when = datetime.fromtimestamp(stamp, timezone.utc).isoformat(timespec='seconds')
                        key = hashlib.sha256(f'{camera}:{bird}:{when}'.encode()).hexdigest()[:24]
                        photo = key + '.jpg'
                        cv2.imwrite(str(photos / photo), image, [cv2.IMWRITE_JPEG_QUALITY, 76])
                        row = dict(id=key,time=when,camera=camera,bird=bird,confidence=confidence,
                            note='Recorded by the local bird detector. Identity is a model estimate.', photo='local:'+photo,
                            source='live',activity='',bbox=json.dumps(coords))
                        with database(config) as db:
                            add_event(db, row)
                        last_event[(camera, bird)] = stamp
                for key, value in list(visits.items()):
                    if stamp-value['start'] >= 10:
                        if value['seconds'] >= 2:
                            with database(config) as db:
                                db.execute('INSERT INTO visits(time,camera,bird,zone,seconds,motion,track) VALUES(?,?,?,?,?,?,?)',
                                    (datetime.fromtimestamp(value['start'], timezone.utc).isoformat(timespec='seconds'),*key[:3], value['seconds'],value['motion'],str(key[3])))
                        visits.pop(key)
                last_position = {k:v for k,v in last_position.items() if stamp-v[0] < 20}
                state['detections'] = records
                state['inference_ms'] = round((time.monotonic()-started)*1000, 1)
                # Only this screened frame is published. Raw originals have a separate, explicit UI.
                ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if ok:
                    atomic(frames / f'{camera}.jpg', encoded.tobytes())
                atomic(frames / f'{camera}.json', json.dumps(state).encode())
            except Exception as exc:
                state['privacy'] = 'unavailable'
                state['error'] = type(exc).__name__
                atomic(frames / f'{camera}.json', json.dumps(state).encode())
                print(json.dumps({'camera': camera, 'error': type(exc).__name__}), flush=True)
        time.sleep(max(.05, 1 - (time.monotonic()-cycle)))


if __name__ == '__main__':
    run()
