"""Local output freshness, separate from API/process liveness and vision state."""
import json
import math
from pathlib import Path
import re
import stat
import time


LIMITS={'playlist':12.,'capture':8.,'recorder':6.}
FUTURE_TOLERANCE_SECONDS=1.


def age_of(timestamp,now):
    if type(timestamp) not in (int,float) or not math.isfinite(timestamp) or timestamp<=0:
        return None
    age=now-timestamp
    # Atomic writers can publish just after this request's sampled clock.
    if not math.isfinite(age) or age < -FUTURE_TOLERANCE_SECONDS:return None
    return max(0.,age)


def freshness(path,kind,now):
    """Return numeric age + a canned reason; never return media/error contents."""
    try:
        info=path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size<=0:return None,kind+'_invalid'
        age=age_of(info.st_mtime,now)
        if age is None:return None,kind+'_invalid'
        if kind=='recorder':
            if info.st_size>65536:return None,'recorder_invalid'
            with path.open('rb') as handle:raw=handle.read(65537)
            if len(raw)>65536:return None,'recorder_invalid'
            heartbeat=json.loads(raw)
            stamp=age_of(heartbeat.get('timestamp'),now) if isinstance(heartbeat,dict) else None
            if stamp is None:return None,'recorder_invalid'
            # Rewriting stale heartbeat contents must not make the recorder fresh.
            age=max(age,stamp)
        return round(age,3),kind+'_stale' if age>=LIMITS[kind] else None
    except FileNotFoundError:
        return None,kind+'_missing'
    except (OSError,ValueError,TypeError):
        return None,kind+'_unavailable'


def camera_readiness(runtime,camera,now=None):
    now=time.time() if now is None else now
    if not isinstance(camera,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',camera):
        return {'video_ready':False,'playlist_age':None,'capture_age':None,'recorder_age':None,
                'video_reasons':['camera_invalid']}
    root=Path(runtime)
    paths={'playlist':root/'hls'/camera/'live.m3u8','capture':root/'capture'/(camera+'.jpg'),
           'recorder':root/'recorder'/(camera+'.json')}
    reasons=[];ages={}
    for kind,path in paths.items():
        age,reason=freshness(path,kind,now)
        ages[kind+'_age']=age
        if reason:reasons.append(reason)
    return {'video_ready':not reasons,**ages,'video_reasons':reasons}


def readiness_status(config,now=None):
    now=time.time() if now is None else now
    cameras=[]
    for camera in config.get('cameras',[]):
        state=camera_readiness(config['runtime'],camera['id'],now)
        cameras.append({'id':camera['id'],'ready':state['video_ready'],
            'reasons':state['video_reasons'],**{name:state[name] for name in ['playlist_age','capture_age','recorder_age']}})
    return {'ready':bool(cameras) and all(c['ready'] for c in cameras),'cameras':cameras,
            'thresholds_seconds':dict(LIMITS),'reasons':[] if cameras else ['no_configured_cameras']}
