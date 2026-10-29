"""Output readiness uses controlled files and clocks, never camera connections."""
import json
import os

import pytest

from backend.readiness import LIMITS, camera_readiness, readiness_status


NOW=1800000000.


def ready_files(root,camera='example',now=NOW,ages=None):
    ages=ages or {}
    paths={'playlist':root/'hls'/camera/'live.m3u8','capture':root/'capture'/(camera+'.jpg'),
           'recorder':root/'recorder'/(camera+'.json')}
    for kind,path in paths.items():
        path.parent.mkdir(parents=True,exist_ok=True)
        age=ages.get(kind,1.)
        path.write_bytes(json.dumps({'timestamp':now-age,'errors':['PRIVATE source address or credential']}).encode() if kind=='recorder' else b'fictional output')
        os.utime(path,(now-age,now-age))
    return paths


def test_all_configured_cameras_must_have_fresh_outputs(tmp_path):
    ready_files(tmp_path,'first');ready_files(tmp_path,'second')
    config={'runtime':str(tmp_path),'cameras':[{'id':'first'},{'id':'second'}]}
    result=readiness_status(config,NOW)
    assert result['ready'] is True
    assert all(c['ready'] and c['reasons']==[] for c in result['cameras'])
    assert 'PRIVATE' not in json.dumps(result) and str(tmp_path) not in json.dumps(result)
    os.utime(tmp_path/'capture/second.jpg',(NOW-10,NOW-10))
    failed=readiness_status(config,NOW)
    assert failed['ready'] is False
    assert failed['cameras'][0]['ready'] is True
    assert failed['cameras'][1]['reasons']==['capture_stale']


@pytest.mark.parametrize('kind',list(LIMITS))
def test_exact_freshness_boundary_fails_closed(tmp_path,kind):
    ready_files(tmp_path,ages={kind:LIMITS[kind]})
    state=camera_readiness(tmp_path,'example',NOW)
    assert state['video_ready'] is False
    assert state['video_reasons']==[kind+'_stale']
    assert state[kind+'_age']==LIMITS[kind]


def test_missing_and_invalid_future_files_do_not_look_fresh(tmp_path):
    state=camera_readiness(tmp_path,'example',NOW)
    assert state['video_reasons']==['playlist_missing','capture_missing','recorder_missing']
    assert all(state[k+'_age'] is None for k in LIMITS)
    paths=ready_files(tmp_path)
    paths['playlist'].write_bytes(b'')
    os.utime(paths['capture'],(NOW+20,NOW+20))
    paths['recorder'].write_text('{"timestamp": NaN}')
    os.utime(paths['recorder'],(NOW,NOW))
    state=camera_readiness(tmp_path,'example',NOW)
    assert state['video_reasons']==['playlist_invalid','capture_invalid','recorder_invalid']
    assert state['video_ready'] is False


def test_fresh_rewrite_does_not_revive_old_heartbeat_and_empty_configuration_is_not_ready(tmp_path):
    paths=ready_files(tmp_path,ages={'recorder':40})
    os.utime(paths['recorder'],(NOW,NOW))
    state=camera_readiness(tmp_path,'example',NOW)
    assert state['recorder_age']==40 and state['video_reasons']==['recorder_stale']
    assert readiness_status({'runtime':str(tmp_path),'cameras':[]},NOW)['ready'] is False
    assert camera_readiness(tmp_path,'../elsewhere',NOW)['video_reasons']==['camera_invalid']


@pytest.mark.parametrize('payload',['broken JSON','[]','{"timestamp": true}','{"timestamp": "1800000000"}','{"timestamp": 1e309}'])
def test_bad_heartbeat_is_reported_without_leaking_contents(tmp_path,payload):
    paths=ready_files(tmp_path)
    paths['recorder'].write_text(payload)
    os.utime(paths['recorder'],(NOW,NOW))
    state=camera_readiness(tmp_path,'example',NOW)
    assert state['video_ready'] is False and state['recorder_age'] is None
    assert state['video_reasons'] in (['recorder_invalid'],['recorder_unavailable'])
