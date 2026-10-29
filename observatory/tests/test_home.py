"""Behavior and privacy boundaries using entirely fictional records."""
import hashlib
import importlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp('home')
    salt = bytes.fromhex('11'*16)
    config = {'runtime': str(root), 'timezone': 'Asia/Tokyo', 'pin_salt': salt.hex(),
        'pin_hash': hashlib.pbkdf2_hmac('sha256', b'12345678', salt, 600000).hex(),
        'birds': [{'id':'fern','name':'Fern','species':'Example bird'}],
        'cameras':[{'id':'example','name':'Example view'}], 'photo_sources':[],
        'relay_ws':'ws://127.0.0.1:1','relay_http':'http://127.0.0.1:1',
        'ollama_url':'http://127.0.0.1:1','ollama_model':'example'}
    path = root/'config.json'; path.write_text(json.dumps(config))
    os.environ['AVIARY_HOME_CONFIG'] = str(path)
    # Imported by other endpoint suites too; each module fixture gets its own
    # fictional configuration, even if backend.app was already imported.
    module = importlib.reload(importlib.import_module('backend.app'))
    from backend.store import import_journal
    journal = root/'example.jsonl'
    journal.write_text(json.dumps({'time':'2026-01-02T00:30:00','birds':['fern'],
        'observations':[{'camera':'Example view','birds':['fern'],'note':'A fictional caption.',
        'photo':'data/server/memories/images/example.jpg','detections':[{'label':'fern','confidence':.9,'bbox':[1,2,30,40]}]}]})+'\n'+
        json.dumps({'time':'2026-01-02T01:30:00','birds':['species-only'],'observations':[{
            'birds':['species-only'],'note':'Fern is drinking (unverified caption).','photo':'','detections':[]}]})+'\n')
    assert import_journal(config,journal)==2
    assert import_journal(config,journal)==0
    return module,config,root


@pytest.fixture
def client(setup):
    with TestClient(setup[0].app,base_url='https://testserver') as c:
        yield c


def unlock(client):
    response=client.post('/api/login',json={'pin':'12345678'})
    assert response.status_code==200
    assert 'Secure' in response.headers['set-cookie']
    assert 'HttpOnly' in response.headers['set-cookie']


def test_private_api_and_photos_require_session(client):
    for route in ['/api/events','/api/overview','/api/readiness','/media/unknown','/api/cameras/example/snapshot']:
        assert client.get(route).status_code==401


def test_player_engine_telemetry_matches_browser_contract(client):
    unlock(client)
    route='/api/cameras/example/playback'
    assert client.post(route,json={'event':'engine','detail':'hls.js'}).status_code==200
    assert client.post(route,json={'event':'engine','detail':'native-hls'}).status_code==200
    assert client.post(route,json={'event':'unknown','detail':''}).status_code==422
    assert client.post(route,json={'event':'error','detail':'x'*601}).status_code==422


def test_readiness_is_authenticated_and_distinct_from_liveness_and_vision(client,setup,monkeypatch,tmp_path):
    from backend import readiness as readiness_module
    from backend.store import initialize
    from test_readiness import NOW,ready_files
    module,_,_=setup
    monkeypatch.setattr(module,'runtime',tmp_path)
    monkeypatch.setitem(module.config,'runtime',str(tmp_path))
    monkeypatch.setattr(readiness_module.time,'time',lambda:NOW)
    initialize(module.config)
    assert client.get('/api/readiness').status_code==401
    unlock(client)
    missing=client.get('/api/readiness')
    assert missing.status_code==503 and missing.json()['ready'] is False
    assert missing.headers['cache-control']=='no-store'
    assert client.get('/api/health').status_code==200
    ready_files(tmp_path)
    result=client.get('/api/readiness')
    assert result.status_code==200 and result.json()['ready'] is True
    state=module.camera_state('example')
    assert state['video_ready'] is True and state['online'] is False
    frames=tmp_path/'frames';frames.mkdir()
    (frames/'example.json').write_text(json.dumps({'timestamp':NOW-1,'privacy':'clear','detections':[]}))
    os.utime(tmp_path/'hls/example/live.m3u8',(NOW-13,NOW-13))
    state=module.camera_state('example')
    assert state['online'] is True and state['video_ready'] is False
    assert client.get('/api/readiness').status_code==503
    assert client.get('/api/health').json()=={'ok':True}


def test_cross_origin_login_and_mutation_rejected(client):
    assert client.post('/api/login',json={'pin':'12345678'},headers={'Origin':'https://attacker.invalid'}).status_code==403
    unlock(client)
    assert client.post('/api/events/anything/favorite',headers={'Origin':'https://attacker.invalid'}).status_code==403


def test_archive_date_and_identity_are_not_invented_from_caption(client):
    unlock(client)
    data=client.get('/api/events',params={'bird':'fern','date':'2026-01-02'}).json()
    assert len(data['events'])==1
    assert data['events'][0]['time']=='2026-01-01T15:30:00+00:00'
    assert 'photo' not in data['events'][0]
    assert len(client.get('/api/events',params={'date':'2026-01-01'}).json()['events'])==0


def test_pagination_no_duplicates(client):
    unlock(client)
    first=client.get('/api/events',params={'limit':1}).json()
    second=client.get('/api/events',params={'limit':1,'before':first['next']}).json()
    assert first['events'][0]['id']!=second['events'][0]['id']
    assert client.get('/api/events',params={'before':'invalid'}).status_code==422


def test_search_escapes_fts_syntax(client):
    unlock(client)
    response=client.get('/api/events',params={'search':'" OR * NEAR('})
    assert response.status_code==200


def test_privacy_fails_closed_on_missing_and_stale_frames(client,setup):
    unlock(client)
    root=setup[2]; (root/'frames').mkdir(exist_ok=True)
    assert client.get('/api/cameras/example/snapshot').status_code==503
    (root/'frames/example.json').write_text(json.dumps({'timestamp':1,'privacy':'clear','detections':[]}))
    (root/'frames/example.jpg').write_bytes(b'not an actual frame')
    assert client.get('/api/cameras/example/snapshot').status_code==503
    (root/'frames/example.json').write_text(json.dumps({'timestamp':datetime.now().timestamp(),'privacy':'person','detections':[]}))
    assert client.get('/api/cameras/example/snapshot').status_code==503


def test_archive_path_validation(setup):
    module=setup[0]
    assert module.remote_photo_path('data/server/memories/images/example.jpg')=='memories/images/example.jpg'
    for path in ['../../etc/passwd','/etc/passwd','data/server/memories/images/x.jpg;echo hi','http://bad.invalid/x.jpg']:
        with pytest.raises(Exception): module.remote_photo_path(path)


def test_zone_rectangles_stay_inside_picture(client,setup):
    unlock(client)
    body={'camera':'example','name':'Example water','kind':'water','rect':[.9,.2,.4,.1]}
    assert client.post('/api/zones',json=body).status_code==422
    body['rect']=[.1,.2,.2,.1]
    assert client.post('/api/zones',json=body).status_code==409
    root=setup[2]
    from PIL import Image
    captured=datetime.now().timestamp()
    Image.new('RGB',(32,18),'green').save(root/'frames/example.jpg')
    os.utime(root/'frames/example.jpg',(captured+.001,captured+.001))
    (root/'frames/example.json').write_text(json.dumps({'timestamp':captured,'privacy':'clear','detections':[]}))
    os.utime(root/'frames/example.json',(captured+.002,captured+.002))
    reference=client.post('/api/cameras/example/zone-reference')
    assert reference.status_code==200, reference.text
    body['reference_token']=reference.headers['x-aviary-zone-reference']
    assert client.post('/api/zones',json=body).status_code==200
    assert client.get('/api/zones').json()[0]['kind']=='water'


def test_memory_followup_and_feeding_unknown(client):
    unlock(client)
    first=client.post('/api/ask',json={'text':'When was Fern seen on 2026-01-02?'}).json()
    assert '1 recorded sightings' in first['text']
    second=client.post('/api/ask',json={'text':'Did she drink on 2026-01-02?'}).json()
    assert 'Fern' in second['text']
    assert 'cannot confirm' in second['text']
    assert len(client.get('/api/chat').json())==4


def test_favorite_and_logout(client):
    unlock(client)
    event=client.get('/api/events').json()['events'][0]['id']
    assert client.post('/api/events/'+event+'/favorite').json()['favorite']
    assert len(client.get('/api/events?favorites=true').json()['events'])==1
    assert not client.post('/api/events/'+event+'/favorite').json()['favorite']
    client.post('/api/logout')
    assert client.get('/api/events').status_code==401


def test_websocket_rejects_unauthenticated(client):
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect('/api/cameras/example/stream'): pass
