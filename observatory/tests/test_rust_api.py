"""HTTP contract and privacy checks against the compiled native API.

All data is fictional. Run on the development host with AVIARY_RUST_BIN set.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import time
import threading
import httpx
import pytest
from websockets.sync.server import serve
from websockets.sync.client import connect
from websockets.exceptions import InvalidStatus, ConnectionClosed
from backend.store import initialize


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    binary=os.environ.get('AVIARY_RUST_BIN')
    if not binary or not Path(binary).is_file(): pytest.skip('Set AVIARY_RUST_BIN to the compiled development binary')
    root=tmp_path_factory.mktemp('native'); web=root/'web';web.mkdir()
    (web/'index.html').write_text('<html><script>window.example=1</script></html>')
    salt=bytes.fromhex('22'*16)
    config={'runtime':str(root),'timezone':'Asia/Tokyo','pin_salt':salt.hex(),
        'pin_hash':hashlib.pbkdf2_hmac('sha256',b'12345678',salt,600000).hex(),
        'cameras':[{'id':'example','name':'Example camera'}],'birds':[]}
    def echo(socket):
        try:
            for value in socket:
                socket.send(value)
                socket.send(b'fictional-video-packet')
        except ConnectionClosed:
            pass
    relay=serve(echo,'127.0.0.1',0)
    threading.Thread(target=relay.serve_forever,daemon=True).start()
    config['relay_ws']=f'ws://127.0.0.1:{relay.socket.getsockname()[1]}'
    initialize(config);path=root/'config.json';path.write_text(json.dumps(config))
    db=sqlite3.connect(root/'aviary.sqlite')
    db.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)",('one','2026-01-01T15:30:00+00:00','example','fern',.9,'Fictional record','','archive','','[]'))
    db.execute("INSERT INTO search VALUES(?,?,?)",('one','Fictional record','fern'))
    (root/'clip.mp4').write_bytes(bytes(range(256))*4)
    db.execute("INSERT INTO clips(id,time,camera,path,duration) VALUES(?,?,?,?,?)",('sample','2026-01-01T15:30:00+00:00','example','clip.mp4',20))
    db.commit();db.close()
    with socket.socket() as sock: sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    process=subprocess.Popen([binary],env={**os.environ,'AVIARY_HOME_CONFIG':str(path),'AVIARY_WEB_ROOT':str(web),'AVIARY_LISTEN':f'127.0.0.1:{port}','AVIARY_ADAPTER_URL':'http://127.0.0.1:1'},stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    url=f'http://127.0.0.1:{port}'
    try:
        for _ in range(50):
            try:
                if httpx.get(url+'/api/health').status_code==200:break
            except httpx.HTTPError:time.sleep(.05)
        else:raise RuntimeError('Native API failed to start')
        yield url,root
    finally:
        relay.shutdown()
        process.terminate()
        try:process.wait(5)
        except subprocess.TimeoutExpired:process.kill();process.wait()


@pytest.fixture
def client(native):
    with httpx.Client(base_url=native[0]) as client:yield client


def unlock(client):
    response=client.post('/api/login',json={'pin':'12345678'})
    assert response.status_code==200,response.text
    assert 'HttpOnly' in response.headers['set-cookie'] and 'SameSite=Strict' in response.headers['set-cookie']


def test_native_auth_origin_and_script_policy(client):
    assert client.get('/api/discovery').json()=={'service':'aviary-observatory','api':1}
    assert client.get('/api/events').status_code==401
    assert client.get('/api/clips/sample/video').status_code==401
    assert client.post('/api/login',json={'pin':'12345678'},headers={'Origin':'https://elsewhere.invalid'}).status_code==403
    unlock(client)
    assert client.get('/api/session').json()['authenticated']
    response=client.get('/')
    assert "'sha256-" in response.headers['content-security-policy']
    assert "script-src 'self' 'unsafe-inline'" not in response.headers['content-security-policy']
    assert client.get('/api/events').headers['cache-control']=='no-store'
    client.post('/api/logout')
    assert client.get('/api/events').status_code==401


def test_native_archive_query_timezone_pagination_and_fts(client):
    unlock(client)
    response=client.get('/api/events',params={'date':'2026-01-02','bird':'fern'})
    assert [e['id'] for e in response.json()['events']]==['one']
    assert 'photo' not in response.json()['events'][0]
    assert client.get('/api/events',params={'date':'2026-01-01'}).json()['events']==[]
    assert client.get('/api/events',params={'before':response.json()['next']}).json()['events']==[]
    assert client.get('/api/events',params={'before':'broken'}).status_code==422
    assert client.get('/api/events',params={'search':'" OR * NEAR('}).status_code==200
    assert len(client.get('/api/events',params={'search':'Fictional'}).json()['events'])==1
    assert client.post('/api/events/one/favorite').json()['favorite'] is True
    assert len(client.get('/api/events?favorites=true').json()['events'])==1
    assert client.post('/api/events/one/favorite').json()['favorite'] is False


def test_native_video_range_and_path_boundaries(client,native):
    unlock(client)
    response=client.get('/api/clips/sample/video',headers={'Range':'bytes=100-149'})
    assert response.status_code==206
    assert response.headers['content-range']=='bytes 100-149/1024'
    assert response.content==bytes(range(100,150))
    assert client.get('/api/clips/sample/video',headers={'Range':'bytes=9999-10000'}).status_code==416
    assert client.get('/api/cameras/example/hls/%2e%2e%2fconfig.json').status_code!=200
    assert client.get('/config.json').status_code==404
    root=native[1]
    with sqlite3.connect(root/'aviary.sqlite') as db:
        db.execute("INSERT INTO clips(id,time,camera,path,duration) VALUES('escape','x','example','../outside.mp4',20)")
    (root.parent/'outside.mp4').write_bytes(b'outside allowed runtime')
    assert client.get('/api/clips/escape/video').status_code==404


def test_first_view_is_recorded_once_and_preserved(client,native):
    unlock(client)
    assert client.post('/api/clips/sample/seen',json={'watched_seconds':.1}).status_code==422
    assert client.post('/api/clips/sample/seen',json={'watched_seconds':3}).status_code==200
    with sqlite3.connect(native[1]/'aviary.sqlite') as db:first=db.execute('SELECT seen_at FROM clip_views').fetchone()[0]
    assert client.post('/api/clips/sample/seen',json={'watched_seconds':12}).status_code==200
    with sqlite3.connect(native[1]/'aviary.sqlite') as db:
        records=db.execute('SELECT seen_at,watched_seconds FROM clip_views').fetchall()
    assert records==[(first,12)]


def test_websocket_auth_playback_and_message_filter(client,native):
    url=native[0].replace('http://','ws://')+'/api/cameras/example/stream'
    with pytest.raises(InvalidStatus) as rejected:
        connect(url)
    assert rejected.value.response.status_code==401
    unlock(client)
    headers={'Cookie':'aviary_session='+client.cookies.get('aviary_session')}
    with pytest.raises(InvalidStatus) as rejected:
        connect(url,additional_headers=headers,origin='https://elsewhere.invalid')
    assert rejected.value.response.status_code==403
    with connect(url,additional_headers=headers) as socket:
        socket.send('{"type":"mse","value":"avc1.640029"}')
        assert json.loads(socket.recv(timeout=3))['type']=='mse'
        assert socket.recv(timeout=3)==b'fictional-video-packet'
        socket.send('{"type":"webrtc/offer","value":"blocked"}')
        with pytest.raises(ConnectionClosed):socket.recv(timeout=3)
