"""Frozen-picture save races and privacy checks. All camera calls are mocked."""
import asyncio
from contextlib import contextmanager
from io import BytesIO
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.zone_reference import (ReferenceStore, ReferenceError, FramePending,
    MAX_IMAGE_BYTES, read_screened, require_screened, stationary_pose, matching_pose)
from test_home import setup, unlock


def jpeg(color='green', size=(64, 36)):
    result = BytesIO()
    Image.new('RGB', size, color).save(result, format='JPEG')
    return result.getvalue()


def publish(root, camera='example', *, image=None, privacy='clear', timestamp=None, now=None):
    now = time.time() if now is None else now
    folder = root / 'frames'; folder.mkdir(exist_ok=True)
    stamp = now-.03 if timestamp is None else timestamp
    image_time = max(now-.02, stamp)
    picture = folder / (camera + '.jpg')
    picture.write_bytes(jpeg() if image is None else image)
    os.utime(picture, (image_time, image_time))
    state = folder / (camera + '.json')
    state.write_text(json.dumps({'privacy': privacy, 'timestamp': stamp}))
    state_time = max(now-.01, image_time)
    os.utime(state, (state_time, state_time))


def test_screened_read_is_bounded_and_rejects_incomplete_publication(tmp_path):
    publish(tmp_path, now=100)
    assert read_screened(tmp_path, 'example', now=100) == jpeg()
    with pytest.raises(FramePending):
        read_screened(tmp_path, 'example', after=100, now=100)
    os.utime(tmp_path/'frames/example.jpg', (100, 100))
    with pytest.raises(FramePending):
        read_screened(tmp_path, 'example', now=100)
    publish(tmp_path, image=b'x' * (MAX_IMAGE_BYTES + 1), now=100)
    with pytest.raises(ReferenceError):
        read_screened(tmp_path, 'example', now=100)


def test_new_status_without_new_jpeg_is_not_a_matching_reference(tmp_path):
    publish(tmp_path, now=100)
    # Encoding failed, or only metadata was replaced: never associate this
    # older JPEG with the later capture timestamp / pan-tilt position.
    state = tmp_path/'frames/example.json'
    state.write_text(json.dumps({'privacy':'clear','timestamp':100}))
    os.utime(state, (100,100))
    with pytest.raises(FramePending): read_screened(tmp_path, 'example', now=100)
    with pytest.raises(FramePending): require_screened(tmp_path, 'example', now=100)


@pytest.mark.parametrize('privacy,stamp', [('person',99),('unavailable',99),('clear',89),('clear',float('nan')),('clear',102)])
def test_screened_state_fails_closed(tmp_path, privacy, stamp):
    publish(tmp_path, privacy=privacy, timestamp=stamp, now=100)
    with pytest.raises(ReferenceError): read_screened(tmp_path, 'example', now=100)
    with pytest.raises(ReferenceError): require_screened(tmp_path, 'example', now=100)


def test_owner_camera_expiry_quotas_and_single_use():
    clock = [0]
    store = ReferenceStore(max_drafts=2, per_owner=1, ttl=10, clock=lambda: clock[0])
    token = store.reserve('owner', 'example')
    with pytest.raises(ReferenceError): store.reserve('owner', 'other')
    store.finish(token, jpeg(), '{"pan":0}')
    for owner, camera in [('other','example'),('owner','other')]:
        with pytest.raises(ReferenceError): store.claim(token, owner, camera)
    assert store.claim(token, 'owner', 'example').image == jpeg()
    with pytest.raises(ReferenceError): store.claim(token, 'owner', 'example')
    token = store.reserve('owner', 'example'); store.finish(token, jpeg())
    store.reserve('other', 'example')
    with pytest.raises(ReferenceError): store.reserve('third', 'example')
    clock[0] = 10
    with pytest.raises(ReferenceError): store.claim(token, 'owner', 'example')
    store.reserve('third', 'example')  # Expired reservations free capacity too.


def test_refresh_invalidates_previous_picture_and_pose_validation():
    store = ReferenceStore()
    first = store.reserve('owner', 'example'); store.finish(first, jpeg())
    store.reserve('owner', 'example')
    with pytest.raises(ReferenceError): store.claim(first, 'owner', 'example')
    pose = stationary_pose({'pan':.1,'tilt':.2,'moving':False})
    assert matching_pose(pose, dict(pose)) == pose
    for bad in [dict(pose, moving=True),dict(pose,pan=float('nan')),dict(pose,pan=.2),{}]:
        with pytest.raises(ReferenceError): matching_pose(pose, bad)


@pytest.fixture
def zone_app(setup, monkeypatch, tmp_path):
    module = setup[0]
    monkeypatch.setattr(module, 'runtime', tmp_path)
    monkeypatch.setitem(module.config, 'runtime', str(tmp_path))
    monkeypatch.setitem(module.config, 'cameras', [
        {'id':'example','name':'Example','ptz_supported':False,'onvif_url':'http://fictional.invalid'},
        {'id':'other','name':'Other','ptz_supported':False}])
    monkeypatch.setattr(module, 'zone_references', ReferenceStore())
    controller = SimpleNamespace(locks={}, position=AsyncMock(), move=AsyncMock())
    monkeypatch.setattr(module, 'ptz', controller)
    async def no_poll(): await asyncio.Event().wait()
    monkeypatch.setattr(module, 'poll_positions', no_poll)
    module.initialize(module.config)
    publish(tmp_path)
    publish(tmp_path, 'other')
    return module, tmp_path, controller


def body(token=None, camera='example'):
    return {'camera':camera,'name':'Fictional bowl','kind':'water','rect':[.1,.2,.3,.4], 'reference_token':token}


def test_auth_fixed_camera_and_exact_shown_bytes_survive_rolling_frame_change(zone_app):
    module, root, controller = zone_app
    with TestClient(module.app,base_url='https://testserver') as client:
        assert client.post('/api/cameras/example/zone-reference').status_code == 401
        unlock(client)
        assert client.post('/api/zones',json=body()).status_code == 409
        frozen = client.post('/api/cameras/example/zone-reference')
        assert frozen.status_code == 200 and frozen.content == jpeg()
        assert frozen.headers['cache-control'] == 'no-store'
        token = frozen.headers['x-aviary-zone-reference']
        publish(root, image=jpeg('red'))
        assert client.post('/api/zones',json=body(token,'other')).status_code == 409
        saved = client.post('/api/zones',json=body(token))
        assert saved.status_code == 200
        assert (root/'zone-references'/(saved.json()['id']+'.jpg')).read_bytes() == frozen.content
        assert client.get('/api/zones').json()[0]['anchor'] == ''
        assert client.post('/api/zones',json=body(token)).status_code == 409
        controller.position.assert_not_awaited(); controller.move.assert_not_awaited()


@pytest.mark.parametrize('failure',['person','stale','missing','write'])
def test_save_fails_closed_without_active_zone(zone_app, monkeypatch, failure):
    module, root, _ = zone_app
    with TestClient(module.app,base_url='https://testserver') as client:
        unlock(client)
        frozen = client.post('/api/cameras/example/zone-reference')
        token = frozen.headers['x-aviary-zone-reference']
        if failure == 'person': publish(root, privacy='person')
        if failure == 'stale': publish(root, timestamp=time.time()-20)
        if failure == 'missing': (root/'frames/example.jpg').unlink()
        if failure == 'write': (root/'zone-references').write_text('fictional filesystem obstruction')
        result = client.post('/api/zones',json=body(token))
        assert result.status_code == 503
        assert str(root) not in result.text
        assert client.get('/api/zones').json() == []


def test_reference_cannot_cross_authenticated_sessions(zone_app):
    module, _, _ = zone_app
    with TestClient(module.app,base_url='https://testserver') as first, TestClient(module.app,base_url='https://testserver') as second:
        unlock(first); unlock(second)
        frozen = first.post('/api/cameras/example/zone-reference')
        token = frozen.headers['x-aviary-zone-reference']
        assert second.post('/api/zones',json=body(token)).status_code == 409
        assert first.post('/api/zones',json=body(token)).status_code == 200


def test_database_failure_removes_reference_and_creates_no_zone(zone_app, monkeypatch):
    module, root, _ = zone_app
    with TestClient(module.app,base_url='https://testserver') as client:
        unlock(client)
        frozen = client.post('/api/cameras/example/zone-reference')
        original_database = module.database
        @contextmanager
        def fail_insert(config):
            with original_database(config) as db:
                def execute(query, *args):
                    if query.startswith('INSERT INTO zones'):
                        # The complete image must already exist before the row.
                        assert next((root/'zone-references').glob('*.jpg')).read_bytes() == frozen.content
                        raise OSError('Fictional database failure')
                    return db.execute(query, *args)
                yield SimpleNamespace(execute=execute)
        monkeypatch.setattr(module, 'database', fail_insert)
        saved = client.post('/api/zones',json=body(frozen.headers['x-aviary-zone-reference']))
        assert saved.status_code == 503
        assert list((root/'zone-references').glob('*.jpg')) == []
        assert client.get('/api/zones').json() == []


@pytest.mark.parametrize('change',['unchanged','moved','person'])
def test_delayed_pose_uses_frame_captured_between_readings_and_never_moves(zone_app, change):
    module, root, controller = zone_app
    module.config['cameras'][0]['ptz_supported'] = True
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0
        async def position(camera):
            nonlocal calls
            calls += 1
            if calls == 1:
                # Return the initial pose; the route must wait for a NEW frame.
                entered.set()
            else:
                await release.wait()
                if change == 'person': publish(root, privacy='person')
            return {'pan':.2 if calls > 1 and change == 'moved' else 0, 'tilt':0,'moving':False}
        controller.position.side_effect = position
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=module.app),base_url='https://testserver') as client:
            assert (await client.post('/api/login',json={'pin':'12345678'})).status_code == 200
            pending = asyncio.create_task(client.post('/api/cameras/example/zone-reference'))
            await asyncio.wait_for(entered.wait(), 2)
            await asyncio.sleep(.02)
            assert not pending.done()
            publish(root, image=jpeg('blue'), timestamp=time.time())
            # A newer rolling image while GetStatus is delayed cannot replace
            # the bytes already captured for this editor.
            deadline = time.monotonic()+2
            while calls < 2:
                assert time.monotonic() < deadline
                await asyncio.sleep(.01)
            publish(root, image=jpeg('red'))
            release.set()
            result = await asyncio.wait_for(pending, 2)
            if change != 'unchanged':
                assert result.status_code == 503
            else:
                assert result.status_code == 200 and result.content == jpeg('blue')
                saved = await client.post('/api/zones',json=body(result.headers['x-aviary-zone-reference']))
                assert saved.status_code == 200
                assert (root/'zone-references'/(saved.json()['id']+'.jpg')).read_bytes() == jpeg('blue')
                assert json.loads((await client.get('/api/zones')).json()[0]['anchor'])['pan'] == 0
        controller.move.assert_not_awaited()
    asyncio.run(scenario())
