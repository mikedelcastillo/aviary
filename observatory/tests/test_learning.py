import io
import json
import zipfile
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from backend.learning import create_router, select_diverse
from backend.store import initialize, database


@pytest.fixture
def context(tmp_path):
    config={'runtime':str(tmp_path),'timezone':'Asia/Tokyo','birds':[{'id':'fern'}],
        'annotation_labels':[{'name':'fern','kind':'individual','models':['live'],'index':3},
            {'name':'cockatiel','kind':'species','models':['live'],'index':8},
            {'name':'unknown_bird','kind':'unknown','models':['live'],'index':9}]}
    initialize(config)
    now=datetime.now(timezone.utc).isoformat()
    for number in range(3):
        key=str(number)*24
        folder=tmp_path/'learning'/key;folder.mkdir(parents=True)
        Image.new('RGB',(80,60),'green').save(folder/'frame.jpg')
        (folder/'context.mp4').write_bytes(b'fictional-media')
        with database(config) as db:db.execute('INSERT INTO learning_samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
            (key,'clip','example',now,10,'day','audit','[]','{}','capture-'+str(number),'pending',now))
    app=FastAPI()
    @app.middleware('http')
    async def reviewer(request,call_next):
        request.state.session=request.headers.get('reviewer','fictional-owner')
        return await call_next(request)
    app.include_router(create_router(config))
    with TestClient(app) as client:yield client,config,tmp_path


def test_complete_labels_export_and_undo_preserve_provenance(context):
    client,config,root=context;key='0'*24
    answer={'outcome':'annotated','lighting':'day','complete':True,
            'boxes':[{'id':'one','cx':.5,'cy':.5,'w':.4,'h':.6,'label':'fern'}]}
    assert client.post(f'/api/learn/{key}/answer',json={**answer,'complete':False}).status_code==422
    assert client.post(f'/api/learn/{key}/answer',json={**answer,'lighting':'ir'}).status_code==422
    result=client.post(f'/api/learn/{key}/answer',json=answer)
    assert result.status_code==200
    assert client.post(f'/api/learn/{key}/answer',json=answer).status_code==409
    export=client.get('/api/learn/export')
    with zipfile.ZipFile(io.BytesIO(export.content)) as archive:
        manifest=json.loads(archive.read('manifest.json'));item=manifest['frames'][0]
        prefix=item['image'][:-4]
        assert prefix.startswith('tapo/day/learn_')
        sidecar=json.loads(archive.read(prefix+'.json'))
        assert sidecar['boxed'] and sidecar['boxes'][0]['label']=='fern'
        assert sidecar['provenance']['source_group']=='capture-0'
        assert archive.read(prefix+'.txt').decode().startswith('3 0.500000 ')
    endpoint='/api/learn/answers/'+result.json()['answer_id']+'/undo'
    assert client.post(endpoint,headers={'reviewer':'different-owner'}).status_code==409
    assert client.post(endpoint).status_code==200
    assert (root/'learning'/key/'frame.jpg').exists()
    with zipfile.ZipFile(io.BytesIO(client.get('/api/learn/export').content)) as archive:
        assert json.loads(archive.read('manifest.json'))['frames']==[]


def test_uncertainty_is_not_a_negative_training_example(context):
    client,_,_=context
    assert client.post('/api/learn/'+'0'*24+'/answer',json={'outcome':'uncertain','lighting':'day'}).status_code==200
    assert client.post('/api/learn/'+'1'*24+'/answer',json={'outcome':'empty','lighting':'day'}).status_code==422
    assert client.post('/api/learn/'+'1'*24+'/answer',json={'outcome':'empty','lighting':'day','complete':True}).status_code==200
    with zipfile.ZipFile(io.BytesIO(client.get('/api/learn/export').content)) as archive:
        frames=json.loads(archive.read('manifest.json'))['frames']
        assert len(frames)==1
        assert archive.read(frames[0]['image'][:-4]+'.txt')==b''


def test_queue_reserves_audit_and_does_not_repeat_capture_windows():
    rows=[{'id':str(i),'camera':'a' if i<4 else 'b','reason':'audit' if i==4 else 'diverse','group_id':str(i//2)} for i in range(8)]
    chosen=select_diverse(rows)
    assert chosen[0]['reason']=='audit'
    assert len({r['group_id'] for r in chosen})==len(chosen)
    assert {r['camera'] for r in chosen}=={'a','b'}


def test_daily_goal_stops_at_five_and_requires_explicit_extra_batch(context):
    client,config,_=context
    now=datetime.now(timezone.utc)
    with database(config) as db:
        for number in range(4):
            db.execute('INSERT INTO learning_answers VALUES(?,?,?,?,?,?,?,0)',
                (f'answer-{number}',f'previous-{number}','owner',now.isoformat(),'uncertain','{}','day'))
        db.execute('INSERT INTO learning_answers VALUES(?,?,?,?,?,?,?,0)',
            ('yesterday','previous-yesterday','owner',(now-timedelta(days=2)).isoformat(),'annotated','{}','day'))
    assert client.get('/api/learn').json()['today']==4
    assert len(client.get('/api/learn').json()['cards'])==1
    skipped=client.post('/api/learn/'+'0'*24+'/answer',json={'outcome':'skip','lighting':'day'}).json()
    assert client.get('/api/learn').json()['today']==4
    assert client.post('/api/learn/answers/'+skipped['answer_id']+'/undo').status_code==200
    result=client.post('/api/learn/'+'0'*24+'/answer',json={'outcome':'uncertain','lighting':'day'}).json()
    assert client.get('/api/learn').json()['cards']==[]
    assert len(client.get('/api/learn?more=true').json()['cards'])==2
    assert client.post('/api/learn/answers/'+result['answer_id']+'/undo').status_code==200
    assert len(client.get('/api/learn').json()['cards'])==1


def test_invalid_geometry_and_roster_cannot_be_exported_as_labels(context):
    client,config,_=context
    box={'id':'one','cx':.99,'cy':.5,'w':.4,'h':.6,'label':'fern'}
    body={'outcome':'annotated','lighting':'day','complete':True,'boxes':[box]}
    assert client.post('/api/learn/'+'0'*24+'/answer',json=body).status_code==422
    body['boxes']=[{**box,'cx':.5},{**box,'cx':.3}]
    assert client.post('/api/learn/'+'0'*24+'/answer',json=body).status_code==422
    config['annotation_labels'].append(dict(config['annotation_labels'][0]))
    assert client.get('/api/learn/export').status_code==409


def test_imported_phone_preserves_filename_and_uses_archive_labels(context):
    client,config,root=context
    config['annotation_labels'].append({'name':'archived_bird','kind':'individual','models':['archive'],'index':12})
    key='0'*24
    (root/'learning'/key/'context.mp4').unlink()
    source={'category':'phone','name':'IMG_1001.png','sha256':'fictional-hash','timestamp_basis':'file_mtime_not_capture_time'}
    with database(config) as db:
        db.execute('UPDATE learning_samples SET signature=?,clip=NULL WHERE id=?',(json.dumps({'source':source}),key))
    card=next(c for c in client.get('/api/learn').json()['cards'] if c['id']==key)
    assert card['video'] is None and card['category']=='phone' and card['imported']
    body={'outcome':'annotated','lighting':'day','complete':True,
          'boxes':[{'id':'one','cx':.5,'cy':.5,'w':.4,'h':.6,'label':'archived_bird'}]}
    assert client.post('/api/learn/'+'1'*24+'/answer',json=body).status_code==422
    assert client.post('/api/learn/'+key+'/answer',json=body).status_code==200
    with zipfile.ZipFile(io.BytesIO(client.get('/api/learn/export').content)) as archive:
        annotation=json.loads(archive.read('phone/IMG_1001.json'))
        assert annotation['v']==1 and annotation['boxed']
        assert annotation['boxes'][0]['label']=='archived_bird'
        assert annotation['provenance']['import']==source
        assert archive.read('phone/IMG_1001.txt').decode().startswith('12 ')
