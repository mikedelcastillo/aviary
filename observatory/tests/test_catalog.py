"""Finished clips stay discoverable without waiting for visual inference."""
import os
from pathlib import Path
import subprocess
import time

from backend import catalog
from backend.store import database, initialize


def test_invalid_recording_is_retried_only_when_changed(tmp_path,monkeypatch):
    config={'runtime':str(tmp_path)};initialize(config)
    folder=tmp_path/'recordings'/'example';folder.mkdir(parents=True)
    path=folder/'20260101T120000Z.mp4';path.write_bytes(b'x'*10001)
    old=time.time()-120;os.utime(path,(old,old))
    calls=[]
    def broken(file):
        calls.append(file)
        raise subprocess.CalledProcessError(1,'ffprobe')
    monkeypatch.setattr(catalog,'probe',broken)
    assert catalog.scan(config)['invalid']==1
    assert catalog.scan(config)['probed']==0
    assert len(calls)==1
    path.write_bytes(b'y'*10002);os.utime(path,(old,old))
    monkeypatch.setattr(catalog,'probe',lambda _:20.0)
    assert catalog.scan(config)['indexed']==1
    with database(config) as db:
        assert db.execute('SELECT count(*) FROM recording_failures').fetchone()[0]==0
        assert db.execute('SELECT duration FROM clips').fetchone()[0]==20
    assert catalog.scan(config)['probed']==0


def test_active_and_changing_segments_do_not_enter_feed(tmp_path,monkeypatch):
    config={'runtime':str(tmp_path)};initialize(config)
    folder=tmp_path/'recordings'/'example';folder.mkdir(parents=True)
    path=folder/'20260101T120000Z.mp4';path.write_bytes(b'x'*10001)
    assert catalog.scan(config)['probed']==0
    old=time.time()-120;os.utime(path,(old,old))
    def changing(file):
        file.write_bytes(b'x'*20000)
        return 20
    monkeypatch.setattr(catalog,'probe',changing)
    assert catalog.scan(config)['indexed']==0
    with database(config) as db:assert db.execute('SELECT count(*) FROM clips').fetchone()[0]==0
