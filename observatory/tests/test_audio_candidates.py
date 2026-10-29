import hashlib
import json

import numpy as np
import pytest

from backend.audio_candidates import benchmark, segment_audio, write_wav


def chirps(rate=8000,duration=4):
    sound=np.random.default_rng(30).normal(0,.0001,round(rate*duration)).astype(np.float32)
    for start,end in [(1,1.25),(1.4,1.65)]:
        t=np.arange(round((end-start)*rate))/rate
        burst=.15*np.sin(2*np.pi*(1100*t+1000*t*t))*np.hanning(len(t))
        sound[round(start*rate):round(start*rate)+len(t)]+=burst
    return sound


def test_chirp_like_energy_forms_one_bounded_unlabeled_review_candidate():
    result=segment_audio(chirps(),8000)
    assert len(result['segments'])==1
    item=result['segments'][0]
    assert .4<item['start_seconds']<1
    assert 1.65<item['end_seconds']<2.2
    assert item['review_label'] is None and item['species'] is None and item['individual'] is None
    assert item['reasons']


def test_silence_low_rumble_and_quiet_stationary_noise_are_not_bird_labels():
    rate=8000;t=np.arange(rate*4)/rate
    for sound in [np.zeros(rate*4),.1*np.sin(2*np.pi*150*t),np.random.default_rng(8).normal(0,.0002,rate*4)]:
        assert segment_audio(sound,rate)['segments']==[]
    assert segment_audio(np.zeros(5),rate)['duration_seconds']>0


def test_source_bandwidth_is_retained_after_resampling():
    result=segment_audio(chirps(rate=16000),16000,source_rate=8000)
    assert result['sample_rate']==16000
    assert result['source_nyquist_hz']==4000
    assert result['band_hz']==[700,3500.0]
    with pytest.raises(ValueError):segment_audio(np.array([float('nan')]),8000)
    with pytest.raises(ValueError):segment_audio(np.zeros((3,2)),8000)


def test_private_listening_package_preserves_source_and_includes_independent_audit(tmp_path):
    source=tmp_path/'copied.wav';write_wav(source,chirps(),8000)
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    output=tmp_path/'review'
    report=benchmark([source],output,source_rate=8000)
    assert report['summary']['decoded_files']==1
    assert report['summary']['candidate_count']==1
    assert report['summary']['audio_seconds']==4
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    row=report['files'][0]
    assert row['sha256']==before and row['audit']['kind']=='independent_audit'
    for item in [*row['segments'],row['audit']]:assert (output/item['audio']).is_file()
    assert '<audio controls' in (output/'index.html').read_text()
    assert json.loads((output/'report.json').read_text())['summary']['accuracy'].startswith('Not measured')
    with pytest.raises(ValueError,match='empty output directory'):benchmark([source],output)


def test_silent_clip_still_gets_audit_and_mutating_source_is_rejected(tmp_path,monkeypatch):
    source=tmp_path/'silence.wav';write_wav(source,np.zeros(8000),8000)
    report=benchmark([source],tmp_path/'silent-review',source_rate=8000)
    assert report['summary']['candidate_count']==0
    assert report['files'][0]['audit']['end_seconds']==1
    def changed(path,maximum_seconds):
        path.write_bytes(b'fictional changed input')
        return np.zeros(8000),8000,{}
    monkeypatch.setattr('backend.audio_candidates.decode_audio',changed)
    result=benchmark([source],tmp_path/'changed-review')
    assert result['summary']['decoded_files']==0
    assert result['summary']['candidate_count']==0
    assert result['files'][0]['error']=='ValueError'
