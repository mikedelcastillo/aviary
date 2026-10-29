import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.learning_import import import_corpus
from backend.store import database


def test_corpus_is_read_only_and_detector_sidecars_are_not_labels(tmp_path):
    source=tmp_path/'copied-corpus'
    folder=source/'tapo'/'day';folder.mkdir(parents=True)
    names=['camera-1_day_2026-10-01_010000_a.jpg','camera-1_day_2026-10-01_012000_b.jpg',
           'camera-1_day_2026-10-01_014000_c.jpg','camera-1_day_2026-10-01_020000_d.jpg']
    image=np.random.default_rng(10).integers(0,255,(180,320,3),dtype=np.uint8)
    for name in names:assert cv2.imwrite(str(folder/name),image)
    (folder/names[0]).with_suffix('.txt').write_text('0 .5 .5 .5 .5\n')
    (folder/names[2]).with_suffix('.json').write_text(json.dumps({'v':1,'boxed':True,'boxes':[]}))
    (folder/names[3]).with_suffix('.json').write_text('broken canonical annotation')
    before={p.relative_to(source):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    config={'runtime':str(tmp_path/'runtime'),'timezone':'Asia/Manila'}
    result=import_corpus(config,source,limit=10)
    assert result=={'added':1,'known':1,'already_reviewed':1,'invalid_sidecar':1}
    with database(config) as db:sample=dict(db.execute('SELECT * FROM learning_samples').fetchone())
    assert sample['suggestions']=='[]' and sample['clip'] is None
    assert sample['group_id'].startswith('camera-1|2026-10-01|')
    metadata=json.loads(sample['signature'])['source']
    assert metadata['category']=='tapo/day' and metadata['timestamp_basis']=='capture_filename'
    assert metadata['sha256']==before[Path('tapo/day')/metadata['name']]
    assert (tmp_path/'runtime'/'learning'/sample['id']/'frame.jpg').is_file()
    assert not (tmp_path/'runtime'/'learning'/sample['id']/'context.mp4').exists()
    after={p.relative_to(source):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    assert before==after
    assert import_corpus(config,source,limit=10).get('added',0)==0


def test_import_rejects_runtime_overlap_and_skips_links_outside_copy(tmp_path):
    runtime=tmp_path/'runtime';runtime.mkdir()
    config={'runtime':str(runtime),'timezone':'UTC'}
    with pytest.raises(ValueError,match='separate local corpus copy'):
        import_corpus(config,tmp_path)
    source=tmp_path/'copy';folder=source/'phone';folder.mkdir(parents=True)
    original=tmp_path/'outside.jpg'
    assert cv2.imwrite(str(original),np.zeros((32,32,3),np.uint8))
    (folder/'symlink.jpg').symlink_to(original)
    assert import_corpus(config,source)=={'outside_source':1}
