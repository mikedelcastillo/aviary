import json
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

from backend.learning_sampling import fingerprint, near_duplicate, offer
from backend.store import initialize, database


def test_dedupe_uses_full_frame_and_tiles():
    rng=np.random.default_rng(42)
    image=rng.integers(0,255,(180,320,3),dtype=np.uint8)
    original=fingerprint(image)
    assert near_duplicate(original,fingerprint(image.copy()))
    changed=image.copy();changed[:60,:80]=255
    assert not near_duplicate(original,fingerprint(changed))
    gray=cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
    assert not near_duplicate(original,fingerprint(gray))


def test_zero_detections_still_enters_queue_and_keeps_video(tmp_path):
    config={'runtime':str(tmp_path)};initialize(config)
    source=tmp_path/'recordings'/'sample.mp4';source.parent.mkdir();source.write_bytes(b'fictional video')
    row={'id':'sample','camera':'example','time':datetime.now(timezone.utc).isoformat(),'path':'recordings/sample.mp4','duration':20}
    image=np.random.default_rng(4).integers(0,255,(180,320,3),dtype=np.uint8)
    assert offer(config,row,image,[])=='added'
    with database(config) as db:sample=dict(db.execute('SELECT * FROM learning_samples').fetchone())
    assert sample['reason']=='audit' and json.loads(sample['suggestions'])==[]
    assert (tmp_path/'learning'/sample['id']/'context.mp4').read_bytes()==source.read_bytes()
    assert offer(config,row,image,[])=='known'


def test_concurrent_offers_obey_capture_window_and_budget(tmp_path):
    config={'runtime':str(tmp_path),'learning_queue_limit':1};initialize(config)
    source=tmp_path/'source.mp4';source.write_bytes(b'fictional video')
    rows=[{'id':f'clip-{i}','camera':'example','time':'2026-10-01T01:01:00+00:00','path':'source.mp4','duration':20} for i in range(2)]
    images=[np.random.default_rng(i).integers(0,255,(180,320,3),dtype=np.uint8) for i in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda item:offer(config,item[0],item[1],[]),zip(rows,images)))
    assert sorted(results)==['added','full']
    with database(config) as db:assert db.execute('SELECT count(*) FROM learning_samples').fetchone()[0]==1


def test_expired_context_does_not_create_an_unreviewable_queue_item(tmp_path):
    config={'runtime':str(tmp_path)};initialize(config)
    row={'id':'missing','camera':'example','time':'2026-10-01T01:01:00+00:00','path':'missing.mp4','duration':20}
    image=np.random.default_rng(4).integers(0,255,(180,320,3),dtype=np.uint8)
    assert offer(config,row,image,[])=='expired'
    assert not (tmp_path/'learning').exists()
