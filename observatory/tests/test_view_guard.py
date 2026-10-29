"""Synthetic textured scenes verify relocation rejection without private images."""
import cv2
import numpy as np
from backend.view_guard import ViewGuard


def scene():
    random = np.random.default_rng(114)
    image = np.full((360,640,3), 130, dtype=np.uint8)
    for _ in range(220):
        x, y = random.integers([15,15], [620,340])
        color = tuple(int(x) for x in random.integers(0,255,3))
        cv2.rectangle(image,(int(x),int(y)),(int(x)+9,int(y)+7),color,-1)
    return image


def test_relocation_stays_invalid_after_picture_settles(tmp_path):
    root=tmp_path/'zone-references';root.mkdir()
    reference=scene();cv2.imwrite(str(root/'example.jpg'),reference)
    guard=ViewGuard(tmp_path);zone={'id':'example'}
    assert not guard.check(zone,reference)['valid']
    assert not guard.check(zone,reference)['valid']
    assert guard.check(zone,reference)['valid']
    shifted=cv2.warpAffine(reference,np.float32([[1,0,35],[0,1,20]]),(640,360))
    for _ in range(8): assert not guard.check(zone,shifted)['valid']
    for _ in range(2): assert not guard.check(zone,reference)['valid']
    assert guard.check(zone,reference)['valid']


def test_missing_or_featureless_view_fails_closed(tmp_path):
    guard=ViewGuard(tmp_path)
    assert not guard.check({'id':'missing'},scene())['valid']
    root=tmp_path/'zone-references';root.mkdir()
    flat=np.full((360,640,3),127,dtype=np.uint8);cv2.imwrite(str(root/'flat.jpg'),flat)
    for _ in range(4): assert not guard.check({'id':'flat'},flat)['valid']
