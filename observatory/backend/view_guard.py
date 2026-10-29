"""Conservative image alignment for zones; this is not a 3D calibration.

A settled or unchanged PTZ encoder does not prove a tabletop camera stayed put.
Compare against the exact saved view, require distributed static features and
three confirmations, and fail closed if the view cannot be established.
"""
from pathlib import Path
import cv2
import numpy as np


def gray(image):
    return cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (640, 360))


class ViewGuard:
    def __init__(self, runtime):
        self.root = Path(runtime) / 'zone-references'
        self.orb = cv2.ORB_create(nfeatures=1200, fastThreshold=14)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.references = {}
        self.confirmations = {}

    def check(self, zone, image, boxes=()):
        key = zone['id']
        path = self.root / (key + '.jpg')
        if not path.is_file():
            self.confirmations[key] = 0
            return {'valid': False, 'reason': 'Reference view needed'}
        signature = path.stat().st_mtime_ns
        cached = self.references.get(key)
        if not cached or cached[0] != signature:
            reference = cv2.imread(str(path))
            if reference is None:
                return {'valid': False, 'reason': 'Reference unavailable'}
            points, descriptors = self.orb.detectAndCompute(gray(reference), None)
            cached = (signature, points, descriptors)
            self.references[key] = cached
            self.confirmations[key] = 0
        mask = np.full((360, 640), 255, dtype=np.uint8)
        height, width = image.shape[:2]
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(mask, (int(x1/width*640)-5, int(y1/height*360)-5),
                          (int(x2/width*640)+5, int(y2/height*360)+5), 0, -1)
        points, descriptors = self.orb.detectAndCompute(gray(image), mask)
        result = {'valid': False, 'reason': 'View not verified'}
        if cached[2] is not None and descriptors is not None and len(descriptors) >= 20:
            pairs = self.matcher.knnMatch(cached[2], descriptors, k=2)
            good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < .7*p[1].distance]
            if len(good) >= 24:
                source = np.float32([cached[1][m.queryIdx].pt for m in good])
                target = np.float32([points[m.trainIdx].pt for m in good])
                transform, inliers = cv2.findHomography(source, target, cv2.RANSAC, 2.5)
                if transform is not None and inliers is not None:
                    keep = inliers.ravel().astype(bool)
                    count = int(keep.sum())
                    coverage = cv2.contourArea(cv2.convexHull(source[keep]))/(640*360) if count >= 3 else 0
                    corners = np.float32([[[0,0],[640,0],[640,360],[0,360]]])
                    projected = cv2.perspectiveTransform(corners, transform)
                    displacement = float(np.max(np.linalg.norm(projected-corners, axis=2)))
                    # Unchanged room features cannot validate a cage that moved.
                    # Require matching structure near the actual saved zone too.
                    x,y,w,h = zone.get('rect',[0,0,1,1])
                    if isinstance(x,str):
                        raise ValueError('Zone rectangle must be decoded')
                    nearby=(source[:,0]>=max(0,x-w/2)*640)&(source[:,0]<=min(1,x+1.5*w)*640)&(source[:,1]>=max(0,y-h/2)*360)&(source[:,1]<=min(1,y+1.5*h)*360)
                    local_count=int((nearby&keep).sum())
                    local_displacement=float(np.median(np.linalg.norm(target[nearby&keep]-source[nearby&keep],axis=1))) if local_count else float('inf')
                    result.update(inliers=count, local_inliers=local_count, coverage=round(coverage, 3), displacement_px=round(displacement, 2))
                    result['valid'] = count >= 20 and count/len(good) >= .55 and coverage >= .15 and displacement <= 7 and local_count >= 6 and local_displacement <= 5
                    if result['valid']: result['reason'] = 'Reference view matched'
        self.confirmations[key] = self.confirmations.get(key, 0)+1 if result['valid'] else 0
        result['valid'] = self.confirmations[key] >= 3
        if self.confirmations[key] in (1, 2): result['reason'] = 'Confirming reference view'
        return result
