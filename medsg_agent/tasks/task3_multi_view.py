
from __future__ import annotations
from typing import Optional, Tuple, Dict, List
import numpy as np, cv2, re
from ..utils import BBox, clip_bbox, to_gray01

def _prep(x: np.ndarray) -> np.ndarray:
    g = to_gray01(x)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = np.sqrt(gx*gx+gy*gy); m -= m.min(); m /= (m.max()+1e-8)
    return m.astype(np.float32)

def _ncc_dyn(ref_patch: np.ndarray, tgt: np.ndarray, rots: Tuple[float,...]=(0,90,-90)):
    best = (-1, None, {"angle": None, "scale": None})
    T = _prep(tgt); Ht, Wt = T.shape
    for ang in rots:
        if ang % 360 == 0:
            rp = ref_patch
        else:
            h,w = ref_patch.shape[:2]; c=(w/2,h/2)
            M = cv2.getRotationMatrix2D(c, ang, 1.0)
            rp = cv2.warpAffine(ref_patch, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        P = _prep(rp); Hp, Wp = P.shape
        smax = min((Wt-1)/max(Wp,1),(Ht-1)/max(Hp,1))
        if smax <= 0: 
            continue
        scales = sorted(set([smax,smax*0.95,smax*0.9,1.0,0.95,0.9,0.85,0.8,0.75,0.7]), reverse=True)
        use = []
        for s in scales:
            s = min(float(s), float(smax))
            ww = int(max(8, round(Wp*s))); hh = int(max(8, round(Hp*s)))
            if ww <= Wt and hh <= Ht: use.append(s)
        if not use: 
            continue
        for s in use:
            Ps = cv2.resize(P, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(T, Ps, cv2.TM_CCOEFF_NORMED)
            _, mv, _, loc = cv2.minMaxLoc(res)
            y, x = loc[1], loc[0]
            hh, ww = Ps.shape
            bb = BBox(int(x), int(y), int(x+ww), int(y+hh))
            if mv > best[0]:
                best = (float(mv), bb, {"angle": ang, "scale": float(s)})
    return best[1], best[0], best[2]

def _orb_map(r: np.ndarray, t: np.ndarray, rb: BBox):
    orb = cv2.ORB_create(2500)
    mask = np.zeros(r.shape[:2], np.uint8); mask[rb.y1:rb.y2, rb.x1:rb.x2] = 255
    k1,d1 = orb.detectAndCompute(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY), mask)
    k2,d2 = orb.detectAndCompute(cv2.cvtColor(t, cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None or len(k1)<6 or len(k2)<6: return None, 0.0, {}
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)[:300]
    if len(matches) < 6: return None, 0.0, {}
    pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([k2[m.trainIdx].pt for m in matches])
    M, inl = cv2.estimateAffinePartial2D(pts1, pts2, ransacReprojThreshold=2.5, confidence=0.99)
    if M is None: return None, 0.0, {}
    crn = np.array([[rb.x1,rb.y1,1],[rb.x2,rb.y1,1],[rb.x2,rb.y2,1],[rb.x1,rb.y2,1]], dtype=np.float32).T
    tg = (M @ crn).T
    tx1, ty1 = tg.min(axis=0); tx2, ty2 = tg.max(axis=0)
    bb = clip_bbox(BBox(int(tx1),int(ty1),int(tx2),int(ty2)), t.shape[0], t.shape[1])
    conf = float((inl.ravel()==1).mean()) if inl is not None else 0.0
    ar_ref = (rb.x2-rb.x1)/(rb.y2-rb.y1+1e-8); ar_t = (bb.x2-bb.x1)/(bb.y2-bb.y1+1e-8) if (bb.y2-bb.y1)!=0 else (bb.x2-bb.x1)
    # Guard against division by zero handled above; coarse aspect-ratio penalty not used to keep code short.
    return bb, conf, {"inliers": int((inl.ravel()==1).sum()) if inl is not None else 0}

def multi_view_grounding_bbox(ref_img: np.ndarray, tgt_img: np.ndarray,
                              ref_bbox: BBox, method: str = "orb") -> Tuple[Optional[BBox], float, Dict]:
    """
    Task 3: Multi-view grounding
    """
    r = ref_img.copy(); t = tgt_img.copy()
    x1,y1,x2,y2 = ref_bbox.x1, ref_bbox.y1, ref_bbox.x2, ref_bbox.y2
    ref_patch = r[y1:y2, x1:x2]

    if method == "ncc":
        bb, sc, dbg = _ncc_dyn(ref_patch, t, rots=(0,90,-90))
        conf = (sc + 1.0) / 2.0
        return bb, conf, {"method":"ncc", **dbg}

    if method == "orb":
        bb, conf, dbg = _orb_map(r, t, ref_bbox)
        if bb is not None and conf >= 0.2:
            return bb, conf, {"method":"orb", **dbg}
        # fallback
        bb2, sc2, dbg2 = _ncc_dyn(ref_patch, t, rots=(0,90,-90))
        conf2 = (sc2 + 1.0) / 2.0
        return bb2, conf2, {"method":"ncc", **dbg2}

    # auto: choose best of both
    bb1, c1, d1 = _orb_map(r, t, ref_bbox)
    bb2, sc2, d2 = _ncc_dyn(ref_patch, t, rots=(0,90,-90))
    c2 = (sc2 + 1.0) / 2.0
    cands = []
    if bb1 is not None: cands.append(("orb", bb1, c1, d1))
    if bb2 is not None: cands.append(("ncc", bb2, c2, d2))
    if not cands: return None, 0.0, {}
    m, bb, cf, ex = max(cands, key=lambda z: z[2])
    return bb, cf, {"method": m, **ex}
