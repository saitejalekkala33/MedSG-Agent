
from __future__ import annotations
from typing import Optional, Tuple, Dict, List
import numpy as np, cv2
from ..utils import BBox, tight_mask, mask_edges, edge_closeness_map, rotate_keep, to_gray01

def concept_match_bbox(ref_img: np.ndarray, ref_bbox: Optional[BBox], tgt_img: np.ndarray, method: str = "auto") -> Tuple[Optional[BBox], float, Dict]:
    """
    Task 5: Concept match via edge-chamfer style correlation.
    If ref_bbox is None, a tight mask is computed on ref_img.
    """
    if ref_bbox is None:
        m = tight_mask(ref_img)
        cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            h,w = m.shape; ref_bbox = BBox(0,0,w,h)
        else:
            c = max(cnts, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c); ref_bbox = BBox(x,y,x+w,y+h)

    x1,y1,x2,y2 = ref_bbox.x1, ref_bbox.y1, ref_bbox.x2, ref_bbox.y2
    patch = ref_img[y1:y2, x1:x2]
    mask = tight_mask(patch)
    edges = mask_edges(mask)
    if edges.sum()==0:
        edges = cv2.Canny((to_gray01(patch)*255).astype(np.uint8), 30, 60)

    C = edge_closeness_map(tgt_img)
    Hp,Wp = edges.shape[:2]; Ht,Wt = C.shape[:2]
    smax = min((Wt-1)/max(Wp,1), (Ht-1)/max(Hp,1))
    if smax <= 0: return None, 0.0, {}
    scales = sorted(set([smax, smax*0.95, smax*0.9, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]), reverse=True)
    rots = [0, 10, -10, 20, -20]
    best = (-1.0, None, {})
    edges_f = edges.astype(np.float32)/255.0

    for ang in rots:
        er = rotate_keep(edges_f, ang)
        for s in scales:
            s = float(min(s, smax))
            ww = int(max(8, round(Wp*s))); hh = int(max(8, round(Hp*s)))
            if ww>Wt or hh>Ht: continue
            ker = cv2.resize(er, (ww, hh), interpolation=cv2.INTER_NEAREST)
            if ker.sum() < 1.0: continue
            res = cv2.matchTemplate(C, ker, cv2.TM_CCORR_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            x,y = ml; bb = BBox(int(x), int(y), int(x+ww), int(y+hh))
            if mv > best[0]:
                best = (float(mv), bb, {"scale": s, "angle": ang})
    return best[1], float(max(0.0, best[0])), {"mode":"edge_chamfer_ccorr", **best[2]}
