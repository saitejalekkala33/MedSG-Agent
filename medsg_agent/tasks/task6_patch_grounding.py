
from __future__ import annotations
from typing import Optional, Tuple, Dict, List
import numpy as np, cv2
from ..utils import BBox, to_gray01, tight_mask, mask_edges, rotate_keep, edge_closeness_map

def _prep_grad(x: np.ndarray) -> np.ndarray:
    g = to_gray01(x)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    m = np.sqrt(gx*gx+gy*gy); m -= m.min(); m /= (m.max()+1e-8)
    return m.astype(np.float32)

def patch_grounding_bbox_robust(patch: np.ndarray, target_img: np.ndarray,
                                scales: List[float] = None,
                                rotations: List[float] = None,
                                alpha: float = 0.6) -> Tuple[Optional[BBox], float, Dict]:
    """
    Task 6: Patch grounding with gradient + edge-closeness fusion.
    """
    if scales is None: scales = [1.05, 1.0, 0.95, 0.9, 0.85, 0.8]
    if rotations is None: rotations = [0, 7.5, -7.5]

    P_mask = tight_mask(patch)
    P_edges = (mask_edges(P_mask).astype(np.float32)/255.0)
    P_grad  = _prep_grad(patch)
    T_grad  = _prep_grad(target_img)
    T_close = edge_closeness_map(target_img)

    Hp,Wp = P_grad.shape; Ht,Wt = T_grad.shape
    smax = min((Wt-1)/max(Wp,1),(Ht-1)/max(Hp,1))
    scales = [min(s, smax) for s in scales if min(int(Wp*s),Wt)>7 and min(int(Hp*s),Ht)>7]
    if not scales: return None, 0.0, {"reason":"no_valid_scale"}

    best = (-1.0, None, {})
    for ang in rotations:
        Pg = rotate_keep(P_grad, ang)
        Pe = rotate_keep(P_edges, ang)
        for s in scales:
            ww,hh = int(round(Wp*s)), int(round(Hp*s))
            Pg_s = cv2.resize(Pg, (ww,hh), interpolation=cv2.INTER_AREA)
            Pe_s = cv2.resize(Pe, (ww,hh), interpolation=cv2.INTER_NEAREST)
            if Pg_s.sum()==0 or Pe_s.sum()==0: continue
            try:
                res1 = cv2.matchTemplate(T_grad, Pg_s, cv2.TM_CCOEFF_NORMED, mask=(Pe_s*255).astype(np.uint8))
            except cv2.error:
                res1 = cv2.matchTemplate(T_grad, Pg_s*Pe_s, cv2.TM_CCOEFF_NORMED)
            res2 = cv2.matchTemplate(T_close, Pe_s, cv2.TM_CCORR_NORMED)
            res = alpha*res1 + (1.0-alpha)*res2
            _, mv, _, ml = cv2.minMaxLoc(res)
            x,y = ml
            bb = BBox(int(x),int(y),int(x+ww),int(y+hh))
            if mv > best[0]:
                best = (float(mv), bb, {"scale":float(s), "angle":float(ang)})
    return best[1], float(max(0.0,best[0])), best[2]
