
from __future__ import annotations
from typing import Optional, Tuple, Dict, List
import numpy as np, cv2
from ..utils import BBox, tight_mask, to_gray01, rotate_keep, rotate_any, resize_any, edge_closeness_map

def _clahe01(x: np.ndarray) -> np.ndarray:
    g = (to_gray01(x)*255).astype(np.uint8)
    cla = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(g)
    return cla.astype(np.float32)/255.0

def _mind_lite(gray01: np.ndarray) -> np.ndarray:
    g = (gray01*255).astype(np.uint8)
    offs = [(0,1),(1,0),(1,1),(-1,1),(0,2),(2,0)]
    feats=[]
    for dy,dx in offs:
        M = np.float32([[1,0,dx],[0,1,dy]])
        sh = cv2.warpAffine(g, M, (g.shape[1], g.shape[0]))
        d  = cv2.GaussianBlur((g.astype(np.float32)-sh.astype(np.float32))**2,(3,3),0)
        feats.append(d)
    F = np.stack(feats,-1).astype(np.float32)
    F -= F.min(); F /= (F.max()+1e-8)
    return F

def _edges_from_mask(mask: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    e = cv2.subtract(cv2.dilate(mask,k,1), cv2.erode(mask,k,1))
    e[e>0]=255
    return (e.astype(np.float32)/255.0)

def _detect_colored_bbox(img_bgr: np.ndarray) -> Optional[BBox]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0,80,80), (10,255,255))
    m2 = cv2.inRange(hsv, (170,80,80), (180,255,255))
    m  = cv2.morphologyEx(cv2.bitwise_or(m1,m2), cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=2)
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        b = img_bgr[:,:,2].astype(np.int16) - np.maximum(img_bgr[:,:,1], img_bgr[:,:,0]).astype(np.int16)
        m = (b>80).astype(np.uint8)*255
        cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    return BBox(x,y,x+w,y+h)

def crossmodal_grounding_bbox(ref_img: np.ndarray, ref_bbox: Optional["BBox"], tgt_img: np.ndarray,
                              scales: List[float]=None, rotations: List[float]=None,
                              weights: Tuple[float,float,float]=(0.45,0.35,0.20)) -> Tuple[Optional["BBox"], float, Dict]:
    if scales is None: scales=[1.15,1.1,1.05,1.0,0.95,0.9,0.85]
    if rotations is None: rotations=[0,7.5,-7.5,15,-15]

    if ref_bbox is None:
        rb = _detect_colored_bbox(ref_img)
        if rb is None:
            m = tight_mask(ref_img)
            cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                h,w = ref_img.shape[:2]; ref_bbox = BBox(0,0,w-1,h-1)
            else:
                c = max(cnts, key=cv2.contourArea); x,y,w,h = cv2.boundingRect(c); ref_bbox = BBox(x,y,x+w,y+h)
        else:
            ref_bbox = rb

    x1,y1,x2,y2 = ref_bbox.x1, ref_bbox.y1, ref_bbox.x2, ref_bbox.y2
    ref_patch   = ref_img[y1:y2, x1:x2]

    rp_g01      = _clahe01(ref_patch)
    gx = cv2.Sobel(rp_g01,cv2.CV_32F,1,0,3); gy = cv2.Sobel(rp_g01,cv2.CV_32F,0,1,3)
    rp_grad     = np.sqrt(gx*gx+gy*gy); rp_grad-=rp_grad.min(); rp_grad/= (rp_grad.max()+1e-8)

    rp_mask     = tight_mask(ref_patch)
    rp_edges    = _edges_from_mask(rp_mask)

    rp_mind     = _mind_lite(rp_g01)  # (H,W,K) float32 in [0,1]

    tg_g01      = _clahe01(tgt_img)
    gx = cv2.Sobel(tg_g01,cv2.CV_32F,1,0,3); gy = cv2.Sobel(tg_g01,cv2.CV_32F,0,1,3)
    tg_grad     = np.sqrt(gx*gx+gy*gy); tg_grad-=tg_grad.min(); tg_grad/= (tg_grad.max()+1e-8)
    tg_close    = edge_closeness_map(tgt_img)
    tg_mind     = _mind_lite(tg_g01)  # (H,W,K)

    Hp,Wp = rp_grad.shape[:2]; Ht,Wt = tg_grad.shape[:2]
    smax = min((Wt-1)/max(Wp,1),(Ht-1)/max(Hp,1))
    scales=[min(s,smax) for s in scales if min(int(Wp*s),Wt)>7 and min(int(Hp*s),Ht)>7]
    if not scales: 
        return None,0.0,{"reason":"no_valid_scale"}

    wa,wc,wm = weights
    best = (-1.0, None, {})

    for ang in rotations:
        rg = rotate_keep(rp_grad, ang).astype(np.float32)
        re = rotate_keep(rp_edges, ang).astype(np.float32)
        rm = rotate_any(rp_mind, ang).astype(np.float32)
        if rm.ndim == 2:
            rm = rm[...,None]
        if tg_mind.ndim == 2:
            tgM = tg_mind[...,None]
        else:
            tgM = tg_mind

        for s in scales:
            ww,hh = int(round(Wp*s)), int(round(Hp*s))
            rg_s = cv2.resize(rg,(ww,hh),interpolation=cv2.INTER_AREA)
            re_s = cv2.resize(re,(ww,hh),interpolation=cv2.INTER_NEAREST)
            rm_s = resize_any(rm,(ww,hh),interpolation=cv2.INTER_AREA)

            if rg_s.sum()==0 or re_s.sum()==0:
                continue

            try:
                resA = cv2.matchTemplate(tg_grad.astype(np.float32), rg_s, cv2.TM_CCOEFF_NORMED, mask=(re_s*255).astype(np.uint8))
            except cv2.error:
                resA = cv2.matchTemplate(tg_grad.astype(np.float32), (rg_s*re_s).astype(np.float32), cv2.TM_CCOEFF_NORMED)

            resC = cv2.matchTemplate(tg_close.astype(np.float32), re_s.astype(np.float32), cv2.TM_CCORR_NORMED)

            K = rm_s.shape[-1]
            acc = None
            for k in range(K):
                r = cv2.matchTemplate(tgM[...,k].astype(np.float32), rm_s[...,k].astype(np.float32), cv2.TM_CCOEFF_NORMED)
                acc = r if acc is None else (acc + r)
            resM = acc / float(K)

            res  = wa*resA + wc*resC + wm*resM
            _, mv, _, ml = cv2.minMaxLoc(res)
            x,y = ml; bb = BBox(int(x),int(y),int(x+ww),int(y+hh))
            if mv > best[0]:
                best = (float(mv), bb, {
                    "scale": float(s),
                    "angle": float(ang),
                    "scoreA": float(cv2.minMaxLoc(resA)[1]),
                    "scoreC": float(cv2.minMaxLoc(resC)[1]),
                    "scoreM": float(cv2.minMaxLoc(resM)[1]),
                })

    return best[1], float(max(0.0,best[0])), best[2]
