
# from __future__ import annotations
# from typing import Optional, Tuple, Dict, List
# import numpy as np, cv2
# from ..utils import BBox, tight_mask, mask_edges, edge_closeness_map, rotate_keep, to_gray01

# def concept_match_bbox(ref_img: np.ndarray, ref_bbox: Optional[BBox], tgt_img: np.ndarray, method: str = "auto") -> Tuple[Optional[BBox], float, Dict]:
#     """
#     Task 5: Concept match via edge-chamfer style correlation.
#     If ref_bbox is None, a tight mask is computed on ref_img.
#     """
#     if ref_bbox is None:
#         m = tight_mask(ref_img)
#         cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not cnts:
#             h,w = m.shape; ref_bbox = BBox(0,0,w,h)
#         else:
#             c = max(cnts, key=cv2.contourArea)
#             x,y,w,h = cv2.boundingRect(c); ref_bbox = BBox(x,y,x+w,y+h)

#     x1,y1,x2,y2 = ref_bbox.x1, ref_bbox.y1, ref_bbox.x2, ref_bbox.y2
#     patch = ref_img[y1:y2, x1:x2]
#     mask = tight_mask(patch)
#     edges = mask_edges(mask)
#     if edges.sum()==0:
#         edges = cv2.Canny((to_gray01(patch)*255).astype(np.uint8), 30, 60)

#     C = edge_closeness_map(tgt_img)
#     Hp,Wp = edges.shape[:2]; Ht,Wt = C.shape[:2]
#     smax = min((Wt-1)/max(Wp,1), (Ht-1)/max(Hp,1))
#     if smax <= 0: return None, 0.0, {}
#     scales = sorted(set([smax, smax*0.95, smax*0.9, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]), reverse=True)
#     rots = [0, 10, -10, 20, -20]
#     best = (-1.0, None, {})
#     edges_f = edges.astype(np.float32)/255.0

#     for ang in rots:
#         er = rotate_keep(edges_f, ang)
#         for s in scales:
#             s = float(min(s, smax))
#             ww = int(max(8, round(Wp*s))); hh = int(max(8, round(Hp*s)))
#             if ww>Wt or hh>Ht: continue
#             ker = cv2.resize(er, (ww, hh), interpolation=cv2.INTER_NEAREST)
#             if ker.sum() < 1.0: continue
#             res = cv2.matchTemplate(C, ker, cv2.TM_CCORR_NORMED)
#             _, mv, _, ml = cv2.minMaxLoc(res)
#             x,y = ml; bb = BBox(int(x), int(y), int(x+ww), int(y+hh))
#             if mv > best[0]:
#                 best = (float(mv), bb, {"scale": s, "angle": ang})
#     return best[1], float(max(0.0, best[0])), {"mode":"edge_chamfer_ccorr", **best[2]}


import numpy as np, cv2, matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List
from medsg_agent.utils import BBox, read_image, to_gray01, tight_mask, rotate_keep, edge_closeness_map, draw_box, list_to_bbox, compute_iou

def feature_extractor():
    if hasattr(cv2,"SIFT_create"):
        try: return cv2.SIFT_create(), "FLANN_F32"
        except: pass
    if hasattr(cv2,"KAZE_create"):
        try: return cv2.KAZE_create(), "FLANN_F32"
        except: pass
    return cv2.ORB_create(nfeatures=4000,scaleFactor=1.2,nlevels=8), "FLANN_BIN"

def flann_for_mode(mode):
    if mode=="FLANN_F32":
        index_params=dict(algorithm=1,trees=5)
        search_params=dict(checks=64)
        return cv2.FlannBasedMatcher(index_params,search_params)
    else:
        index_params=dict(algorithm=6,table_number=6,key_size=12,multi_probe_level=1)
        search_params=dict(checks=128)
        return cv2.FlannBasedMatcher(index_params,search_params)

def detect_desc(feat, img):
    g=(to_gray01(img)*255).astype(np.uint8)
    kp,des=feat.detectAndCompute(g,None)
    return kp,des

def good_knn_matches(des1, des2, flann, ratio=0.75):
    if des1 is None or des2 is None: return []
    m=flann.knnMatch(des1,des2,k=2)
    good=[]
    for a in m:
        if len(a)<2: continue
        m1,m2=a
        if m1.distance<ratio*m2.distance: good.append(m1)
    return good

def homography_from_matches(kp1, kp2, matches):
    if len(matches)<4: return None,None,None
    src=np.float32([kp1[m.queryIdx].pt for m in matches])
    dst=np.float32([kp2[m.trainIdx].pt for m in matches])
    H,mask=cv2.findHomography(src,dst,cv2.RANSAC,4.0,maxIters=5000,confidence=0.995)
    if H is None or mask is None: return None,None,None
    inl=mask.ravel().astype(bool)
    if inl.sum()<4: return None,None,None
    src_in=src[inl]
    dst_in=dst[inl]
    src_h=np.concatenate([src_in,np.ones((src_in.shape[0],1))],1)
    proj=(src_h@H.T)
    proj=proj[:,:2]/proj[:,2:3]
    err=np.linalg.norm(proj-dst_in,axis=1)
    med=float(np.median(err))
    return H, int(inl.sum()), med

def project_bbox(H, bbox, shape):
    pts=np.float32([[bbox.x1,bbox.y1],[bbox.x2,bbox.y1],[bbox.x2,bbox.y2],[bbox.x1,bbox.y2]]).reshape(-1,1,2)
    dst=cv2.perspectiveTransform(pts,H).reshape(-1,2)
    x1=int(np.clip(np.min(dst[:,0]),0,shape[1]-1))
    y1=int(np.clip(np.min(dst[:,1]),0,shape[0]-1))
    x2=int(np.clip(np.max(dst[:,0]),0,shape[1]-1))
    y2=int(np.clip(np.max(dst[:,1]),0,shape[0]-1))
    if x2<=x1 or y2<=y1: return None
    return BBox(x1,y1,x2,y2)

def auto_ref_bbox(img, pad=4):
    m=tight_mask(img)
    cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h,w=img.shape[:2]; return BBox(0,0,w,h)
    c=max(cnts,key=cv2.contourArea)
    x,y,w,h=cv2.boundingRect(c)
    H,W=img.shape[:2]
    return BBox(max(0,x-pad),max(0,y-pad),min(W,x+w+pad),min(H,y+h+pad))

def concept_match_bbox(ref_img: np.ndarray, ref_bbox: Optional[BBox], tgt_img: np.ndarray, ratio=0.75) -> Tuple[Optional[BBox], float, Dict]:
    if ref_bbox is None: ref_bbox=auto_ref_bbox(ref_img)
    feat,mode=feature_extractor()
    kp1,des1=detect_desc(feat,ref_img)
    kp2,des2=detect_desc(feat,tgt_img)
    sel=[]
    for i,k in enumerate(kp1):
        x,y=k.pt
        if ref_bbox.x1<=x<=ref_bbox.x2 and ref_bbox.y1<=y<=ref_bbox.y2: sel.append(i)
    if not sel: return None,0.0,{"reason":"no_kp_in_ref_bbox"}
    kp1_s=[kp1[i] for i in sel]
    des1_s=des1[sel] if des1 is not None else None
    flann=flann_for_mode(mode)
    good=good_knn_matches(des1_s,des2,flann,ratio=ratio)
    H,n_inl,med_err=homography_from_matches(kp1_s,kp2,good)
    if H is None: return None,0.0,{"reason":"no_homography"}
    bb=project_bbox(H,ref_bbox,tgt_img.shape[:2])
    if bb is None: return None,0.0,{"reason":"bad_projection"}
    score=float(n_inl)/(med_err+1.0)
    conf=1.0-np.exp(-score/40.0)
    return bb,float(conf),{"mode":mode,"inliers":int(n_inl),"median_err":float(med_err)}

def grad_fields(x):
    g=to_gray01(x)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,3)
    gy=cv2.Sobel(g,cv2.CV_32F,0,1,3)
    mag=np.sqrt(gx*gx+gy*gy)
    eps=1e-6
    gx=(gx-np.mean(gx))/(np.std(gx)+eps)
    gy=(gy-np.mean(gy))/(np.std(gy)+eps)
    mag=(mag-mag.min())/(mag.max()-mag.min()+eps)
    return gx.astype(np.float32),gy.astype(np.float32),mag.astype(np.float32)

def strong_edge_bbox_fb(x, q=90, pad=6):
    gx,gy,mag=grad_fields(x)
    thr=np.percentile(mag, q)
    m=(mag>=thr).astype(np.uint8)*255
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,k)
    cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        u=(to_gray01(x)*255).astype(np.uint8)
        _,m2=cv2.threshold(u,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cnts2,_=cv2.findContours(m2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not cnts2:
            h,w=x.shape[:2]; return BBox(0,0,w,h)
        c=max(cnts2,key=cv2.contourArea)
        x0,y0,w0,h0=cv2.boundingRect(c)
    else:
        c=max(cnts,key=cv2.contourArea)
        x0,y0,w0,h0=cv2.boundingRect(c)
    h,w=x.shape[:2]
    x1=max(0,x0-pad); y1=max(0,y0-pad); x2=min(w,x0+w0+pad); y2=min(h,y0+h0+pad)
    return BBox(x1,y1,x2,y2)

def edge_kernel_from_patch(patch):
    g=(to_gray01(patch)*255).astype(np.uint8)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    e=cv2.morphologyEx(g,cv2.MORPH_GRADIENT,k)
    e[e>0]=255
    return (e.astype(np.float32)/255.0)

def closeness_map(x):
    return edge_closeness_map(x)

def concept_match_bbox_fallback(ref_img: np.ndarray, ref_bbox: Optional[BBox], tgt_img: np.ndarray, scales: List[float]=None, rotations: List[float]=None, alpha: float=0.85) -> Tuple[Optional[BBox], float, Dict]:
    if scales is None: scales=[1.2,1.1,1.05,1.0,0.95,0.9,0.85,0.8]
    if rotations is None: rotations=[0,5,-5,10,-10]
    if ref_bbox is None: ref_bbox=strong_edge_bbox_fb(ref_img, q=90, pad=6)
    x1,y1,x2,y2=ref_bbox.x1,ref_bbox.y1,ref_bbox.x2,ref_bbox.y2
    x1=max(0,x1); y1=max(0,y1); x2=min(ref_img.shape[1],x2); y2=min(ref_img.shape[0],y2)
    if x2-x1<3 or y2-y1<3: return None,0.0,{"reason":"invalid_ref_bbox"}
    patch=ref_img[y1:y2,x1:x2]
    pgx,pgy,pmag=grad_fields(patch)
    tgx,tgy,tmag=grad_fields(tgt_img)
    T=np.dstack([tgx,tgy,tmag]).astype(np.float32)
    Pe=edge_kernel_from_patch(patch)
    C=closeness_map(tgt_img)
    Hp,Wp=pmag.shape; Ht,Wt=T.shape[:2]
    smax=min((Wt-1)/max(Wp,1),(Ht-1)/max(Hp,1))
    scales=[min(s,smax) for s in scales if min(int(Wp*s),Wt)>7 and min(int(Hp*s),Ht)>7]
    if not scales: return None,0.0,{"reason":"no_valid_scale"}
    best=(-1.0,None,{})
    for ang in rotations:
        pgx_r=rotate_keep(pgx,ang); pgy_r=rotate_keep(pgy,ang); pmag_r=rotate_keep(pmag,ang); Pe_r=rotate_keep(Pe,ang)
        for s in scales:
            ww,hh=int(round(Wp*s)),int(round(Hp*s))
            pgx_s=cv2.resize(pgx_r,(ww,hh),interpolation=cv2.INTER_AREA)
            pgy_s=cv2.resize(pgy_r,(ww,hh),interpolation=cv2.INTER_AREA)
            pmg_s=cv2.resize(pmag_r,(ww,hh),interpolation=cv2.INTER_AREA)
            Pe_s=cv2.resize(Pe_r,(ww,hh),interpolation=cv2.INTER_NEAREST)
            if pmg_s.sum()<=0 or Pe_s.sum()<=0: continue
            P=np.dstack([pgx_s,pgy_s,pmg_s]).astype(np.float32)
            P=P*(Pe_s[:,:,None])
            try:
                res1=cv2.matchTemplate(T,P,cv2.TM_CCOEFF_NORMED)
            except cv2.error:
                t8=(np.clip((T - T.min())/(T.max()-T.min()+1e-6)*255,0,255).astype(np.uint8))
                p8=(np.clip((P - P.min())/(P.max()-P.min()+1e-6)*255,0,255).astype(np.uint8))
                res1=cv2.matchTemplate(t8,p8,cv2.TM_CCOEFF_NORMED)
            res2=cv2.matchTemplate(C,Pe_s,cv2.TM_CCORR_NORMED)
            res=alpha*res1+(1.0-alpha)*res2
            res=np.nan_to_num(res,nan=-1.0,posinf=-1.0,neginf=-1.0)
            _,mv,_,ml=cv2.minMaxLoc(res)
            x,y=ml
            bb=BBox(int(x),int(y),int(x+ww),int(y+hh))
            if mv>best[0]:
                best=(float(mv),bb,{"scale":float(s),"angle":float(ang),"alpha":float(alpha)})
    return best[1],float(max(0.0,best[0])),best[2]

def concept_match_bbox_unified(ref_img: np.ndarray, ref_bbox: Optional[BBox], tgt_img: np.ndarray, ratio=0.75, scales: List[float]=None, rotations: List[float]=None, alpha: float=0.85) -> Tuple[Optional[BBox], float, Dict]:
    bb,conf,dbg=concept_match_bbox(ref_img,ref_bbox,tgt_img,ratio=ratio)
    if bb is not None:
        return bb,conf,{"stage":"feature",**dbg}
    bb2,cf2,dbg2=concept_match_bbox_fallback(ref_img,ref_bbox,tgt_img,scales=scales,rotations=rotations,alpha=alpha)
    return bb2,cf2,{"stage":"fallback",**dbg2}

def run(image_concept_path: str, image_target_path: str, ref_bbox: Optional[List[int]] = None, ratio: float = 0.72, scales: Optional[List[float]] = None, rotations: Optional[List[float]] = None, alpha: float = 0.85) -> Dict:
    ref=read_image(image_concept_path)
    tgt=read_image(image_target_path)
    rb = list_to_bbox(ref_bbox) if ref_bbox is not None else None
    pred, conf, dbg = concept_match_bbox_unified(ref, rb, tgt, ratio=ratio, scales=scales if scales is not None else [1.25,1.2,1.1,1.0,0.95,0.9], rotations=rotations if rotations is not None else [0,5,-5,10,-10], alpha=alpha)
    out = {"bbox": None if pred is None else {"x1": int(pred.x1), "y1": int(pred.y1), "x2": int(pred.x2), "y2": int(pred.y2)}, "confidence": float(conf), "details": dbg}
    return out

__all__ = ["run","concept_match_bbox_unified","concept_match_bbox"]
