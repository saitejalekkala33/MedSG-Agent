from __future__ import annotations
from typing import Optional, Tuple, Dict, List
import numpy as np, cv2, warnings
from ..utils import BBox, ensure_same_size, to_gray01

try:
    from skimage.metrics import structural_similarity as _ssim
except Exception:
    _ssim = None

def _build_diff_map_registered(A, B, method: str):
    ga, gb = to_gray01(A), to_gray01(B)
    if method == "ssim" and _ssim is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, ssim_map = _ssim(ga, gb, full=True)
        diff = 1.0 - ssim_map
    else:
        diff = np.abs(ga - gb)
    return diff

def _candidate_thresholds(diff: np.ndarray, base: float) -> List[float]:
    base = float(np.clip(base, 1e-3, 0.95))
    med = float(np.median(diff)); sd = float(np.std(diff))
    grids = [base*0.5, base*0.75, base, base*1.25, base*1.5, med + 1.5*sd, med + 2.0*sd, med + 2.5*sd, med + 3.0*sd]
    grids = [float(np.clip(t, 1e-3, 0.99)) for t in grids]
    return sorted(set(round(t, 5) for t in grids))

def _mask_postprocess(diff: np.ndarray, t: float, morph: int) -> np.ndarray:
    mask = (diff >= t).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask

def _largest_component(mask: np.ndarray) -> np.ndarray:
    m = (mask>0).astype(np.uint8)*255
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return np.zeros_like(mask, dtype=np.uint8)
    c = max(cnts, key=cv2.contourArea)
    out = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(out, [c], -1, 1, thickness=-1)
    return out

def _compactness(contour) -> float:
    A = cv2.contourArea(contour)
    if A <= 1: return 0.0
    P = cv2.arcLength(contour, True) + 1e-8
    return float(max(0.0, min(1.0, (4.0*np.pi*A)/(P*P))))

def _mask_quality_no_gt(masks: List[np.ndarray], k: int, diff: np.ndarray, band: int) -> Tuple[float, Optional[BBox], Dict]:
    mk = (masks[k] > 0)
    if mk.sum() == 0:
        return -1.0, None, {"reason":"empty"}
    H, W = diff.shape
    inside_vals = diff[mk]
    outside_mask = ~mk
    if band > 0:
        outside_mask[:band, :] = False
        outside_mask[-band:, :] = False
        outside_mask[:, :band] = False
        outside_mask[:, -band:] = False
    outside_vals = diff[outside_mask]
    if outside_vals.size == 0:
        outside_vals = diff[~mk]
    mean_in  = float(inside_vals.mean())
    mean_out = float(outside_vals.mean())
    std_out  = float(outside_vals.std() + 1e-6)
    contrast_z = (mean_in - mean_out) / std_out
    def iou(a,b):
        inter = np.logical_and(a>0, b>0).sum()
        union = np.logical_or(a>0, b>0).sum()
        return float(inter)/float(union+1e-8)
    J_prev = iou(masks[k], masks[k-1]) if k-1 >= 0 else 0.0
    J_next = iou(masks[k], masks[k+1]) if k+1 < len(masks) else 0.0
    stability = 0.5*(J_prev + J_next)
    cnts,_ = cv2.findContours((mk>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    comp = _compactness(c)
    x,y,w,h = cv2.boundingRect(c)
    bb = BBox(int(x),int(y),int(x+w),int(y+h))
    area = float(w*h)
    border_touch = int(x<=0 or y<=0 or (x+w)>=W-1 or (y+h)>=H-1)
    score = (0.50*contrast_z) + (0.30*stability) + (0.15*comp) + (0.08*np.log1p(area)) - (0.20*border_touch)
    details = {"contrast_z": contrast_z, "stability": stability, "compactness": comp, "area": area, "border_touch": border_touch}
    return float(score), bb, details

def _choose_best_threshold_no_gt(diff: np.ndarray, morph: int, thresholds: List[float], band: int) -> Tuple[Optional[BBox], float, Dict]:
    masks = [_largest_component(_mask_postprocess(diff, t, morph)) for t in thresholds]
    best = {"score": -1e9, "idx": None, "bbox": None, "details": None}
    for k in range(len(thresholds)):
        score, bb, det = _mask_quality_no_gt(masks, k, diff, band)
        if bb is None:
            continue
        if score > best["score"]:
            best.update({"score": score, "idx": k, "bbox": bb, "details": det})
    if best["bbox"] is None:
        return None, 0.0, {"reason":"no_valid_mask"}
    conf = float(max(0.0, min(1.0, best["details"]["contrast_z"]/3.0)))
    out_details = {"chosen_threshold": float(thresholds[best["idx"]]), "score": float(best["score"]), **best["details"]}
    return best["bbox"], conf, out_details

def get_subtracted_bbox_dynamic(img_a: np.ndarray, img_b: np.ndarray, method: str = "ssim", base_thresh: float = 0.05, morph: int = 3) -> Tuple[Optional[BBox], float, Dict]:
    A, B = ensure_same_size(img_a, img_b)
    diff = _build_diff_map_registered(A, B, method=method)
    cands = _candidate_thresholds(diff, base_thresh)
    bb, conf, det = _choose_best_threshold_no_gt(diff, morph, cands, band=0)
    det = {"method": method, "chosen_threshold": det.get("chosen_threshold", None), "search_thresholds": cands, **det}
    return bb, conf, det
