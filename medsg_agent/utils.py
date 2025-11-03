from __future__ import annotations
import os, re, glob, math, warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2

try:
    from skimage.metrics import structural_similarity as _ssim
except Exception:
    _ssim = None

@dataclass
class BBox:
    x1: int; y1: int; x2: int; y2: int
    def to_dict(self) -> Dict[str, int]:
        return {"x1": int(self.x1), "y1": int(self.y1), "x2": int(self.x2), "y2": int(self.y2)}

def clip_bbox(b: BBox, h: int, w: int) -> BBox:
    return BBox(max(0, b.x1), max(0, b.y1), min(w-1, b.x2), min(h-1, b.y2))

def compute_iou(a: BBox, b: BBox) -> float:
    xi1, yi1 = max(a.x1,b.x1), max(a.y1,b.y1)
    xi2, yi2 = min(a.x2,b.x2), min(a.y2,b.y2)
    if xi2 <= xi1 or yi2 <= yi1: return 0.0
    inter = (xi2-xi1)*(yi2-yi1)
    area_a = (a.x2-a.x1)*(a.y2-a.y1)
    area_b = (b.x2-b.x1)*(b.y2-b.y1)
    return float(inter) / float(area_a + area_b - inter + 1e-8)

def to_gray01(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img.copy()
    g = g.astype(np.float32)
    mn, mx = np.percentile(g, 1), np.percentile(g, 99)
    if mx <= mn: mx = mn + 1.0
    g = (np.clip(g, mn, mx) - mn) / (mx - mn)
    return g

def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def draw_box(img_bgr: np.ndarray, bbox: Optional[BBox], color=(0,255,0), thickness=2, label: Optional[str]=None) -> np.ndarray:
    out = img_bgr.copy()
    if bbox is not None:
        cv2.rectangle(out, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, thickness)
        if label:
            cv2.putText(out, label, (bbox.x1, max(0, bbox.y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out

def list_to_bbox(b: List[int]) -> BBox:
    return BBox(int(b[0]), int(b[1]), int(b[2]), int(b[3]))

def bbox_from_mask(mask: np.ndarray, min_area: int = 8) -> Optional[BBox]:
    m = (mask>0).astype(np.uint8)*255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area: return None
    x,y,w,h = cv2.boundingRect(cnt)
    return BBox(x,y,x+w,y+h)

def bbox_from_mask_clean(mask: np.ndarray, min_area: int = 8, border_margin: int = 4) -> Optional[BBox]:
    m = (mask > 0).astype(np.uint8) * 255
    h, w = m.shape
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None; best_area = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x,y,ww,hh = cv2.boundingRect(c)
        if x <= border_margin or y <= border_margin or (x+ww) >= (w-1-border_margin) or (y+hh) >= (h-1-border_margin):
            continue
        if area > best_area:
            best_area = area
            best = BBox(x,y,x+ww,y+hh)
    return best

def tight_mask(img: np.ndarray) -> np.ndarray:
    g = (to_gray01(img)*255).astype(np.uint8)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m

def mask_edges(mask: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    dil = cv2.dilate(mask, k, 1)
    ero = cv2.erode(mask, k, 1)
    e = cv2.subtract(dil, ero)
    e[e>0]=255
    return e

def rotate_keep(img: np.ndarray, ang: float) -> np.ndarray:
    if ang % 360 == 0: return img
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), ang, 1.0)
    return cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def rotate_any(arr: np.ndarray, ang: float) -> np.ndarray:
    if arr.ndim == 2:
        return rotate_keep(arr, ang)
    ch = arr.shape[2]
    out = [rotate_keep(arr[:,:,c], ang) for c in range(ch)]
    return np.stack(out, axis=2)

def resize_any(arr: np.ndarray, size: Tuple[int,int], interpolation: int) -> np.ndarray:
    if arr.ndim == 2:
        return cv2.resize(arr, size, interpolation=interpolation)
    ch = arr.shape[2]
    out = [cv2.resize(arr[:,:,c], size, interpolation=interpolation) for c in range(ch)]
    return np.stack(out, axis=2)

def edge_closeness_map(img: np.ndarray) -> np.ndarray:
    g = (to_gray01(img)*255).astype(np.uint8)
    edges = cv2.Canny(g, 30, 60)
    inv = cv2.bitwise_not(edges)
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    s = max(1.0, 0.015*max(dt.shape))
    closeness = np.exp(-(dt/s)).astype(np.float32)
    return closeness

def phase_correlation_shift(a_gray01: np.ndarray, b_gray01: np.ndarray) -> Tuple[float,float]:
    win = cv2.createHanningWindow((a_gray01.shape[1], a_gray01.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate((a_gray01*win).astype(np.float32), (b_gray01*win).astype(np.float32))
    return float(dx), float(dy)

def ssim_map(a_gray01: np.ndarray, b_gray01: np.ndarray) -> np.ndarray:
    if _ssim is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, smap = _ssim(a_gray01, b_gray01, full=True)
        return 1.0 - smap
    return np.abs(a_gray01 - b_gray01)

def smart_path(path_str: str, base_dir: str = ".", fallback_dir: Optional[str] = None) -> str:
    if os.path.exists(os.path.join(base_dir, path_str)):
        return os.path.join(base_dir, path_str)
    if fallback_dir:
        p = os.path.join(base_dir, fallback_dir, os.path.basename(path_str))
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(base_dir, "**", os.path.basename(path_str)), recursive=True)
    return hits[0] if hits else path_str

def parse_two_image_indices(question: str, n_images: int) -> Tuple[int, int]:
    q = question.lower()
    ord_map = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4, "sixth": 5, "seventh": 6, "eighth": 7, "ninth": 8, "tenth": 9, "last": -1}
    ord_hits = re.findall(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)\b", q)
    indices: List[int] = []
    for w in ord_hits:
        idx = ord_map[w]
        if idx == -1:
            idx = max(0, n_images - 1)
        if 0 <= idx < n_images and idx not in indices:
            indices.append(idx)
    num_hits = re.findall(r"\b(?:image|img|images)\s*(\d+)\b", q)
    for num in num_hits:
        k = int(num) - 1
        if 0 <= k < n_images and k not in indices:
            indices.append(k)
    if len(indices) < 2:
        loose_nums = re.findall(r"\b(\d+)(?:st|nd|rd|th)?\b", q)
        for num in loose_nums:
            k = int(num) - 1
            if 0 <= k < n_images and k not in indices:
                indices.append(k)
            if len(indices) >= 2:
                break
    if not indices:
        return 0, min(1, n_images - 1)
    if len(indices) == 1:
        return indices[0], (indices[0] + 1) % n_images
    return indices[0], indices[1]

def parse_ref_bbox_from_question(question: str) -> Optional[BBox]:
    s = question
    m = re.search(r"<\|box_start\|>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*<\|box_end\|>", s)
    if not m:
        return None
    x1,y1,x2,y2 = map(int, m.groups())
    return BBox(x1,y1,x2,y2)

def parse_region_index_from_question(question: str, default: int = 1) -> int:
    q = question.lower()
    ord_map = {"first":1,"second":2,"third":3,"fourth":4,"fifth":5}
    for w,v in ord_map.items():
        if re.search(rf"\b{w}\b", q):
            return v
    m = re.search(r"\bregion\s*(\d+)\b", q)
    if m:
        return max(1, int(m.group(1)))
    return default

def ensure_same_size(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if a.shape[:2] == b.shape[:2]:
        return a, b
    H, W = b.shape[:2]
    return cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR), b

def ecc_register(src: np.ndarray, dst: np.ndarray, warp_mode: int = cv2.MOTION_EUCLIDEAN, iters: int = 100, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    s, d = to_gray01(src), to_gray01(dst)
    warp_matrix = np.eye(3,3, dtype=np.float32) if warp_mode == cv2.MOTION_HOMOGRAPHY else np.eye(2,3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    try:
        _, warp_matrix = cv2.findTransformECC(d, s, warp_matrix, warp_mode, criteria, None, 5)
    except cv2.error:
        pass
    h, w = d.shape
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        ws = cv2.warpPerspective(s, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    else:
        ws = cv2.warpAffine(s, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    return ws, warp_matrix

def is_registered_task(img_a: np.ndarray, img_b: np.ndarray, shift_tol_px: Optional[float] = None, ssim_gain_tol: float = 0.02) -> Tuple[bool, Dict]:
    A, B = ensure_same_size(img_a, img_b)
    ga, gb = to_gray01(A), to_gray01(B)
    min_side = min(ga.shape)
    dyn_shift_tol = max(2.0, 0.01 * float(min_side)) if shift_tol_px is None else float(shift_tol_px)
    try:
        dx, dy = phase_correlation_shift(ga, gb)
        shift_mag = math.hypot(dx, dy)
    except cv2.error:
        shift_mag = 0.0
    if _ssim is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ssim_before = float(_ssim(ga, gb))
        ws, _ = ecc_register(A, B, warp_mode=cv2.MOTION_EUCLIDEAN)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ssim_after = float(_ssim(ws, gb))
        ssim_gain = max(0.0, ssim_after - ssim_before)
        decision = (shift_mag <= dyn_shift_tol) and (ssim_gain <= ssim_gain_tol)
        metrics = {"shift_px": shift_mag, "ssim_before": ssim_before, "ssim_after": ssim_after, "ssim_gain": ssim_gain, "shift_tol_px": dyn_shift_tol, "ssim_gain_tol": ssim_gain_tol}
    else:
        mae_before = float(np.mean(np.abs(ga - gb)))
        ws, _ = ecc_register(A, B, warp_mode=cv2.MOTION_EUCLIDEAN)
        mae_after = float(np.mean(np.abs(ws - gb)))
        mae_gain = max(0.0, mae_before - mae_after)
        decision = (shift_mag <= dyn_shift_tol) and (mae_gain <= 0.01)
        metrics = {"shift_px": shift_mag, "mae_before": mae_before, "mae_after": mae_after, "mae_gain": mae_gain, "shift_tol_px": dyn_shift_tol, "mae_gain_tol": 0.01}
    return bool(decision), metrics
