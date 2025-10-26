
from __future__ import annotations
from typing import Optional, Tuple, Dict
import numpy as np, cv2
from ..utils import BBox, to_gray01, bbox_from_mask_clean, ssim_map, edge_closeness_map, phase_correlation_shift

def ecc_register(src: np.ndarray, dst: np.ndarray, warp_mode: int = cv2.MOTION_EUCLIDEAN,
                 iters: int = 100, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    s, d = to_gray01(src), to_gray01(dst)
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3,3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2,3, dtype=np.float32)
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

def get_diff_bbox_with_registration_robust(
    img_a: np.ndarray, img_b: np.ndarray, use_phasecorr_first: bool = True,
    edge_band_frac: float = 0.02, morph: int = 3, thresh: float = 0.25
) -> Tuple[Optional[BBox], float, Dict]:
    """
    Task 2: Non-Registered Difference with robust alignment (phase correlation + ECC).
    """
    A, B = img_a.copy(), img_b.copy()
    if use_phasecorr_first:
        ga, gb = to_gray01(A), to_gray01(B)
        dx, dy = phase_correlation_shift(ga, gb)
        M = np.array([[1,0,dx],[0,1,dy]], dtype=np.float32)
        A = cv2.warpAffine(A, M, (ga.shape[1], ga.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ws_gray, W = ecc_register(A, B, warp_mode=cv2.MOTION_EUCLIDEAN)

    Bg = to_gray01(B)
    diff = ssim_map(ws_gray, Bg)
    h, w = diff.shape
    band = max(2, int(edge_band_frac * min(h, w)))
    diff[:band, :] = 0
    diff[-band:, :] = 0
    diff[:, :band] = 0
    diff[:, -band:] = 0

    t = max(thresh, float(np.median(diff) + 2.5*np.std(diff)))
    mask = (diff >= t).astype(np.uint8)
    if morph and morph > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    bb = bbox_from_mask_clean(mask, min_area=8, border_margin=band)
    conf = float(np.clip(diff[mask>0].mean() if bb else 0.0, 0.0, 1.0))
    return bb, conf, {"threshold": float(t), "band": band, "warp": W.tolist()}
