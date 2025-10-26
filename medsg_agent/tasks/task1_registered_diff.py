
from __future__ import annotations
from typing import Optional, Tuple, Dict
import numpy as np, cv2
from ..utils import BBox, to_gray01, bbox_from_mask, ssim_map

def get_subtracted_bbox(img_a: np.ndarray, img_b: np.ndarray,
                        method: str = "ssim", thresh: float = 0.25,
                        morph: int = 3) -> Tuple[Optional[BBox], float, Dict]:
    """
    Task 1: Registered Difference
    Assumes images are already aligned; finds region of change via SSIM or abs-diff.
    """
    A, B = to_gray01(img_a), to_gray01(img_b)
    diff = ssim_map(A, B) if method == "ssim" else np.abs(A - B)
    t = max(thresh, np.median(diff) + 2.5*np.std(diff))
    mask = (diff >= t).astype(np.uint8)
    if morph and morph > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    bb = bbox_from_mask(mask)
    conf = float(np.clip(diff[mask>0].mean() if bb is not None else 0.0, 0.0, 1.0))
    return bb, conf, {"method": method, "threshold": float(t)}
