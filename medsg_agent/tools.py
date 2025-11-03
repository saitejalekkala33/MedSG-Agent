from __future__ import annotations
import json
from typing import Optional, List, Tuple, Dict
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from medsg_agent.utils import (
    read_image,
    BBox,
    is_registered_task,
)

from medsg_agent.tasks.task1_registered_diff import get_subtracted_bbox
from medsg_agent.tasks.task2_nonregistered_diff import get_diff_bbox_with_registration_robust
from medsg_agent.tasks.task3_multi_view import multi_view_grounding_bbox
from medsg_agent.tasks.task5_concept_match import concept_match_bbox
from medsg_agent.tasks.task6_patch_grounding import patch_grounding_bbox_robust
from medsg_agent.tasks.task7_crossmodal import crossmodal_grounding_bbox

try:
    from medsg_agent.tasks.task1_registered_diff_dynamic import get_subtracted_bbox_dynamic as _rdg_dynamic
except Exception:
    _rdg_dynamic = None

try:
    from medsg_agent.tasks.task2_nonregistered_diff_dynamic import get_diff_bbox_with_registration_dynamic as _nrdg_dynamic
except Exception:
    _nrdg_dynamic = None


class TwoImageArgs(BaseModel):
    image_a: str = Field(..., description="Path to first image (reference A).")
    image_b: str = Field(..., description="Path to second image (reference B).")
    method: Optional[str] = Field(default="ssim", description="Difference method: 'ssim' or 'abs'.")
    thresh: Optional[float] = Field(default=0.25, description="Base threshold for difference.")
    morph: Optional[int] = Field(default=3, description="Morph kernel size (odd integer >=1).")

class NonRegArgs(BaseModel):
    image_a: str
    image_b: str
    use_phasecorr_first: Optional[bool] = True
    edge_band_frac: Optional[float] = 0.02
    morph: Optional[int] = 3
    thresh: Optional[float] = 0.25

class MultiViewArgs(BaseModel):
    ref_image: str
    tgt_image: str
    ref_bbox: List[int] = Field(..., description="[x1,y1,x2,y2] on ref_image")
    method: Optional[str] = Field(default="auto", description="'orb'|'ncc'|'auto'")

class ConceptArgs(BaseModel):
    ref_image: str
    tgt_image: str
    ref_bbox: Optional[List[int]] = Field(default=None, description="Optional [x1,y1,x2,y2]; if omitted, auto-detect.")

class PatchArgs(BaseModel):
    source_image: str
    patch_image: str
    scales: Optional[List[float]] = Field(default=None)
    rotations: Optional[List[float]] = Field(default=None)
    alpha: Optional[float] = Field(default=0.6)

class CrossmodalArgs(BaseModel):
    ref_image: str
    tgt_image: str
    ref_bbox: Optional[List[int]] = None
    scales: Optional[List[float]] = None
    rotations: Optional[List[float]] = None
    weights: Optional[List[float]] = Field(default=None, description="[wa, wc, wm]")

class IsRegisteredArgs(BaseModel):
    image_a: str
    image_b: str
    shift_tol_px: Optional[float] = Field(default=None)
    ssim_gain_tol: Optional[float] = Field(default=0.02)

class AutoDiffArgs(BaseModel):
    image_a: str
    image_b: str
    method: Optional[str] = Field(default="ssim")
    morph: Optional[int] = Field(default=3)
    dynamic: Optional[bool] = Field(default=True)
    base_thresh_registered: Optional[float] = Field(default=0.05)
    use_phasecorr_first: Optional[bool] = Field(default=True)
    edge_band_frac: Optional[float] = Field(default=0.02)
    base_thresh_nonregistered: Optional[float] = Field(default=0.25)


def _fmt_out(bbox, conf, details):
    out = {"bbox": None if bbox is None else BBox(bbox.x1, bbox.y1, bbox.x2, bbox.y2).to_dict(), "confidence": float(conf), "details": details}
    return json.dumps(out)


def multi_view_grounding_bbox_tool(ref_image: str, tgt_image: str, ref_bbox: List[int], method: str = "auto") -> str:
    R = read_image(ref_image); T = read_image(tgt_image)
    rb = BBox(*ref_bbox)
    bbox, conf, det = multi_view_grounding_bbox(R, T, rb, method=method)
    det = {**det, "tool": "multi_view_grounding_bbox"}
    return _fmt_out(bbox, conf, det)

def concept_match_bbox_tool(ref_image: str, tgt_image: str, ref_bbox: Optional[List[int]] = None) -> str:
    R = read_image(ref_image); T = read_image(tgt_image)
    rb = BBox(*ref_bbox) if ref_bbox else None
    bbox, conf, det = concept_match_bbox(R, rb, T, method="auto")
    det = {**det, "tool": "concept_match_bbox"}
    return _fmt_out(bbox, conf, det)

def patch_grounding_bbox_tool(source_image: str, patch_image: str, scales: Optional[List[float]] = None, rotations: Optional[List[float]] = None, alpha: float = 0.6) -> str:
    S = read_image(source_image); P = read_image(patch_image)
    bbox, conf, det = patch_grounding_bbox_robust(P, S, scales=scales, rotations=rotations, alpha=alpha)
    det = {**det, "tool": "patch_grounding_bbox"}
    return _fmt_out(bbox, conf, det)

def crossmodal_grounding_bbox_tool(ref_image: str, tgt_image: str, ref_bbox: Optional[List[int]] = None, scales: Optional[List[float]] = None, rotations: Optional[List[float]] = None, weights: Optional[List[float]] = None) -> str:
    R = read_image(ref_image); T = read_image(tgt_image)
    rb = BBox(*ref_bbox) if ref_bbox else None
    wtuple = tuple(weights) if weights else (0.45, 0.35, 0.20)
    bbox, conf, det = crossmodal_grounding_bbox(R, rb, T, scales=scales, rotations=rotations, weights=wtuple)
    det = {**det, "tool": "crossmodal_grounding_bbox"}
    return _fmt_out(bbox, conf, det)

def is_registered_task_tool(image_a: str, image_b: str, shift_tol_px: Optional[float] = None, ssim_gain_tol: float = 0.02) -> str:
    A = read_image(image_a); B = read_image(image_b)
    decision, metrics = is_registered_task(A, B, shift_tol_px=shift_tol_px, ssim_gain_tol=ssim_gain_tol)
    return json.dumps({"registered": bool(decision), "metrics": metrics, "tool": "is_registered_task"})

def _call_registered(A, B, method: str, morph: int, base_thresh: float, dynamic: bool):
    if dynamic and _rdg_dynamic is not None:
        return _rdg_dynamic(A, B, method=method, base_thresh=base_thresh, morph=morph)
    return get_subtracted_bbox(A, B, method=method, thresh=base_thresh, morph=morph)

def _call_nonregistered(A, B, use_phasecorr_first: bool, edge_band_frac: float, morph: int, base_thresh: float, dynamic: bool):
    if dynamic and _nrdg_dynamic is not None:
        return _nrdg_dynamic(A, B, use_phasecorr_first=use_phasecorr_first, edge_band_frac=edge_band_frac, morph=morph, base_thresh=base_thresh)
    return get_diff_bbox_with_registration_robust(A, B, use_phasecorr_first=use_phasecorr_first, edge_band_frac=edge_band_frac, morph=morph, thresh=base_thresh)

def auto_diff_bbox_tool(image_a: str, image_b: str, method: str = "ssim", morph: int = 3, dynamic: bool = True, base_thresh_registered: float = 0.05, use_phasecorr_first: bool = True, edge_band_frac: float = 0.02, base_thresh_nonregistered: float = 0.25) -> str:
    A = read_image(image_a); B = read_image(image_b)
    decision, metrics = is_registered_task(A, B, shift_tol_px=None, ssim_gain_tol=0.02)
    if decision:
        bbox, conf, det = _call_registered(A, B, method=method, morph=morph, base_thresh=base_thresh_registered, dynamic=dynamic)
        det = {**det, "mode": "registered", "routings_metrics": metrics, "tool": "auto_diff_bbox"}
        return _fmt_out(bbox, conf, det)
    bbox, conf, det = _call_nonregistered(A, B, use_phasecorr_first=use_phasecorr_first, edge_band_frac=edge_band_frac, morph=morph, base_thresh=base_thresh_nonregistered, dynamic=dynamic)
    det = {**det, "mode": "nonregistered", "routings_metrics": metrics, "tool": "auto_diff_bbox"}
    return _fmt_out(bbox, conf, det)


def get_tools():
    return [
        StructuredTool.from_function(
            func=is_registered_task_tool,
            name="is_registered_task",
            description="Decide if two images are already registered. Returns {'registered': bool, 'metrics': {...}}.",
            args_schema=IsRegisteredArgs,
            return_direct=True,
        ),
        StructuredTool.from_function(
            func=auto_diff_bbox_tool,
            name="auto_diff_bbox",
            description="Auto-select Registered vs Non-registered diff. Calls is_registered_task, then runs the appropriate diff and returns bbox JSON with mode.",
            args_schema=AutoDiffArgs,
            return_direct=True,
        ),
        StructuredTool.from_function(
            func=multi_view_grounding_bbox_tool,
            name="multi_view_grounding_bbox",
            description="Given a reference image + bbox and a target view image of the same subject, locate corresponding bbox in target.",
            args_schema=MultiViewArgs,
            return_direct=True,
        ),
        StructuredTool.from_function(
            func=concept_match_bbox_tool,
            name="concept_match_bbox",
            description="Given a concept example image (with or without bbox) and a target image, locate the concept in the target.",
            args_schema=ConceptArgs,
            return_direct=True,
        ),
        StructuredTool.from_function(
            func=patch_grounding_bbox_tool,
            name="patch_grounding_bbox",
            description="Given a source image and a patch image, find the patch location in the source and return a bbox.",
            args_schema=PatchArgs,
            return_direct=True,
        ),
        StructuredTool.from_function(
            func=crossmodal_grounding_bbox_tool,
            name="crossmodal_grounding_bbox",
            description="Given a region in a reference image and a target of different modality, ground the corresponding region in target.",
            args_schema=CrossmodalArgs,
            return_direct=True,
        ),
    ]
