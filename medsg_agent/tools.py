from __future__ import annotations
import json
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from medsg_agent.utils import read_image, BBox
from medsg_agent.tasks.task1_registered_diff import get_subtracted_bbox
from medsg_agent.tasks.task2_nonregistered_diff import get_diff_bbox_with_registration_robust
from medsg_agent.tasks.task3_multi_view import multi_view_grounding_bbox
from medsg_agent.tasks.task5_concept_match import concept_match_bbox
from medsg_agent.tasks.task6_patch_grounding import patch_grounding_bbox_robust
from medsg_agent.tasks.task7_crossmodal import crossmodal_grounding_bbox


# ---------- Argument Schemas ----------
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


# ---------- Helpers ----------
def _fmt_out(bbox, conf, details):
    out = {
        "bbox": None if bbox is None else BBox(bbox.x1, bbox.y1, bbox.x2, bbox.y2).to_dict(),
        "confidence": float(conf),
        "details": details,
    }
    return json.dumps(out)


# ---------- Tools ----------
def registered_diff_bbox_tool(
    image_a: str,
    image_b: str,
    method: str = "ssim",
    thresh: float = 0.05,
    morph: int = 3,
) -> str:
    A = read_image(image_a); B = read_image(image_b)
    bbox, conf, det = get_subtracted_bbox(A, B, method=method, thresh=thresh, morph=morph)
    det = {**det, "tool": "registered_diff_bbox"}
    return _fmt_out(bbox, conf, det)

def nonregistered_diff_bbox_tool(
    image_a: str,
    image_b: str,
    use_phasecorr_first: bool = True,
    edge_band_frac: float = 0.02,
    morph: int = 3,
    thresh: float = 0.25,
) -> str:
    A = read_image(image_a); B = read_image(image_b)
    bbox, conf, det = get_diff_bbox_with_registration_robust(
        A, B,
        use_phasecorr_first=use_phasecorr_first,
        edge_band_frac=edge_band_frac,
        morph=morph,
        thresh=thresh,
    )
    det = {**det, "tool": "nonregistered_diff_bbox"}
    return _fmt_out(bbox, conf, det)

def multi_view_grounding_bbox_tool(
    ref_image: str,
    tgt_image: str,
    ref_bbox: List[int],
    method: str = "auto",
) -> str:
    R = read_image(ref_image); T = read_image(tgt_image)
    rb = BBox(*ref_bbox)
    bbox, conf, det = multi_view_grounding_bbox(R, T, rb, method=method)
    det = {**det, "tool": "multi_view_grounding_bbox"}
    return _fmt_out(bbox, conf, det)

def concept_match_bbox_tool(
    ref_image: str,
    tgt_image: str,
    ref_bbox: Optional[List[int]] = None,
) -> str:
    R = read_image(ref_image); T = read_image(tgt_image)
    rb = BBox(*ref_bbox) if ref_bbox else None
    bbox, conf, det = concept_match_bbox(R, rb, T, method="auto")
    det = {**det, "tool": "concept_match_bbox"}
    return _fmt_out(bbox, conf, det)

def patch_grounding_bbox_tool(
    source_image: str,
    patch_image: str,
    scales: Optional[List[float]] = None,
    rotations: Optional[List[float]] = None,
    alpha: float = 0.6,
) -> str:
    S = read_image(source_image); P = read_image(patch_image)
    bbox, conf, det = patch_grounding_bbox_robust(P, S, scales=scales, rotations=rotations, alpha=alpha)
    det = {**det, "tool": "patch_grounding_bbox"}
    return _fmt_out(bbox, conf, det)

def crossmodal_grounding_bbox_tool(
    ref_image: str,
    tgt_image: str,
    ref_bbox: Optional[List[int]] = None,
    scales: Optional[List[float]] = None,
    rotations: Optional[List[float]] = None,
    weights: Optional[List[float]] = None,
) -> str:
    R = read_image(ref_image); T = read_image(tgt_image)
    rb = BBox(*ref_bbox) if ref_bbox else None
    wtuple = tuple(weights) if weights else (0.45, 0.35, 0.20)
    bbox, conf, det = crossmodal_grounding_bbox(R, rb, T, scales=scales, rotations=rotations, weights=wtuple)
    det = {**det, "tool": "crossmodal_grounding_bbox"}
    return _fmt_out(bbox, conf, det)


# ---------- Registry ----------
def get_tools():
    return [
        StructuredTool.from_function(
            func=registered_diff_bbox_tool,
            name="registered_diff_bbox",
            description="Given two *registered* medical images (same pose), find the bounding box of the differing region. Inputs: image_a, image_b.",
            args_schema=TwoImageArgs,
            return_direct=True,
        ),
        StructuredTool.from_function(
            func=nonregistered_diff_bbox_tool,
            name="nonregistered_diff_bbox",
            description="Given two *unregistered* medical images (slight pose shift), align them and find the difference bbox. Inputs: image_a, image_b.",
            args_schema=NonRegArgs,
            return_direct=True,
        ),
        StructuredTool.from_function(
            func=multi_view_grounding_bbox_tool,
            name="multi_view_grounding_bbox",
            description="Given a reference image + bbox and a target view image of same subject, locate corresponding bbox in target.",
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
            description="Given a source whole image and a patch (region) image, find where the patch lies in the source and return a bbox.",
            args_schema=PatchArgs,
            return_direct=True,
        ),
        StructuredTool.from_function(
            func=crossmodal_grounding_bbox_tool,
            name="crossmodal_grounding_bbox",
            description="Given a region in a reference image and a target image of different modality, ground the corresponding region in target.",
            args_schema=CrossmodalArgs,
            return_direct=True,
        ),
    ]
