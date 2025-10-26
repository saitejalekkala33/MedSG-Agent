from __future__ import annotations
import json
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from medsg_agent.utils import (
    read_image,
    BBox,
    parse_two_image_indices,
    parse_ref_bbox_from_question,
    parse_region_index_from_question,
    tight_mask,
    bbox_from_mask,
)
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

class GalleryDiffArgs(BaseModel):
    images: List[str] = Field(..., description="List of image file paths.")
    question: str = Field(..., description="Natural language pointing to which two images to compare.")
    registered: Optional[bool] = Field(default=None, description="True for registered; False for non-registered; None to infer.")
    method: Optional[str] = Field(default="ssim", description="For registered mode: 'ssim' or 'abs'.")
    thresh: Optional[float] = Field(default=0.25, description="Base threshold for difference.")
    morph: Optional[int] = Field(default=3, description="Morph kernel size.")
    use_phasecorr_first: Optional[bool] = Field(default=True, description="For non-registered mode.")
    edge_band_frac: Optional[float] = Field(default=0.02, description="For non-registered mode.")

class GalleryRouteArgs(BaseModel):
    images: List[str] = Field(..., description="List of image paths (2+).")
    question: str = Field(..., description="Natural-language instruction indicating which images/regions to use and the task.")
    task_hint: Optional[str] = Field(
        default=None,
        description="Optional explicit task: one of {'registered_diff','nonregistered_diff','multi_view','concept','patch','crossmodal'}."
    )
    method: Optional[str] = Field(default="ssim", description="For registered diff: 'ssim' or 'abs'.")
    thresh: Optional[float] = Field(default=0.25, description="Base threshold for difference.")
    morph: Optional[int] = Field(default=3, description="Morph kernel size.")
    use_phasecorr_first: Optional[bool] = Field(default=True, description="For non-registered diff.")
    edge_band_frac: Optional[float] = Field(default=0.02, description="For non-registered diff.")
    mv_method: Optional[str] = Field(default="auto", description="For multi-view: 'orb'|'ncc'|'auto'.")
    patch_scales: Optional[List[float]] = Field(default=None, description="For patch task.")
    patch_rotations: Optional[List[float]] = Field(default=None, description="For patch task.")
    patch_alpha: Optional[float] = Field(default=0.6, description="For patch task.")
    xmod_scales: Optional[List[float]] = Field(default=None, description="For crossmodal task.")
    xmod_rotations: Optional[List[float]] = Field(default=None, description="For crossmodal task.")
    xmod_weights: Optional[List[float]] = Field(default=None, description="For crossmodal task [wa,wc,wm].")


# ---------- Tool utilities ----------
def _fmt_out(bbox, conf, details):
    out = {
        "bbox": None if bbox is None else BBox(bbox.x1, bbox.y1, bbox.x2, bbox.y2).to_dict(),
        "confidence": float(conf),
        "details": details,
    }
    return json.dumps(out)

def _infer_task(question: str, task_hint: Optional[str]) -> str:
    if task_hint:
        hint = task_hint.lower().strip()
        alias = {
            "registered": "registered_diff",
            "nonregistered": "nonregistered_diff",
            "non-registered": "nonregistered_diff",
            "multi": "multi_view",
            "multi-view": "multi_view",
            "concept_match": "concept",
            "cross-modal": "crossmodal",
            "crossmodal": "crossmodal",
            "patch_grounding": "patch",
        }
        return alias.get(hint, hint)

    q = question.lower()
    if "patch" in q or "region picture" in q or "regions" in q:
        return "patch"
    if "red bounding box" in q or "marked with red bounding box" in q or "<|box_start|>" in question:
        return "multi_view"
    if "cross-modal" in q or "cross modal" in q or "different modality" in q or "serves a similar function" in q or ("ct" in q and "mri" in q):
        return "crossmodal"
    if "multi-view" in q or "multi view" in q:
        return "multi_view"
    if "concept" in q:
        return "concept"
    if "registered" in q and "compare" in q:
        return "registered_diff"
    if "compare" in q:
        return "nonregistered_diff"
    return "nonregistered_diff"


# ---------- Base task tools ----------
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


# ---------- Gallery diff tool (2-image selection + diff) ----------
class _InternalGalleryMode:
    REGISTERED = "registered"
    NONREGISTERED = "nonregistered"

def gallery_diff_bbox_tool(
    images: List[str],
    question: str,
    registered: Optional[bool] = None,
    method: str = "ssim",
    thresh: float = 0.25,
    morph: int = 3,
    use_phasecorr_first: bool = True,
    edge_band_frac: float = 0.02,
) -> str:
    n = len(images)
    if n < 2:
        return json.dumps({"bbox": None, "confidence": 0.0, "details": {"error": "need >=2 images", "tool": "gallery_diff_bbox"}})

    i, j = parse_two_image_indices(question, n)
    if i == j:
        j = (i + 1) % n

    A = read_image(images[i]); B = read_image(images[j])

    ql = question.lower()
    if registered is True:
        mode = _InternalGalleryMode.REGISTERED
    elif registered is False:
        mode = _InternalGalleryMode.NONREGISTERED
    else:
        if "non-registered" in ql or "unregistered" in ql or "unaligned" in ql:
            mode = _InternalGalleryMode.NONREGISTERED
        elif "registered" in ql:
            mode = _InternalGalleryMode.REGISTERED
        else:
            mode = _InternalGalleryMode.NONREGISTERED

    if mode == _InternalGalleryMode.REGISTERED:
        bbox, conf, det = get_subtracted_bbox(A, B, method=method, thresh=thresh, morph=morph)
        tool_name = "registered_diff_bbox"
    else:
        bbox, conf, det = get_diff_bbox_with_registration_robust(
            A, B,
            use_phasecorr_first=use_phasecorr_first,
            edge_band_frac=edge_band_frac,
            morph=morph,
            thresh=thresh,
        )
        tool_name = "nonregistered_diff_bbox"

    det = {
        **det,
        "mode": mode,
        "tool": tool_name,
        "selected_indices": [int(i), int(j)],
        "selected_paths": [images[i], images[j]],
    }
    return _fmt_out(bbox, conf, det)


def gallery_grounding_router_tool(
    images: List[str],
    question: str,
    task_hint: Optional[str] = None,
    method: str = "ssim",
    thresh: float = 0.25,
    morph: int = 3,
    use_phasecorr_first: bool = True,
    edge_band_frac: float = 0.02,
    mv_method: str = "auto",
    patch_scales: Optional[List[float]] = None,
    patch_rotations: Optional[List[float]] = None,
    patch_alpha: float = 0.6,
    xmod_scales: Optional[List[float]] = None,
    xmod_rotations: Optional[List[float]] = None,
    xmod_weights: Optional[List[float]] = None,
) -> str:
    n = len(images)
    if n < 2:
        return json.dumps({"bbox": None, "confidence": 0.0, "details": {"error": "need >=2 images", "tool": "gallery_grounding_router"}})

    task = _infer_task(question, task_hint)

    if task == "patch":
        src_path = images[0]
        r_idx_1based = parse_region_index_from_question(question, default=1)
        p_idx = r_idx_1based
        if p_idx >= n:
            return json.dumps({"bbox": None, "confidence": 0.0, "details": {"error": f"region index {r_idx_1based} out of range", "task": task, "tool": "patch_grounding_bbox"}})
        S = read_image(src_path)
        P = read_image(images[p_idx])
        bbox, conf, det = patch_grounding_bbox_robust(
            P, S,
            scales=patch_scales,
            rotations=patch_rotations,
            alpha=patch_alpha,
        )
        det = {**det, "task": task, "tool": "patch_grounding_bbox", "selected_indices": [0, p_idx], "selected_paths": [src_path, images[p_idx]]}
        return _fmt_out(bbox, conf, det)

    i, j = parse_two_image_indices(question, n)
    if i == j:
        j = (i + 1) % n
    A_path, B_path = images[i], images[j]
    A, B = read_image(A_path), read_image(B_path)

    if task == "registered_diff":
        bbox, conf, det = get_subtracted_bbox(A, B, method=method, thresh=thresh, morph=morph)
        det = {**det, "task": task, "tool": "registered_diff_bbox", "selected_indices": [i, j], "selected_paths": [A_path, B_path]}
        return _fmt_out(bbox, conf, det)

    if task == "nonregistered_diff":
        bbox, conf, det = get_diff_bbox_with_registration_robust(
            A, B,
            use_phasecorr_first=use_phasecorr_first,
            edge_band_frac=edge_band_frac,
            morph=morph,
            thresh=thresh,
        )
        det = {**det, "task": task, "tool": "nonregistered_diff_bbox", "selected_indices": [i, j], "selected_paths": [A_path, B_path]}
        return _fmt_out(bbox, conf, det)

    if task == "multi_view":
        rb = parse_ref_bbox_from_question(question)
        if rb is None:
            m = tight_mask(A)
            rb = bbox_from_mask(m) or BBox(0, 0, A.shape[1]-1, A.shape[0]-1)
        bbox, conf, det = multi_view_grounding_bbox(A, B, rb, method=mv_method)
        det = {**det, "task": task, "tool": "multi_view_grounding_bbox", "selected_indices": [i, j], "selected_paths": [A_path, B_path]}
        return _fmt_out(bbox, conf, det)

    if task == "concept":
        rb = parse_ref_bbox_from_question(question)
        bbox, conf, det = concept_match_bbox(A, rb, B, method="auto")
        det = {**det, "task": task, "tool": "concept_match_bbox", "selected_indices": [i, j], "selected_paths": [A_path, B_path]}
        return _fmt_out(bbox, conf, det)

    if task == "crossmodal":
        rb = parse_ref_bbox_from_question(question)
        weights = tuple(xmod_weights) if xmod_weights else (0.45, 0.35, 0.20)
        bbox, conf, det = crossmodal_grounding_bbox(
            A, rb, B,
            scales=xmod_scales,
            rotations=xmod_rotations,
            weights=weights,
        )
        det = {**det, "task": task, "tool": "crossmodal_grounding_bbox", "selected_indices": [i, j], "selected_paths": [A_path, B_path]}
        return _fmt_out(bbox, conf, det)

    bbox, conf, det = get_diff_bbox_with_registration_robust(
        A, B,
        use_phasecorr_first=use_phasecorr_first,
        edge_band_frac=edge_band_frac,
        morph=morph,
        thresh=thresh,
    )
    det = {**det, "task": "nonregistered_diff", "tool": "nonregistered_diff_bbox", "selected_indices": [i, j], "selected_paths": [A_path, B_path]}
    return _fmt_out(bbox, conf, det)


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
        StructuredTool.from_function(
            func=gallery_diff_bbox_tool,
            name="gallery_diff_bbox",
            description="Given a list of images and a natural-language question, pick two images to compare and return a difference bbox (registered or non-registered).",
            args_schema=GalleryDiffArgs,
            return_direct=True,
        ),
        StructuredTool.from_function(
            func=gallery_grounding_router_tool,
            name="gallery_grounding_router",
            description=(
                "Given a list of images and a natural-language question (e.g., "
                "'compare the second and fourth images' or 'locate the first region picture'), "
                "automatically pick the right images and run the correct task: "
                "registered/nonregistered diff, multi-view, concept, patch, or crossmodal. "
                "Returns a bbox JSON."
            ),
            args_schema=GalleryRouteArgs,
            return_direct=True,
        ),
    ]
