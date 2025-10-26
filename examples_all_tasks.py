
"""
All task examples in ONE file.
Comment/uncomment blocks as you wish.
"""
import os, re, glob, json
import numpy as np, cv2, matplotlib.pyplot as plt

from medsg_agent.utils import BBox, read_image, draw_box, list_to_bbox, compute_iou, smart_path
from medsg_agent.tasks.task1_registered_diff import get_subtracted_bbox
from medsg_agent.tasks.task2_nonregistered_diff import get_diff_bbox_with_registration_robust, ecc_register
from medsg_agent.tasks.task3_multi_view import multi_view_grounding_bbox
from medsg_agent.tasks.task5_concept_match import concept_match_bbox
from medsg_agent.tasks.task6_patch_grounding import patch_grounding_bbox_robust
from medsg_agent.tasks.task7_crossmodal import crossmodal_grounding_bbox

# ---------- Task 1: Registered Difference ----------
def run_task1_example():
    sample = {
        "task": "Registered_Diff",
        "images": [
            "./registered_Diff/CTPelvic1K_CT_LumbarSpine_npy_imgs_ori_casenum_0027_sliceid_219.png",
            "./registered_Diff/CTPelvic1K_CT_LumbarSpine_npy_imgs_casenum_0027_sliceid_219.png"
        ],
        "question": "Compare these two images carefully and give me the coordinates of their difference.",
        "answer": [186, 295, 193, 312] 
    }

    p1, p2 = sample["images"]
    img_a = read_image(p1); img_b = read_image(p2)
    gt = list_to_bbox(sample["answer"])

    pred, conf, extra = get_subtracted_bbox(img_a, img_b, method="ssim", thresh=0.05, morph=3)
    iou = compute_iou(pred, gt) if pred else 0.0

    print(f"[Task1] Predicted: {pred} | GT: {gt} | Conf: {conf:.4f} | IoU: {iou:.4f} | {extra}")

    # Visualization
    try:
        from skimage.metrics import structural_similarity as ssim
        import numpy as np
        from medsg_agent.utils import to_gray01
        A, B = to_gray01(img_a), to_gray01(img_b)
        _, ssim_map = ssim(A, B, full=True)
        diff_vis = (1.0 - ssim_map)
    except Exception:
        from medsg_agent.utils import to_gray01
        diff_vis = np.abs(to_gray01(img_a) - to_gray01(img_b))
    dv = diff_vis.copy(); dv -= dv.min(); dv /= (dv.max() + 1e-8)

    vis_pred = draw_box(img_b, pred, color=(0,255,0), label="Pred")
    vis_gt   = draw_box(vis_pred, gt,   color=(0,0,255), label="GT")

    plt.figure(figsize=(14,4))
    plt.subplot(1,3,1); plt.imshow(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)); plt.title("Image A"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(cv2.cvtColor(vis_gt, cv2.COLOR_BGR2RGB)); plt.title("Image B + Pred/GT"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(dv, cmap="gray"); plt.title("Diff (visualization)"); plt.axis("off")
    plt.tight_layout(); plt.show()

# ---------- Task 2: Non-Registered Difference ----------
def run_task2_example():
    sample = {
        "task": "Non_Registered_Diff",
        "images": [
            "MedSG-Bench/MedSG-Bench/Task2/LUNA16_CT_RightLung_npy_imgs_ori_casenum_173931884906244951746140865701_sliceid_113.png",
            "MedSG-Bench/MedSG-Bench/Task2/LUNA16_CT_RightLung_npy_imgs_casenum_173931884906244951746140865701_sliceid_113.png"
        ],
        "question": "Compare these two images carefully and give me the coordinates of their difference.",
        "answer": [103,188,124,202]
    }

    p1, p2 = sample["images"]
    img_a, img_b = read_image(p1), read_image(p2)
    gt = list_to_bbox(sample["answer"])

    pred, conf, extra = get_diff_bbox_with_registration_robust(img_a, img_b, use_phasecorr_first=True)
    iou = compute_iou(pred, gt) if pred else 0.0
    print(f"[Task2] Pred: {pred} | GT: {gt} | Conf: {conf:.4f} | IoU: {iou:.4f} | Band: {extra.get('band')}")

    # Visualize post-reg diff
    from medsg_agent.utils import to_gray01
    ws_gray, _ = ecc_register(img_a, img_b, warp_mode=cv2.MOTION_EUCLIDEAN)
    try:
        from skimage.metrics import structural_similarity as ssim
        _, ssim_map1 = ssim(ws_gray, to_gray01(img_b), full=True)
        diff_post = 1.0 - ssim_map1
    except Exception:
        diff_post = np.abs(ws_gray - to_gray01(img_b))
    h,w = diff_post.shape; band = extra["band"]
    diff_post[:band,:]=0; diff_post[-band:,:]=0; diff_post[:,:band]=0; diff_post[:,-band:]=0
    dv = diff_post - diff_post.min(); dv /= (dv.max() + 1e-8)

    vis = draw_box(img_b, gt,   color=(0,0,255), label="GT")
    vis = draw_box(vis, pred, color=(0,255,0), label="Pred")

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.imshow(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)); plt.title("Image B (dst)"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(dv, cmap="gray"); plt.title("Post-reg diff (edges suppressed)"); plt.axis("off")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); plt.title("Target + GT (red) & Pred (green)")
    plt.axis("off"); plt.show()

# ---------- Task 3: Multi-view ----------
def run_task3_examples():
    samples = [
        {
            "task": "multi_view",
            "images": [
                "./multi_view/MyoPS2020_MRI_LeftVentricularMyocardium_C0_npy_imgs_casenum_myops_training_110_front.png",
                "./multi_view/MyoPS2020_MRI_LeftVentricularMyocardium_C0_npy_imgs_casenum_myops_training_110_top.png",
                "./multi_view/MyoPS2020_MRI_LeftVentricularMyocardium_C0_npy_imgs_casenum_myops_training_110_side.png"
            ],
            "question": "These images share one object in common(the object marked with red bounding box in the first image(<|box_start|>(155,120),(204,173)<|box_end|>). Recognize and locate this object in the second image.",
            "answer": [119, 0, 173, 55]
        },
        {
            "task": "multi_view",
            "images": [
                "./multi_view/LUNA16_CT_LeftLung_npy_imgs_casenum_259018373683540453277752706262_front.png",
                "./multi_view/LUNA16_CT_LeftLung_npy_imgs_casenum_259018373683540453277752706262_top.png",
                "./multi_view/LUNA16_CT_LeftLung_npy_imgs_casenum_259018373683540453277752706262_side.png"
            ],
            "question": "These images share one object in common(the object marked with red bounding box in the first image(<|box_start|>(168,83),(286,270)<|box_end|>). Recognize and locate this object in the third image.",
            "answer": [186, 4, 299, 54]
        }
    ]

    def parse_ref_bbox_from_question(q: str):
        m = re.search(r"<\|box_start\|\>\s*\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)\s*<\|box_end\|\>", q)
        x1, y1, x2, y2 = map(int, m.groups())
        return BBox(x1, y1, x2, y2)

    def target_index_from_question(q: str, default=1):
        ql = q.lower()
        if "second" in ql: return 1
        if "third"  in ql: return 2
        if "first"  in ql: return 0
        return default

    for idx, sample in enumerate(samples, 1):
        ref_idx = 0
        tgt_idx = target_index_from_question(sample["question"], default=1)
        p_ref = sample["images"][ref_idx]
        p_tgt = sample["images"][tgt_idx]
        ref_img = read_image(p_ref)
        tgt_img = read_image(p_tgt)
        ref_bbox = parse_ref_bbox_from_question(sample["question"])
        gt_bbox = list_to_bbox(sample["answer"])

        pred_bb, pred_conf, dbg = multi_view_grounding_bbox(ref_img, tgt_img, ref_bbox, method="auto")
        iou = compute_iou(pred_bb, gt_bbox) if pred_bb else 0.0

        print(f"[Task3|{idx}] Pred: {pred_bb} | GT: {gt_bbox} | Method: {dbg.get('method','?')} | Conf: {pred_conf:.4f} | IoU: {iou:.4f}")
        ref_vis = draw_box(ref_img, ref_bbox, color=(0,0,255), label="Ref")
        tgt_vis = draw_box(tgt_img, gt_bbox,  color=(255,0,0), label="GT")
        tgt_vis = draw_box(tgt_vis, pred_bb,  color=(0,255,0), label="Pred")
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1); plt.imshow(cv2.cvtColor(ref_vis, cv2.COLOR_BGR2RGB)); plt.title("Reference (with ref box)"); plt.axis("off")
        plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(tgt_vis, cv2.COLOR_BGR2RGB)); plt.title("Target: GT (red) & Pred (green)"); plt.axis("off")
        plt.tight_layout(); plt.show()

# ---------- Task 5: Concept match ----------
def run_task5_examples():
    samples = [
        {
            "images": [
                "./ufaq_concept/AbdomenCT1K_CT_RightKidney_npy_imgs_casenum_00709_sliceid_395.png",
                "./ufaq_concept/AbdomenCT1K_CT_RightKidney_npy_imgs_ori_casenum_00709_sliceid_395.png"
            ],
            "answer": [192, 113, 232, 157]
        },
        {
            "images": [
                "./ufaq_concept/AbdomenCT1K_CT_Liver_npy_imgs_casenum_00919_sliceid_149.png",
                "./ufaq_concept/AbdomenCT1K_CT_Liver_npy_imgs_ori_casenum_00919_sliceid_149.png"
            ],
            "answer": [69, 167, 94, 236]
        },
        {
            "images": [
                "./ufaq_concept/LUNA16_CT_LeftLung_npy_imgs_casenum_619372068417051974713149104919_sliceid_143.png",
                "./ufaq_concept/LUNA16_CT_LeftLung_npy_imgs_ori_casenum_619372068417051974713149104919_sliceid_143.png"
            ],
            "answer": [171, 84, 290, 250]
        },
        {
            "images": [
                "./ufaq_concept/HaNSeg_sagittal_CT_Mandible_npy_imgs_casenum_25_sliceid_551.png",
                "./ufaq_concept/HaNSeg_sagittal_CT_Mandible_npy_imgs_ori_casenum_25_sliceid_551.png"
            ],
            "answer": [129, 172, 139, 195]
        }
    ]

    for i,s in enumerate(samples,1):
        ref = read_image(s["images"][0])
        tgt = read_image(s["images"][1])
        gt = list_to_bbox(s["answer"])
        pred, conf, dbg = concept_match_bbox(ref, None, tgt, method="auto")
        iou = compute_iou(pred, gt) if pred else 0.0
        print(f"[Task5|{i}] Pred: {pred} | Conf: {conf:.4f} | IoU: {iou:.4f} | {dbg}")
        ref_vis = ref.copy()
        tgt_vis = draw_box(tgt, gt, (0,0,255), 2, "GT")
        tgt_vis = draw_box(tgt_vis, pred, (0,255,0), 2, "Pred")
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1); plt.imshow(cv2.cvtColor(ref_vis, cv2.COLOR_BGR2RGB)); plt.title("Image-1 (concept)"); plt.axis("off")
        plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(tgt_vis, cv2.COLOR_BGR2RGB)); plt.title("Image-2 (GT red, Pred green)"); plt.axis("off")
        plt.tight_layout(); plt.show()

# ---------- Task 6: Patch grounding ----------
def run_task6_examples():
    samples = [
        {
            "task":"patch",
            "images":[
                "./patch/CTSpine1K_Z_CT_LumbarSpine5_npy_imgs_ct_00--CTSpine1K_Full--1.3.6.1.4.1.9328.50.4.0103--z_0281.png",
                "./patch/CTSpine1K_Z_CT_LumbarSpine5_npy_imgs_ct_00--CTSpine1K_Full--1.3.6.1.4.1.9328.50.4.0103--z_0281_0.png",
                "./patch/CTSpine1K_Z_CT_LumbarSpine5_npy_imgs_ct_00--CTSpine1K_Full--1.3.6.1.4.1.9328.50.4.0103--z_0281_1.png",
                "./patch/CTSpine1K_Z_CT_LumbarSpine5_npy_imgs_ct_00--CTSpine1K_Full--1.3.6.1.4.1.9328.50.4.0103--z_0281_2.png"
            ],
            "question":"You are given a source image followed by its several regions. Please locate the first region picture in the source image.",
            "answer":[38,22,295,167]
        },
        {
            "task":"patch",
            "images":[
                "./patch/LUNA16_CT_Bronchus_npy_imgs_casenum_311236942972970815890902714604_sliceid_349.png",
                "./patch/LUNA16_CT_Bronchus_npy_imgs_casenum_311236942972970815890902714604_sliceid_349_0.png",
                "./patch/LUNA16_CT_Bronchus_npy_imgs_casenum_311236942972970815890902714604_sliceid_349_1.png",
                "./patch/LUNA16_CT_Bronchus_npy_imgs_casenum_311236942972970815890902714604_sliceid_349_2.png",
                "./patch/LUNA16_CT_Bronchus_npy_imgs_casenum_311236942972970815890902714604_sliceid_349_3.png"
            ],
            "question":"You are given a source image followed by its several regions. Please locate the second region picture in the source image.",
            "answer":[102,190,267,303]
        },
        {
            "task":"patch",
            "images":[
                "./patch/BAGLS_Endoscope_Glottis_npy_imgs_casenum_15491_sliceid_0.png",
                "./patch/BAGLS_Endoscope_Glottis_npy_imgs_casenum_15491_sliceid_0_0.png",
                "./patch/BAGLS_Endoscope_Glottis_npy_imgs_casenum_15491_sliceid_0_1.png",
                "./patch/BAGLS_Endoscope_Glottis_npy_imgs_casenum_15491_sliceid_0_2.png",
                "./patch/BAGLS_Endoscope_Glottis_npy_imgs_casenum_15491_sliceid_0_3.png"
            ],
            "question":"You are given a source image followed by its several regions. Please locate the first region picture in the source image.",
            "answer":[126,29,273,191]
        }
    ]

    def region_index_from_question(q: str, default: int = 1) -> int:
        ords={"first":1,"second":2,"third":3,"fourth":4,"fifth":5}
        m=re.search(r'\b(first|second|third|fourth|fifth)\b', q.lower())
        return ords.get(m.group(1),default) if m else default

    for i,s in enumerate(samples,1):
        src = read_image(s["images"][0])
        ridx = region_index_from_question(s["question"], default=1)
        patch = read_image(s["images"][ridx])
        gt = list_to_bbox(s["answer"])
        pred, conf, dbg = patch_grounding_bbox_robust(patch, src, scales=[1.1,1.05,1.0,0.95,0.9,0.85], rotations=[0,5,-5,10,-10], alpha=0.65)
        iou = compute_iou(pred, gt) if pred else 0.0
        print(f"[Task6|{i}] Region: {ridx} | Pred: {pred} | Conf: {conf:.4f} | IoU: {iou:.4f} | {dbg}")
        vis = draw_box(src, gt, (0,0,255), 2, "GT")
        vis = draw_box(vis, pred, (0,255,0), 2, "Pred")
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1); plt.imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)); plt.title("Patch"); plt.axis("off")
        plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); plt.title("Source (GT red, Pred green)"); plt.axis("off")
        plt.tight_layout(); plt.show()

# ---------- Task 7: Cross-modal ----------
def run_task7_examples():
    samples = [
        {
            "images":[
                "./crossmodal/CMRxMotions_MRI_LeftVentricle_npy_imgs_mr_cmr--CMRxMotions--P010-4-ES--x_0004.png",
                "./crossmodal/CAMUS_US_LeftVentricleEpicardium_2CH_ED_npy_imgs_casenum_297_sliceid_0.png"
            ],
            "question":"The following are two images for you to consider. For the area marked by the red bounding box in the first image, identify and locate the corresponding area in the second image that serves a similar function or shares a similar meaning.",
            "answer":[112,44,279,334]
        },
        {
            "images":[
                "./crossmodal/AbdomenCT1K_CT_Spleen_npy_imgs_casenum_00220_sliceid_470.png",
                "./crossmodal/AMOS22_MRI_Spleen_npy_imgs_casenum_0548_sliceid_41.png"
            ],
            "question":"The following are two images for you to consider. For the area marked by the red bounding box in the first image, identify and locate the corresponding area in the second image that serves a similar function or shares a similar meaning.",
            "answer":[92,111,141,144]
        },
        {
            "images":[
                "./crossmodal/BraTS2020_MRI_BrainWholeTumor_imgs_casenum_194_sliceid_59_T2.png",
                "./crossmodal/BraTS2020_MRI_BrainWholeTumor_imgs_casenum_194_sliceid_59_T1ce.png",
                "./crossmodal/BraTS2020_MRI_BrainWholeTumor_imgs_casenum_194_sliceid_59_T1.png",
                "./crossmodal/BraTS2020_MRI_BrainWholeTumor_imgs_casenum_194_sliceid_59_FLAIR.png"
            ],
            "question":"The following are four images for you to consider. For the area marked by the red bounding box in the first image, identify and locate the corresponding area in the third image that serves a similar function or shares a similar meaning.",
            "answer":[191,132,258,238]
        },
        {
            "images":[
                "./crossmodal/CAMUS_US_LeftAtrium_imgs_casenum_46_sliceid_0_4CH_ED.png",
                "./crossmodal/CAMUS_US_LeftAtrium_imgs_casenum_46_sliceid_0_2CH_ED.png",
                "./crossmodal/CAMUS_US_LeftAtrium_imgs_casenum_46_sliceid_0_2CH_ES.png",
                "./crossmodal/CAMUS_US_LeftAtrium_imgs_casenum_46_sliceid_0_4CH_ES.png"
            ],
            "question":"The following are four images for you to consider. For the area marked by the red bounding box in the first image, identify and locate the corresponding area in the second image that serves a similar function or shares a similar meaning.",
            "answer":[113,210,179,284]
        }
    ]

    def parse_target_index(q: str, default: int = 1) -> int:
        words={"first":0,"second":1,"third":2,"fourth":3}
        ms=list(re.finditer(r'\b(first|second|third|fourth)\s+image', q.lower()))
        return words[ms[-1].group(1)] if ms else default

    for i,s in enumerate(samples,1):
        ref = read_image(s["images"][0])
        tgt_idx = parse_target_index(s["question"], default=1)
        tgt = read_image(s["images"][tgt_idx])
        gt  = list_to_bbox(s["answer"])
        pred, conf, dbg = crossmodal_grounding_bbox(ref, None, tgt,
                                                    scales=[1.2,1.15,1.1,1.05,1.0,0.95,0.9,0.85],
                                                    rotations=[0,5,-5,10,-10,15,-15],
                                                    weights=(0.45,0.35,0.20))
        iou = compute_iou(pred, gt) if pred else 0.0
        print(f"[Task7|{i}] Pred: {pred} | Conf: {conf:.4f} | IoU: {iou:.4f} | {dbg}")
        ref_vis = ref.copy()
        tgt_vis = draw_box(tgt, gt, (0,0,255), 3, "GT")
        tgt_vis = draw_box(tgt_vis, pred, (0,255,0), 3, "Pred")
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1); plt.imshow(cv2.cvtColor(ref_vis, cv2.COLOR_BGR2RGB)); plt.axis("off"); plt.title("Image-1")
        plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(tgt_vis, cv2.COLOR_BGR2RGB)); plt.axis("off"); plt.title("Target")
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    # Uncomment what you want to run:
    # run_task1_example()
    # run_task2_example()
    # run_task3_examples()
    # run_task5_examples()
    run_task6_examples()
    # run_task7_examples()
    pass
