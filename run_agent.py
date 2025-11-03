from medsg_agent import build_agent

agent = build_agent()

q1 = (
    "Compare these two images carefully and give me the coordinates of their difference (registered) and output bbox JSON. "
    '{"images": ['
    '"C:/cpp/Medical/human_samples/Tools/registered_Diff/CTPelvic1K_CT_LumbarSpine_npy_imgs_ori_casenum_0027_sliceid_219.png", '
    '"C:/cpp/Medical/human_samples/Tools/registered_Diff/CTPelvic1K_CT_LumbarSpine_npy_imgs_casenum_0027_sliceid_219.png"'
    ']}'
)
print(agent.run(q1))

q2 = (
    "Compare these two images carefully and give me the coordinates of their NR difference and output bbox JSON. "
    '{"images": ['
    '"C:/cpp/Medical/human_samples/Tools/non_registered_Diff/LUNA16_CT_RightLung_npy_imgs_ori_casenum_173931884906244951746140865701_sliceid_113.png", '
    '"C:/cpp/Medical/human_samples/Tools/non_registered_Diff/LUNA16_CT_RightLung_npy_imgs_casenum_173931884906244951746140865701_sliceid_113.png"'
    ']}'
)
print(agent.run(q2))

q3 = (
    "These images share one object in common (the object marked with red bounding box in the first image "
    "<|box_start|>(155,120),(204,173)<|box_end|>). Recognize and locate this object in the second image. "
    '{"images": ['
    '"C:/cpp/Medical/human_samples/Tools/multi_view/MyoPS2020_MRI_LeftVentricularMyocardium_C0_npy_imgs_casenum_myops_training_110_front.png", '
    '"C:/cpp/Medical/human_samples/Tools/multi_view/MyoPS2020_MRI_LeftVentricularMyocardium_C0_npy_imgs_casenum_myops_training_110_top.png", '
    '"C:/cpp/Medical/human_samples/Tools/multi_view/MyoPS2020_MRI_LeftVentricularMyocardium_C0_npy_imgs_casenum_myops_training_110_side.png"'
    '], "hint": "multi_view"}'
)
print(agent.run(q3))

q4 = (
    "You are given a source image followed by its several regions. Please locate the first region picture in the source image. "
    '{"images": ['
    '"C:/cpp/Medical/human_samples/Tools/patch/CTSpine1K_Z_CT_LumbarSpine5_npy_imgs_ct_00--CTSpine1K_Full--1.3.6.1.4.1.9328.50.4.0103--z_0281.png", '
    '"C:/cpp/Medical/human_samples/Tools/patch/CTSpine1K_Z_CT_LumbarSpine5_npy_imgs_ct_00--CTSpine1K_Full--1.3.6.1.4.1.9328.50.4.0103--z_0281_0.png", '
    '"C:/cpp/Medical/human_samples/Tools/patch/CTSpine1K_Z_CT_LumbarSpine5_npy_imgs_ct_00--CTSpine1K_Full--1.3.6.1.4.1.9328.50.4.0103--z_0281_1.png", '
    '"C:/cpp/Medical/human_samples/Tools/patch/CTSpine1K_Z_CT_LumbarSpine5_npy_imgs_ct_00--CTSpine1K_Full--1.3.6.1.4.1.9328.50.4.0103--z_0281_2.png"'
    ']}'
)
print(agent.run(q4))

q5 = (
    "The following are two images for you to consider. For the area marked by the red bounding box in the first image, "
    "identify and locate the corresponding area in the second image that serves a similar function or shares a similar meaning. "
    '{"images": ['
    '"C:/cpp/Medical/human_samples/Tools/crossmodal/CMRxMotions_MRI_LeftVentricle_npy_imgs_mr_cmr--CMRxMotions--P010-4-ES--x_0004.png", '
    '"C:/cpp/Medical/human_samples/Tools/crossmodal/CAMUS_US_LeftVentricleEpicardium_2CH_ED_npy_imgs_casenum_297_sliceid_0.png"'
    ']}'
)
print(agent.run(q5))

q6 = (
    "Find and locate where does the object in image-1 locate in the image-2."
    "visual-concept match"
    '{"images": ['
    '"C:/cpp/Medical/human_samples/Tools/ufaq_concept/LUNA16_CT_LeftLung_npy_imgs_casenum_619372068417051974713149104919_sliceid_143.png", '
    '"C:/cpp/Medical/human_samples/Tools/ufaq_concept/LUNA16_CT_LeftLung_npy_imgs_ori_casenum_619372068417051974713149104919_sliceid_143.png"'
    ']}'
)
print(agent.run(q6))
