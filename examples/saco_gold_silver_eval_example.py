# Copyright (c) Meta Platforms, Inc. and affiliates.


import copy
import json
import os

import numpy as np

from pycocotools.coco import COCO
from sam3.eval.cgf1_eval import CGF1Evaluator


# Update to the directory where the GT annotation and PRED files exist
GT_DIR = None  # PUT YOUR PATH HERE
PRED_DIR = None  # PUT YOUR PATH HERE


# Relative file names for GT files for 7 SA-Co/Gold subsets
saco_gold_gts = {
    # MetaCLIP Captioner
    "metaclip_nps": [
            "gold_metaclip_merged_a_release_test.json",
            "gold_metaclip_merged_b_release_test.json",
            "gold_metaclip_merged_c_release_test.json",
    ],
    # SA-1B captioner
    "sa1b_nps": [
            "gold_sa1b_merged_a_release_test.json",
            "gold_sa1b_merged_b_release_test.json",
            "gold_sa1b_merged_c_release_test.json",
    ],
    # Crowded
    "crowded": [
            "gold_crowded_merged_a_release_test.json",
            "gold_crowded_merged_b_release_test.json",
            "gold_crowded_merged_c_release_test.json",
    ],
    # FG Food
    "fg_food": [
            "gold_fg_food_merged_a_release_test.json",
            "gold_fg_food_merged_b_release_test.json",
            "gold_fg_food_merged_c_release_test.json",
    ],
    # FG Sports
    "fg_sports_equipment": [
            "gold_fg_sports_equipment_merged_a_release_test.json",
            "gold_fg_sports_equipment_merged_b_release_test.json",
            "gold_fg_sports_equipment_merged_c_release_test.json",
    ],
    # Attributes
    "attributes": [
            "gold_attributes_merged_a_release_test.json",
            "gold_attributes_merged_b_release_test.json",
            "gold_attributes_merged_c_release_test.json",
    ],
    # Wiki common
    "wiki_common": [
            "gold_wiki_common_merged_a_release_test.json",
            "gold_wiki_common_merged_b_release_test.json",
            "gold_wiki_common_merged_c_release_test.json",
    ],
}


results_gold = {}
results_gold_bbox = {}

for subset_name, gts in saco_gold_gts.items():
    print("Processing subset: ", subset_name)
    gt_paths = [os.path.join(GT_DIR, gt) for gt in gts]
    pred_path = os.path.join(PRED_DIR, f"gold_{subset_name}/dumps/gold_{subset_name}/coco_predictions_segm.json")
    
    evaluator = CGF1Evaluator(gt_path=gt_paths, verbose=True, iou_type="segm") 
    summary = evaluator.evaluate(pred_path)
    print(summary)

    cur_results = {}
    cur_results["cgf1"] = summary["cgF1_eval_segm_cgF1"] * 100
    cur_results["il_mcc"] = summary["cgF1_eval_segm_IL_MCC"]
    cur_results["pmf1"] = summary["cgF1_eval_segm_positive_micro_F1"] * 100
    results_gold[subset_name] = cur_results

    # Also eval bbox    
    evaluator = CGF1Evaluator(gt_path=gt_paths, verbose=True, iou_type="bbox") 
    summary = evaluator.evaluate(pred_path)
    print(summary)

    cur_results = {}
    cur_results["cgf1"] = summary["cgF1_eval_bbox_cgF1"] * 100
    cur_results["il_mcc"] = summary["cgF1_eval_bbox_IL_MCC"]
    cur_results["pmf1"] = summary["cgF1_eval_bbox_positive_micro_F1"] * 100
    results_gold_bbox[subset_name] = cur_results


# Compute averages
METRICS = ["cgf1", "il_mcc", "pmf1"]
avg_stats, avg_stats_bbox = {}, {}
for key in METRICS:
    avg_stats[key] = sum(res[key] for res in results_gold.values()) / len(results_gold)
    avg_stats_bbox[key] = sum(res[key] for res in results_gold_bbox.values()) / len(results_gold_bbox)
results_gold["Average"] = avg_stats
results_gold_bbox["Average"] = avg_stats_bbox


# Pretty print segmentation results
from IPython.display import HTML, display

row1, row2, row3 = "", "", ""
for subset in results_gold:
    row1 += f'<th colspan="3" style="text-align:center;border-left-style:solid;border-left-width:1px">{subset}</th>'
    row2 += "<th style='border-left-style:solid;border-left-width:1px'>" + "</th><th>".join(METRICS) + "</th>"
    row3 += "<td style='border-left-style:solid;border-left-width:1px'>" + "</td><td>".join([str(round(results_gold[subset][k], 2)) for k in METRICS])  + "</td>"

display(HTML(
   f"<table><thead><tr>{row1}</tr><tr>{row2}</tr></thead><tbody><tr>{row3}</tr></tbody></table>"
))


# Pretty print bbox detection results
from IPython.display import HTML, display

row1, row2, row3 = "", "", ""
for subset in results_gold:
    row1 += f'<th colspan="3" style="text-align:center;border-left-style:solid;border-left-width:1px">{subset}</th>'
    row2 += "<th style='border-left-style:solid;border-left-width:1px'>" + "</th><th>".join(METRICS) + "</th>"
    row3 += "<td style='border-left-style:solid;border-left-width:1px'>" + "</td><td>".join([str(round(results_gold_bbox[subset][k], 2)) for k in METRICS])  + "</td>"

display(HTML(
   f"<table><thead><tr>{row1}</tr><tr>{row2}</tr></thead><tbody><tr>{row3}</tr></tbody></table>"
))


# Update to the directory where the GT annotation and PRED files exist
GT_DIR = None  # PUT YOUR PATH HERE
PRED_DIR = None  # PUT YOUR PATH HERE


saco_silver_gts = {
    "bdd100k": "silver_bdd100k_merged_test.json",
    "droid": "silver_droid_merged_test.json",
    "ego4d": "silver_ego4d_merged_test.json",
    "food_rec": "silver_food_rec_merged_test.json",
    "geode": "silver_geode_merged_test.json",
    "inaturalist": "silver_inaturalist_merged_test.json",
    "nga_art": "silver_nga_art_merged_test.json",
    "sav": "silver_sav_merged_test.json",
    "yt1b": "silver_yt1b_merged_test.json",
    "fathomnet": "silver_fathomnet_test.json",
}


results_silver = {}
results_silver_bbox = {}

for subset_name, gt in saco_silver_gts.items():
    print("Processing subset: ", subset_name)
    gt_path = os.path.join(GT_DIR, gt)
    pred_path = os.path.join(PRED_DIR, f"silver_{subset_name}/dumps/silver_{subset_name}/coco_predictions_segm.json")
    
    evaluator = CGF1Evaluator(gt_path=gt_path, verbose=True, iou_type="segm") 
    summary = evaluator.evaluate(pred_path)
    print(summary)

    cur_results = {}
    cur_results["cgf1"] = summary["cgF1_eval_segm_cgF1"] * 100
    cur_results["il_mcc"] = summary["cgF1_eval_segm_IL_MCC"]
    cur_results["pmf1"] = summary["cgF1_eval_segm_positive_micro_F1"] * 100
    results_silver[subset_name] = cur_results

    # Also eval bbox    
    evaluator = CGF1Evaluator(gt_path=gt_path, verbose=True, iou_type="bbox") 
    summary = evaluator.evaluate(pred_path)
    print(summary)

    cur_results = {}
    cur_results["cgf1"] = summary["cgF1_eval_bbox_cgF1"] * 100
    cur_results["il_mcc"] = summary["cgF1_eval_bbox_IL_MCC"]
    cur_results["pmf1"] = summary["cgF1_eval_bbox_positive_micro_F1"] * 100
    results_silver_bbox[subset_name] = cur_results


# Compute averages
METRICS = ["cgf1", "il_mcc", "pmf1"]
avg_stats, avg_stats_bbox = {}, {}
for key in METRICS:
    avg_stats[key] = sum(res[key] for res in results_silver.values()) / len(results_silver)
    avg_stats_bbox[key] = sum(res[key] for res in results_silver_bbox.values()) / len(results_silver_bbox)
results_silver["Average"] = avg_stats
results_silver_bbox["Average"] = avg_stats_bbox


# Pretty print segmentation results
from IPython.display import HTML, display

row1, row2, row3 = "", "", ""
for subset in results_silver:
    row1 += f'<th colspan="3" style="text-align:center;border-left-style:solid;border-left-width:1px">{subset}</th>'
    row2 += "<th style='border-left-style:solid;border-left-width:1px'>" + "</th><th>".join(METRICS) + "</th>"
    row3 += "<td style='border-left-style:solid;border-left-width:1px'>" + "</td><td>".join([str(round(results_silver[subset][k], 2)) for k in METRICS])  + "</td>"

display(HTML(
   f"<table><thead><tr>{row1}</tr><tr>{row2}</tr></thead><tbody><tr>{row3}</tr></tbody></table>"
))


# Pretty print bbox detection results
from IPython.display import HTML, display

row1, row2, row3 = "", "", ""
for subset in results_silver_bbox:
    row1 += f'<th colspan="3" style="text-align:center;border-left-style:solid;border-left-width:1px">{subset}</th>'
    row2 += "<th style='border-left-style:solid;border-left-width:1px'>" + "</th><th>".join(METRICS) + "</th>"
    row3 += "<td style='border-left-style:solid;border-left-width:1px'>" + "</td><td>".join([str(round(results_silver_bbox[subset][k], 2)) for k in METRICS])  + "</td>"

display(HTML(
   f"<table><thead><tr>{row1}</tr><tr>{row2}</tr></thead><tbody><tr>{row3}</tr></tbody></table>"
))
