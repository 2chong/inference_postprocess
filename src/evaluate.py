import os
import geopandas as gpd
import numpy as np
import pandas as pd
from utils.io import load_raster, load_vector, export_csv, in_dir
from utils.mask_to_vector import mask_to_vector
from core.matching_tool import matching_pipeline
from core.eval_tool import evaluate_summary_table


def load_data(gt_path: str, result_path: str, min_area: float = 10):
    gt = load_vector(gt_path)
    result_path = in_dir(result_path)
    mask, transform, crs = load_raster(result_path)
    result = mask_to_vector(mask, transform, crs=crs, min_area=min_area)
    result = result.to_crs(epsg=5186)
    return gt, result


def generate_vectorized_thresholds(conf_map_path):
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    conf_map_path = in_dir(conf_map_path)
    conf_map, transform, crs = load_raster(conf_map_path)
    vectorized_results = {}

    for thr in thresholds:
        binary_mask = (conf_map >= thr).astype(np.uint8)
        vec = mask_to_vector(binary_mask, transform, crs=crs, min_area=10)
        vec = vec.to_crs(epsg=5186)
        vectorized_results[thr] = vec

    return vectorized_results


def matching(gt, result):
    return matching_pipeline(gt, result)


def cal_metrics(method, gt, result, component):
    return evaluate_summary_table(method, gt, result, component, iou_thresholds=np.arange(0.5, 1.0, 0.05))


def run_baseline_evaluate(gt_path, conf_map_path):
    gt = load_vector(gt_path).to_crs(epsg=5186)
    threshold_results = generate_vectorized_thresholds(conf_map_path)

    all_summaries = []
    for thr, result in threshold_results.items():
        component, matched_gt, matched_result = matching(gt, result)
        method = f"thresholding@{thr}"
        summary = cal_metrics(method, matched_gt, matched_result, component)
        all_summaries.append(summary)

    combined = pd.concat(all_summaries, ignore_index=True)
    return combined


def run_custom_evaluate(method, gt_path, result_path):
    gt = load_vector(gt_path).to_crs(epsg=5186)

    result_path = in_dir(result_path)
    mask, transform, crs = load_raster(result_path)
    result = mask_to_vector(mask, transform, crs=crs, min_area=10)
    result = result.to_crs(epsg=5186)

    component, matched_gt, matched_result = matching(gt, result)
    summary = cal_metrics(method, matched_gt, matched_result, component)
    return summary


def run(gt_path, result_path, conf_map_path, output_path, method):
    baseline_summary = run_baseline_evaluate(gt_path, conf_map_path)
    custom_summary = run_custom_evaluate(method, gt_path, result_path)
    final_summary = pd.concat([baseline_summary, custom_summary], ignore_index=True)
    filename = f"{method}_evaluation.csv"
    full_path = os.path.join(output_path, filename)

    export_csv(final_summary, full_path)


if __name__ == "__main__":
    method_name = 'graph_cut'
    ground_truth = "data/groundtruth/test"
    postprocess_result = f"data/postprocess_result/{method_name}/test"
    confidence_map = "data/confidence_map/test"
    output = "evaluation/test"
    run(ground_truth, postprocess_result, confidence_map, output, method_name)
