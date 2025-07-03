import os
from typing import Tuple
import geopandas as gpd
import numpy as np

from utils.io import load_raster, load_vector, export_csv, in_dir
from utils.mask_to_vector import mask_to_vector
from core.matching_tool import matching_pipeline
from core.eval_tool import evaluate_summary_table


def load_data(gt_path: str, result_path: str, min_area: float = 10) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    gt = load_vector(gt_path)

    result_path = in_dir(result_path)
    mask, transform, crs = load_raster(result_path)
    result = mask_to_vector(mask, transform, crs=crs, min_area=min_area)
    result = result.to_crs(epsg=5186)
    return gt, result


def matching(gt, result):
    return matching_pipeline(gt, result)


def cal_metrics(method, gt, result, component):
    return evaluate_summary_table(method, gt, result, component, iou_thresholds=np.arange(0.5, 1.0, 0.05))


def run_evaluate(gt_path, result_path, output_path, method):
    gt, result = load_data(gt_path, result_path)
    component, gt, result = matching(gt, result)
    output_path = os.path.join(output_path, f"{method}_summary.csv")
    summary = cal_metrics(method, gt, result, component)
    export_csv(summary, output_path)


if __name__ == "__main__":
    method_name = 'morph_close'
    ground_truth = "data/groundtruth/test"
    postprocess_result = f"data/postprocess_result/test/{method_name}"
    output = "evaluation/test"
    run_evaluate(ground_truth, postprocess_result, output, method_name)
