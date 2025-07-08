# run_postprocessing.py

import os
import glob
from core.postprocess import apply_postprocessing
from utils.io import load_raster, save_raster


def run_postprocessing(method, method_kwargs=None):
    input_dir = "data/confidence_map/test"
    output_dir = f"data/postprocess_result/{method}/test"
    os.makedirs(output_dir, exist_ok=True)

    conf_paths = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    method_kwargs = method_kwargs or {}

    if len(conf_paths) == 1:
        tile_path = conf_paths[0]
        conf_map, transform, crs = load_raster(tile_path)

        binary = apply_postprocessing(conf_map, method=method, **method_kwargs)

        save_path = os.path.join(output_dir, f"postprocess_result.tif")
        save_raster(binary, transform, crs, save_path)

    else:
        for tile_path in conf_paths:
            tile_id = os.path.basename(tile_path).replace(".tif", "").split("_")[-1]

            conf_map, transform, crs = load_raster(tile_path)
            binary = apply_postprocessing(conf_map, method=method, **method_kwargs)

            save_path = os.path.join(output_dir, f"postprocess_result_tile_{tile_id}.tif")
            save_raster(binary, transform, crs, save_path)


if __name__ == "__main__":
    method = 'graph_cut'

    params = {
        "edge_sigma": 0.1,
        "sigma": 2,
        "window_size": 120,
        "stride": 100,
        "threshold": 0.4,
        "delta": 0.1,
        "bg_spacing": 20
    }

    run_postprocessing(method=method, method_kwargs=params)
