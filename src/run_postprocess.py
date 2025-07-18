# run_postprocessing.py

import os
import glob
from core.postprocess import apply_postprocessing
from utils.io import load_raster, save_raster


def get_params_string(method_kwargs):
    """
    Generate a string representation of the parameters for the filename.

    Args:
        method_kwargs: Dictionary of parameters

    Returns:
        String representation of parameters
    """
    # Select key parameters to include in the filename
    key_params = ["mode", "lambda_param", "n_iterations", "n_components", "connectivity", "beta"]

    # Parameters to exclude (paths, etc.)
    exclude_params = ["conf_map_path", "gt_path"]

    # Build the parameter string
    param_parts = []
    for key, value in method_kwargs.items():
        # Skip excluded parameters and None values
        if key in exclude_params or value is None:
            continue

        # Include key parameters and any other numerical or string parameters
        if key in key_params or isinstance(value, (int, float, str)):
            param_parts.append(f"{key}={value}")

    return "_".join(param_parts)

def run_postprocessing(method, method_kwargs=None):
    """
    Run postprocessing on confidence maps.

    Args:
        method: Postprocessing method
        method_kwargs: Additional parameters for the specific method
    """
    input_dir = "data/confidence_map/test_av"
    output_dir = f"data/postprocess_result/{method}/test9"
    os.makedirs(output_dir, exist_ok=True)

    conf_paths = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    method_kwargs = method_kwargs or {}

    if len(conf_paths) == 1:
        tile_path = conf_paths[0]
        conf_map, transform, crs = load_raster(tile_path)

        # Add conf_map_path to method_kwargs if using graph_cut
        if method == 'graph_cut':
            method_kwargs['conf_map_path'] = tile_path

        result = apply_postprocessing(conf_map, method=method, **method_kwargs)

        # Generate parameter string for filename
        params_str = get_params_string(method_kwargs)

        # Save result
        save_path = os.path.join(output_dir, f"postprocess_result_{params_str}.tif")
        save_raster(result, transform, crs, save_path)

    else:
        for tile_path in conf_paths:
            tile_id = os.path.basename(tile_path).replace(".tif", "").split("_")[-1]

            conf_map, transform, crs = load_raster(tile_path)

            # Add conf_map_path to method_kwargs if using graph_cut
            if method == 'graph_cut':
                method_kwargs['conf_map_path'] = tile_path

            result = apply_postprocessing(conf_map, method=method, **method_kwargs)

            # Generate parameter string for filename
            params_str = get_params_string(method_kwargs)

            # Save result
            save_path = os.path.join(output_dir, f"postprocess_result_tile_{tile_id}_{params_str}.tif")
            save_raster(result, transform, crs, save_path)


if __name__ == "__main__":
    method = 'graph_cut'

    params = {
        "mode": "manual",               # Threshold setting mode: 'manual', 'gmm'
        "th1": 0.1,                  # Lower threshold for background (required for manual mode)
        "th2": 0.9,                  # Upper threshold for foreground (required for manual mode)
        "lambda_param": 100,         # Weight for smoothness term (typical values: 50-200)
        "n_iterations": 3,           # Number of iterations for graph cut
        "n_components": 1,           # Number of GMM components
        "connectivity": 8,           # Connectivity type (4 or 8)
        "beta": 1                  # Parameter controlling the sensitivity to intensity differences
    }

    run_postprocessing(method=method, method_kwargs=params)
