# postprocess.py

import numpy as np
import cv2
import maxflow
from scipy.ndimage import sobel
from scipy.ndimage import gaussian_filter
from core.graph_cut import graph_cut_segmentation


def thresholding(conf_map, thresh=0.5):
    return (conf_map > thresh).astype(np.uint8)


def morphology_to_mask(mask, open_k=5, close_k=5, iterations=1):
    mask = thresholding(mask, 0.5)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=iterations)
    morphed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=iterations)
    return morphed


def connected_components_filter(conf_map, thresh=0.5, min_area=100):
    binary = thresholding(conf_map, thresh)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            binary[labels == i] = 0
    return binary


def graph_cut(conf_map, **kwargs):
    """
    Run graph cut segmentation using the standard formula with threshold-based seed generation.

    Args:
        conf_map: Confidence map
        **kwargs: Additional parameters
            - mode: Threshold setting mode ('gt', 'gmm', or 'kmeans', default: 'gmm')
            - lambda_param: Weight for smoothness term (default: 100)
            - n_iterations: Number of iterations (default: 3)
            - n_components: Number of GMM components (default: 5)
            - conf_map_path: Path to the confidence map (optional, used to infer gt_path and get transform/crs)
            - connectivity: Connectivity type (4 or 8, default: 4)
            - beta: Parameter controlling the sensitivity to intensity differences (default: 30)

    Returns:
        Binary segmentation mask
    """
    # Extract parameters
    lambda_param = kwargs.get("lambda_param", 100)
    n_iterations = kwargs.get("n_iterations", 3)
    n_components = kwargs.get("n_components", 5)
    mode = kwargs.get("mode", "gmm")
    connectivity = kwargs.get("connectivity", 4)
    beta = kwargs.get("beta", 30)

    # Run graph cut segmentation
    # Filter out parameters that are already explicitly passed
    filtered_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['conf_map_path', 'mode', 'n_iterations', 'n_components', 'lambda_param', 'connectivity', 'beta']}

    result = graph_cut_segmentation(
        conf_map,
        mode=mode,
        n_iterations=n_iterations,
        n_components=n_components,
        lambda_param=lambda_param,
        connectivity=connectivity,
        beta=beta,
        **filtered_kwargs  # Pass only additional parameters
    )

    return result


def apply_postprocessing(conf_map, method, **kwargs):
    """
    Apply postprocessing to confidence map.

    Args:
        conf_map: Confidence map
        method: Postprocessing method
        **kwargs: Additional parameters for the specific method

    Returns:
        Binary mask
    """
    if method == 'threshold':
        return thresholding(conf_map, **kwargs)
    elif method == 'morphology':
        return morphology_to_mask(conf_map, **kwargs)
    elif method == 'connected_components':
        return connected_components_filter(conf_map, **kwargs)
    elif method == 'graph_cut':
        return graph_cut(conf_map, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
