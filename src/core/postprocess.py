# postprocess.py

import numpy as np
import cv2
import maxflow
from scipy.ndimage import sobel
from core.generate_seed import run_generate_seed


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
    fg_mask, bg_mask = run_generate_seed(conf_map, **kwargs)
    h, w = conf_map.shape
    num_pixels = h * w
    edge_sigma = kwargs.get("edge_sigma", 0.1)

    # 1. 그래프 생성
    g = maxflow.Graph[float](num_pixels, num_pixels * 4)
    node_ids = g.add_grid_nodes((h, w))

    # 2. Gradient 기반 edge weight 계산
    grad_x = sobel(conf_map, axis=1)
    grad_y = sobel(conf_map, axis=0)
    gradient_magnitude = np.hypot(grad_x, grad_y)
    weights = np.exp(-(gradient_magnitude ** 2) / (edge_sigma ** 2))

    # 3. 각 픽셀 간 edge 연결 (4-neighbor)
    structure = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]])

    g.add_grid_edges(node_ids, weights=weights, structure=structure, symmetric=True)

    # 4. Source/Sink 연결 (seed 사용)
    # 전경 seed → source
    g.add_grid_tedges(node_ids, fg_mask.astype(bool), 0)

    # 배경 seed → sink
    g.add_grid_tedges(node_ids, 0, bg_mask.astype(bool))

    # 5. Min-cut 수행
    g.maxflow()
    segmentation = g.get_grid_segments(node_ids)

    # 6. 결과 처리
    cut_mask = np.logical_not(segmentation).astype(np.uint8)  # foreground = 1

    return cut_mask


def apply_postprocessing(conf_map, method='threshold', **kwargs):
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
