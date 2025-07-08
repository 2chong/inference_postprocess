import maxflow
import numpy as np
from scipy.ndimage import sobel


def run_graphcut(conf_map, fg_mask, bg_mask, sigma=0.1, decay_factor=0.2):
    h, w = conf_map.shape
    num_pixels = h * w

    # 1. 그래프 생성
    g = maxflow.Graph[float](num_pixels, num_pixels * 4)
    node_ids = g.add_grid_nodes((h, w))

    # 2. Gradient 기반 edge weight 계산
    grad_x = sobel(conf_map, axis=1)
    grad_y = sobel(conf_map, axis=0)
    gradient_magnitude = np.hypot(grad_x, grad_y)
    weights = np.exp(-(gradient_magnitude ** 2) / (sigma ** 2))

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
