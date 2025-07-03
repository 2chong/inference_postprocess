# postprocess.py

import numpy as np
import cv2


def thresholding(conf_map, thresh=0.5):
    return (conf_map > thresh).astype(np.uint8)


def morph_close(conf_map, thresh=0.5, kernel_size=5):
    binary = thresholding(conf_map, thresh)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def connected_components_filter(conf_map, thresh=0.5, min_area=100):
    binary = thresholding(conf_map, thresh)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            binary[labels == i] = 0
    return binary


def apply_postprocessing(conf_map, method='threshold', **kwargs):
    if method == 'threshold':
        return thresholding(conf_map, **kwargs)
    elif method == 'morph_close':
        return morph_close(conf_map, **kwargs)
    elif method == 'connected_components':
        return connected_components_filter(conf_map, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
