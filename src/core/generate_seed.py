import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.morphology import binary_dilation, footprint_rectangle
from skimage.segmentation import flood
from utils.io import load_raster
import rasterio
import json
import os


# region growing -> foreground seeds
def generate_foreground_seeds(conf_map, **kwargs):
    sigma = kwargs["sigma"]
    window_size = kwargs["window_size"]
    stride = kwargs["stride"]
    threshold = kwargs["threshold"]
    delta = kwargs["delta"]

    smoothed = gaussian_filter(conf_map, sigma=sigma)
    seeds = np.zeros_like(smoothed, dtype=bool)
    h, w = smoothed.shape

    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            patch = smoothed[y:y+window_size, x:x+window_size]
            max_val = np.max(patch)
            if max_val < threshold:
                continue
            max_idx = np.unravel_index(np.argmax(patch), patch.shape)
            seed_y, seed_x = y + max_idx[0], x + max_idx[1]
            seeds[seed_y, seed_x] = True

    seed_coords = np.column_stack(np.nonzero(seeds))

    maxima_result = np.zeros_like(smoothed, dtype=np.uint8)
    for y, x in seed_coords:
        maxima_result[y, x] = 1

    maxima_result = binary_dilation(
        maxima_result,
        footprint=footprint_rectangle((5, 5))
    ).astype(np.uint8)
    fg_seed = np.zeros_like(conf_map, dtype=np.int32)
    visited = np.zeros_like(conf_map, dtype=bool)
    for y, x in seed_coords:
        if visited[y, x]:
            continue

        mask = flood(smoothed, (y, x), tolerance=delta)
        mask = mask & (~visited)

        if np.sum(mask) < 5:
            continue

        fg_seed[mask] = 1
        visited[mask] = True
    return maxima_result, fg_seed


# 5. Background seed = 전경 바깥에 spacing 간격으로 격자 찍기
def generate_background_seeds(fg_seed, **kwargs):
    bg_spacing = kwargs.get("bg_spacing", 20)
    h, w = fg_seed.shape
    bg_seed_mask = np.zeros_like(fg_seed, dtype=np.uint8)

    inverse_fg = fg_seed == 0
    for y in range(0, h, bg_spacing):
        step_index = (y // bg_spacing) % 5  # 0~4 반복
        offset = step_index * (bg_spacing // 5)
        for x in range(offset, w, bg_spacing):
            if inverse_fg[y, x]:
                bg_seed_mask[y, x] = 1
    return bg_seed_mask


def run_generate_seed(conf_map, **kwargs):
    maxima_result, fg_seed = generate_foreground_seeds(conf_map, **kwargs)
    bg_seed = generate_background_seeds(fg_seed, **kwargs)

    ############ 임시 #################
    conf_path = r"E:\chong_convert_onnx\convert_onnx\workspace\data\confidence_map\test\conf_map.tif"
    conf_map, transform, crs = load_raster(conf_path)

    # 3. 저장 경로 하드코딩
    out_dir = r"E:\chong_convert_onnx\convert_onnx\workspace\data\Z_research_temp\seed_debug"
    os.makedirs(out_dir, exist_ok=True)

    maxima_path = os.path.join(out_dir, "maxima_result.tif")
    fg_path = os.path.join(out_dir, "fg_seed.tif")
    bg_path = os.path.join(out_dir, "bg_seed.tif")

    # 4. 저장 프로파일
    profile = {
        'driver': 'GTiff',
        'dtype': rasterio.uint8,
        'count': 1,
        'height': conf_map.shape[0],
        'width': conf_map.shape[1],
        'transform': transform,
        'crs': crs
    }

    # 5. 저장
    with rasterio.open(maxima_path, 'w', **profile) as dst:
        dst.write(maxima_result.astype(rasterio.uint8), 1)
    with rasterio.open(fg_path, 'w', **profile) as dst:
        dst.write(fg_seed.astype(rasterio.uint8), 1)
    with rasterio.open(bg_path, 'w', **profile) as dst:
        dst.write(bg_seed.astype(rasterio.uint8), 1)

    return fg_seed, bg_seed
