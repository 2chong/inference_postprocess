import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter, maximum_filter, label
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import white_tophat, disk
from skimage.segmentation import flood
from skimage.feature import blob_log
from scipy.ndimage import binary_dilation
from skimage.morphology import square

import os
import json


def cluster_with_tophat(conf_map, radius=10, threshold=0.05, min_size=10):
    # 1. Top-hat filtering
    filtered = white_tophat(conf_map, footprint=disk(radius))  # ← 여기 수정됨

    # 2. Thresholding
    mask = filtered > threshold
    if np.sum(mask) == 0:
        return np.zeros_like(conf_map, dtype=np.uint8)

    # 3. Connected component labeling
    labeled, num = label(mask)

    # 4. 너무 작은 클러스터 제거
    output = np.zeros_like(conf_map, dtype=np.uint8)
    for i in range(1, num + 1):
        region = (labeled == i)
        if np.sum(region) >= min_size:
            output[region] = 1

    return output


def extract_dense_local_maxima(conf_map, window_size=15, stride=7, threshold=0.3):
    seeds = np.zeros_like(conf_map, dtype=bool)
    h, w = conf_map.shape

    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            patch = conf_map[y:y+window_size, x:x+window_size]
            max_val = np.max(patch)
            if max_val < threshold:
                continue  # 너무 어두운 patch는 무시

            max_idx = np.unravel_index(np.argmax(patch), patch.shape)
            seed_y, seed_x = y + max_idx[0], x + max_idx[1]
            seeds[seed_y, seed_x] = True

    return np.column_stack(np.nonzero(seeds))


def cluster_with_region_growing(conf_map, sigma=2, window_size=15, stride=7,
                                threshold_abs=0.3, delta=0.1):
    # 1. Gaussian Smoothing
    smoothed = gaussian_filter(conf_map, sigma=sigma)

    # 2. Overlapping window-based seed extraction
    seed_coords = extract_dense_local_maxima(smoothed,
                                             window_size=window_size,
                                             stride=stride,
                                             threshold=threshold_abs)

    seed_mask = np.zeros_like(conf_map, dtype=np.uint8)
    for y, x in seed_coords:
        seed_mask[y, x] = 1
    seed_mask = binary_dilation(seed_mask, structure=square(5)).astype(np.uint8)

    # 저장 경로 하드코딩
    seed_save_path = r"E:\chong_convert_onnx\convert_onnx\workspace\data\Z_research_temp\seed_debug\seed_coords.tif"
    reference_path = r"E:\chong_convert_onnx\convert_onnx\workspace\data\confidence_map\test\conf_map.tif"

    with rasterio.open(reference_path) as src:
        profile = src.profile
        profile.update(dtype=rasterio.uint8, count=1)

        with rasterio.open(seed_save_path, 'w', **profile) as dst:
            dst.write(seed_mask, 1)

    print(f"[INFO] seed mask saved to: {seed_save_path}")
    cluster_map = np.zeros_like(conf_map, dtype=np.int32)
    cluster_id = 1
    visited = np.zeros_like(conf_map, dtype=bool)

    for y, x in seed_coords:
        if visited[y, x]:
            continue

        mask = flood(smoothed, (y, x), tolerance=delta)
        mask = mask & (~visited)

        if np.sum(mask) < 5:
            continue

        cluster_map[mask] = cluster_id
        visited[mask] = True
        cluster_id += 1

    return (cluster_map > 0).astype(np.uint8)


def cluster_with_watershed(conf_map, sigma=2, contrast_thresh=0.05, min_distance=5):
    # 1. contrast 계산
    blurred = gaussian_filter(conf_map, sigma=sigma)
    contrast = conf_map - blurred

    # 2. contrast 기반 마스크
    seed_mask = contrast > contrast_thresh
    if np.sum(seed_mask) == 0:
        return np.zeros_like(conf_map, dtype=np.uint8)

    # 3. 로컬 최대값 좌표 추출 (최신 버전에서는 indices=True가 기본값)
    coordinates = peak_local_max(contrast, min_distance=min_distance, labels=seed_mask)

    # 4. 좌표를 마커 이미지로 변환
    markers = np.zeros_like(conf_map, dtype=np.int32)
    for i, (y, x) in enumerate(coordinates, start=1):
        markers[y, x] = i

    # 5. Watershed
    elevation = -contrast
    labels = watershed(elevation, markers=markers, mask=seed_mask)

    return (labels > 0).astype(np.uint8)


def cluster_relative_peaks(conf_map, sigma=3, contrast_thresh=0.05, min_size=10):
    # 1. 주변 대비 계산
    blurred = gaussian_filter(conf_map, sigma=sigma)
    contrast = conf_map - blurred

    # 2. contrast 기반 마스크 생성
    mask = contrast > contrast_thresh
    if np.sum(mask) == 0:
        return np.zeros_like(conf_map, dtype=np.uint8)

    # 3. 연결된 영역 라벨링
    labeled, num = label(mask)

    # 4. 너무 작은 클러스터 제거
    output = np.zeros_like(conf_map, dtype=np.uint8)
    for i in range(1, num + 1):
        region = (labeled == i)
        if np.sum(region) >= min_size:
            output[region] = 1

    return output


def cluster_with_log_blob(conf_map, min_sigma=2, max_sigma=10, num_sigma=10,
                          threshold=0.03, scale_factor=1.5, min_size=10):
    blobs = blob_log(conf_map, min_sigma=min_sigma, max_sigma=max_sigma,
                     num_sigma=num_sigma, threshold=threshold)

    # 결과 mask
    mask = np.zeros_like(conf_map, dtype=np.uint8)
    for y, x, sigma in blobs:
        r = sigma * scale_factor
        rr, cc = np.ogrid[:conf_map.shape[0], :conf_map.shape[1]]
        circle = (rr - int(y))**2 + (cc - int(x))**2 <= r**2
        mask[circle] = 1

    # 연결된 blob 정리
    from scipy.ndimage import label
    labeled, num = label(mask)

    output = np.zeros_like(conf_map, dtype=np.uint8)
    for i in range(1, num + 1):
        region = (labeled == i)
        if np.sum(region) >= min_size:
            output[region] = 1

    return output


def run_clustering(conf_map, method, **kwargs):
    if method == "region_growing":
        return cluster_with_region_growing(conf_map,
                                           sigma=kwargs.get("sigma", 2),
                                           window_size=kwargs.get("window_size", 15),
                                           stride=kwargs.get("stride", 7),
                                           threshold_abs=kwargs.get("threshold_abs", 0.3),
                                           delta=kwargs.get("delta", 0.1))

    elif method == "watershed":
        return cluster_with_watershed(conf_map,
                                      sigma=kwargs.get("sigma", 2),
                                      contrast_thresh=kwargs.get("contrast_thresh", 0.05),
                                      min_distance=kwargs.get("min_distance", 5))
    elif method == "relative_peaks":
        return cluster_relative_peaks(conf_map,
                                      sigma=kwargs.get("sigma", 3),
                                      contrast_thresh=kwargs.get("contrast_thresh", 0.05),
                                      min_size=kwargs.get("min_size", 10))
    elif method == "tophat":
        return cluster_with_tophat(conf_map,
                                   radius=kwargs.get("radius", 10),
                                   threshold=kwargs.get("threshold", 0.05),
                                   min_size=kwargs.get("min_size", 10))
    elif method == "log_blob":
        return cluster_with_log_blob(conf_map,
                                     min_sigma=kwargs.get("min_sigma", 2),
                                     max_sigma=kwargs.get("max_sigma", 10),
                                     num_sigma=kwargs.get("num_sigma", 10),
                                     threshold=kwargs.get("threshold", 0.03),
                                     scale_factor=kwargs.get("scale_factor", 1.5),
                                     min_size=kwargs.get("min_size", 10))
    else:
        raise ValueError(f"Unsupported clustering method: {method}")


def save_cluster_map(cluster_map, reference_path, save_path, params=None):
    with rasterio.open(reference_path) as src:
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width

    # GeoTIFF 저장
    with rasterio.open(
        save_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=rasterio.int32,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(cluster_map, 1)

    # 파라미터 + 클러스터 개수 저장
    if params is not None:
        json_path = os.path.splitext(save_path)[0] + "_params.json"
        num_clusters = int(np.max(cluster_map))  # 라벨 1부터 시작한 경우

        save_data = dict(params)  # params 복사
        save_data["num_clusters"] = num_clusters

        with open(json_path, "w") as f:
            json.dump(save_data, f, indent=4)


def select_method_by_input():
    print("=== 클러스터링 메서드를 선택하세요 ===")
    print("1. Region Growing")
    print("2. Watershed")
    print("3. Relative Peaks")
    print("4. Top-hat Morphology")
    print("5. LoG Blob Detection")

    option = input("번호를 입력하세요 (1~5): ").strip()

    if option == "1":
        return params_region_growing
    elif option == "2":
        return params_watershed
    elif option == "3":
        return params_relative_peaks
    elif option == "4":
        return params_tophat
    elif option == "5":
        return params_log_blob
    else:
        print("잘못된 입력입니다. 기본값 Region Growing으로 진행합니다.")
        return params_region_growing


# === 사용 예시 ===
# Region Growing 방식
params_region_growing = {
    "method": "region_growing",
    "sigma": 2,
    "window_size": 120,
    "stride": 100,
    "threshold_abs": 0.4,
    "delta": 0.1
}


params_watershed = {
    "method": "watershed",
    "sigma": 3,
    "contrast_thresh": 0.05,
    "min_distance": 5
}

params_relative_peaks = {
    "method": "relative_peaks",
    "sigma": 3,
    "contrast_thresh": 0.05,
    "min_size": 10
}

params_tophat = {
    "method": "tophat",
    "radius": 10,
    "threshold": 0.05,
    "min_size": 10
}

params_log_blob = {
    "method": "log_blob",
    "min_sigma": 20,
    "max_sigma": 40,
    "num_sigma": 10,
    "threshold": 0.03,
    "scale_factor": 1.5,
    "min_size": 10
}

params = select_method_by_input()


# 2. 경로 구성: method 이름이 중간 폴더에 포함되도록
base_root_dir = r"E:\chong_convert_onnx\convert_onnx\workspace\data\Z_research_temp\clustering_result"
method_dir = params["method"]
output_dir = os.path.join(base_root_dir, method_dir, "test")
os.makedirs(output_dir, exist_ok=True)


# 3. 파일명 생성 함수
def make_filename_from_params(base_name, params):
    method_short = {
        "region_growing": "rg",
        "watershed": "ws",
        "relative_peaks": "rp",
        "tophat": "th",
        "log_blob": "log"
    }
    method = method_short.get(params["method"], "unknown")
    parts = [f"{method}"]

    if "sigma" in params:
        parts.append(f"sigma{params['sigma']}")
    if "threshold_abs" in params:
        parts.append(f"thr{params['threshold_abs']}")
    if "delta" in params:
        parts.append(f"delta{params['delta']}")
    if "window_size" in params:
        parts.append(f"w{params['window_size']}")
    if "stride" in params:
        parts.append(f"s{params['stride']}")
    if "contrast_thresh" in params:
        parts.append(f"cthr{params['contrast_thresh']}")
    if "min_distance" in params:
        parts.append(f"md{params['min_distance']}")
    if "min_size" in params:
        parts.append(f"msz{params['min_size']}")
    if "radius" in params:
        parts.append(f"r{params['radius']}")
    if "min_sigma" in params:
        parts.append(f"mins{params['min_sigma']}")
    if "max_sigma" in params:
        parts.append(f"maxs{params['max_sigma']}")
    if "threshold" in params:
        parts.append(f"thr{params['threshold']}")
    if "scale_factor" in params:
        parts.append(f"sf{params['scale_factor']}")
    name = base_name + "_" + "_".join(parts)
    return name


# 4. 파일명 + 전체 경로 완성
base_name = "cluster"
file_name = make_filename_from_params(base_name, params)
output_path = os.path.join(output_dir, f"{file_name}.tif")

# 5. Confidence map 불러오기
conf_map_path = r"E:\chong_convert_onnx\convert_onnx\workspace\data\confidence_map\test\conf_map.tif"
with rasterio.open(conf_map_path) as src:
    conf_map = src.read(1)

# 6. 클러스터링 및 저장
cluster_map = run_clustering(conf_map, **params)
save_cluster_map(cluster_map, conf_map_path, output_path, params=params)