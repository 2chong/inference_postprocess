import argparse
import os
import rasterio
import numpy as np
import rasterio.features
import json
import cv2
import geopandas as gpd
import math
import warnings
import onnxruntime as ort
import networkx as nx
import pandas as pd
import time


from rasterio.windows import transform as window_transform
from more_itertools import chunked
from multiprocessing import Pool
from shapely.ops import unary_union
from rasterio.windows import Window
from shapely.geometry import shape
from rasterio.features import shapes
from rasterio.warp import transform
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")


def make_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type=str, default="data/orthoimage", help="input folder containing T1/, T2/")
    parser.add_argument("-m", "--model", type=str, default="model", help="Model folder containing .onnx")
    parser.add_argument("-o", "--output", type=str, default="inf_status", help="Output folder")

    parser.add_argument("-c", "--conf-threshold", type=float, default=None)
    parser.add_argument("-r", "--resolution", type=float, default=None)
    parser.add_argument("--classes", type=str, default=None)
    parser.add_argument("-t", "--max-threads", type=int, default=None)
    parser.add_argument("-b", "--batch-size", type=int, default=8)

    parser.add_argument("--cut-threshold", type=float, default=0.05)
    parser.add_argument("--cd-threshold", type=float, default=0.7)

    return parser


def resolve_paths(args):
    # t1_path = os.path.join(args.input, "T1")
    # t1_vec = [f for f in os.listdir(t1_path) if f.endswith(('.shp', '.geojson'))]
    # if not t1_vec:
    #     raise FileNotFoundError("No vector file found in input/T1/")
    # args.prev_gdf = os.path.join(t1_path, t1_vec[0])

    t2_path = os.path.join(args.input, "test")
    t2_tifs = [f for f in os.listdir(t2_path) if f.endswith(".tif")]
    if not t2_tifs:
        raise FileNotFoundError("No .tif file found in input/T2/")
    args.geotiff = os.path.join(t2_path, t2_tifs[0])

    model_files = [f for f in os.listdir(args.model) if f.endswith(".onnx")]
    if not model_files:
        raise FileNotFoundError("No .onnx model found in model folder")
    args.model = os.path.join(args.model, model_files[0])

    return args



class ProgressBar:
    def __init__(self):
        self.pbar = tqdm(total=100, desc="Start")
        self.current = 0
        self.closed = False

    def update(self, text, perc=0):
        self.current += perc
        self.pbar.n = int(self.current)
        self.pbar.set_description(text)
        self.pbar.refresh()

        if self.current >= 100 and not self.closed:
            self.pbar.set_description("End")
            self.pbar.close()
            self.closed = True

    @staticmethod
    def write(text):
        tqdm.write(text)


class StatusManager:
    def __init__(self, dir_path="."):
        self.filepath = os.path.join(dir_path, "status.json")
        self.status = {
            "CurrentTask": "init",
            "ElapsedTime": {},
            "Error": None
        }

    def set_task(self, task_name):
        self.status["CurrentTask"] = task_name
        self.save()

    def record_time(self, label, seconds):
        self.status["ElapsedTime"][label] = f"{round(seconds, 2)}s"
        self.save()

    def log_error(self, msg):
        self.status["CurrentTask"] = "error"
        self.status["Error"] = msg
        self.save()

    def save(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True) if os.path.dirname(self.filepath) else None
        with open(self.filepath, "w") as f:
            json.dump(self.status, f, ensure_ascii=False, indent=4)

    def task(self, name):
        return _StatusTaskContext(self, name)


class _StatusTaskContext:
    def __init__(self, manager, name):
        self.manager = manager
        self.name = name

    def __enter__(self):
        self.manager.set_task(self.name)
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        self.manager.record_time(self.name, elapsed)
        if exc_type:
            self.manager.log_error(str(exc_val))


def import_shapefile(file_path, crs=5186):
    if os.path.isdir(file_path):
        vec_files = [f for f in os.listdir(file_path) if f.endswith(('.shp', '.geojson'))]
        if not vec_files:
            raise FileNotFoundError(f"No .shp or .geojson file found in directory: {file_path}")
        file_path = os.path.join(file_path, vec_files[0])  # 첫 번째 vector 파일 사용

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in [".shp", ".geojson"]:
        raise ValueError("Only .shp or .geojson")

    gdf = gpd.read_file(str(file_path))
    if gdf.crs != f"epsg:{crs}":
        gdf = gdf.to_crs(epsg=crs)

    return gdf


"""
1. TIF -> 건물 추론 모델 -> GDF
"""


# 1) 모델 및 설정 불러오기
def get_model_file(path):
    # 모델 경로 반환
    if os.path.isfile(path):
        return os.path.abspath(path)
    else:
        raise FileNotFoundError(f"Model file not found: {path}")


def create_session(model_file, max_threads=None):
    # ONNX 모델 로드 및 config 생성
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    if max_threads is not None:
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = max_threads

    providers = [
        'CUDAExecutionProvider',
        'CPUExecutionProvider'
    ]

    session = ort.InferenceSession(model_file, sess_options=options, providers=providers)
    inputs = session.get_inputs()
    if len(inputs) > 1:
        raise Exception("ONNX model: unsupported number of inputs")

    meta = session.get_modelmeta().custom_metadata_map

    config = {
        'det_type': json.loads(meta.get('det_type', '"YOLO_v5_or_v7_default"')),
        'det_conf': float(meta.get('det_conf', 0.3)),
        'det_iou_thresh': float(meta.get('det_iou_thresh', 0.1)),
        'classes': ['background', 'building'],
        'seg_thresh': float(meta.get('seg_thresh', 0.5)),
        'seg_small_segment': int(meta.get('seg_small_segment', 11)),
        'resolution': float(20),
        'class_names': json.loads(meta.get('class_names', '{}')),
        'model_type': json.loads(meta.get('model_type', '"Detector"')),
        'tiles_overlap': float(meta.get('tiles_overlap', 50)),  # percentage
        'tiles_size': inputs[0].shape[-1],
        'input_shape': inputs[0].shape,
        'input_name': inputs[0].name,
    }
    return session, config


def override_config(config, conf_threshold=None, resolution=None, classes=None):
    # 사용자 입력이 있다면 우선 순위
    if conf_threshold is not None:
        config['det_conf'] = conf_threshold
    if resolution is not None:
        config['resolution'] = resolution
    if classes is not None:
        cn_map = cls_names_map(config['class_names'])
        config['classes'] = [cn_map[cls_name] for cls_name in cn_map if cls_name in classes]
    return config


def cls_names_map(class_names):
    # {"0": "tree"} --> {"tree": 0}
    d = {}
    for i in class_names:
        d[class_names[i]] = int(i)
    return d


# 2) Import TIF, edit config
def load_raster(geotiff_path: str):
    raster = rasterio.open(geotiff_path, 'r')
    return raster


def get_input_resolution(raster) -> float:
    # tif의 transform 기반 해상도 계산
    input_res = round(max(abs(raster.transform[0]), abs(raster.transform[4])), 4) * 100
    if input_res <= 0:
        input_res = estimate_raster_resolution(raster)
    return input_res


def estimate_raster_resolution(raster):
    # transform 정보가 없을 경우 해상도 추정
    if raster.crs is None:
        return 10  # Wild guess cm/px

    bounds = raster.bounds
    width = raster.width
    height = raster.height
    crs = raster.crs
    res_x = (bounds.right - bounds.left) / width
    res_y = (bounds.top - bounds.bottom) / height

    if crs.is_geographic:
        center_lat = (bounds.top + bounds.bottom) / 2
        earth_radius = 6378137.0
        meters_lon = math.pi / 180 * earth_radius * math.cos(math.radians(center_lat))
        meters_lat = math.pi / 180 * earth_radius
        res_x *= meters_lon
        res_y *= meters_lat

    return round(max(abs(res_x), abs(res_y)), 4) * 100  # cm/px


# 3) 타일링-처리 준비
def compute_tiling_params(raster, config, input_res):
    # 타일 스케일 비율, 오버랩, 타일 리스트 생성
    model_res = config['resolution']
    scale_factor = max(1, int(model_res // input_res)) if input_res < model_res else 1
    height, width = raster.shape
    tiles_overlap = config['tiles_overlap'] / 100.0
    windows = generate_for_size(width, height, config['tiles_size'] * scale_factor, tiles_overlap, clip=False)
    return height, width, scale_factor, tiles_overlap, windows


def generate_for_size(width, height, max_window_size, overlap_percent, clip=True):
    # Window 생성
    window_size_x = max_window_size
    window_size_y = max_window_size

    # If the input data is smaller than the specified window size,
    # clip the window size to the input size on both dimensions
    if clip:
        window_size_x = min(window_size_x, width)
        window_size_y = min(window_size_y, height)

    # Compute the window overlap and step size
    window_overlap_x = int(math.floor(window_size_x * overlap_percent))
    window_overlap_y = int(math.floor(window_size_y * overlap_percent))
    step_size_x = window_size_x - window_overlap_x
    step_size_y = window_size_y - window_overlap_y

    # Determine how many windows we will need in order to cover the input data
    last_x = width - window_size_x
    last_y = height - window_size_y
    x_offsets = list(range(0, last_x + 1, step_size_x))
    y_offsets = list(range(0, last_y + 1, step_size_y))

    # Unless the input data dimensions are exact multiples of the step size,
    # we will need one additional row and column of windows to get 100% coverage
    if len(x_offsets) == 0 or x_offsets[-1] != last_x:
        x_offsets.append(last_x)
    if len(y_offsets) == 0 or y_offsets[-1] != last_y:
        y_offsets.append(last_y)

    # Generate the list of windows
    windows = []
    for x_offset in x_offsets:
        for y_offset in y_offsets:
            windows.append(Window(
                x_offset,
                y_offset,
                window_size_x,
                window_size_y,
            ))

    return windows


def determine_indexes(raster):
    # alpha 제거
    indexes = raster.indexes
    if len(indexes) > 1 and raster.colorinterp[-1] == rasterio.enums.ColorInterp.alpha:
        indexes = indexes[:-1]
    return indexes


# 4) 추론, 타일 마스크 생성 및 병합
def read_tile(args):
    # 타일 생성
    raster_path, window, indexes, config = args
    with rasterio.open(raster_path) as src:
        img = src.read(
            indexes=indexes,
            window=window,
            boundless=True,
            fill_value=0,
            out_shape=(len(indexes), config['tiles_size'], config['tiles_size']),
            resampling=rasterio.enums.Resampling.bilinear
        )
    return img


def preprocess_batch(image_list):
    # 배치 만들기
    stacked = np.stack(image_list, axis=0)  # (N, C, H, W) 또는 (N, H, W, C)
    return preprocess(stacked)


def execute_batch_segmentation(images_batch, session, config):
    images_batch = preprocess_batch(images_batch)
    outs = session.run(None, {config['input_name']: images_batch})
    final_out = outs[0][:, 0, :, :]
    return [final_out[i] for i in range(final_out.shape[0])]


def save_tile_outputs(conf_map, rgb_tile, window, index, raster_path,
                      rgb_output_dir="data/tiles/rgb_tiles/test1",
                      conf_output_dir="data/tiles/confidence_map_tiles/test1",
                      binary_output_dir="data/tiles/binary_result_tiles/test1"):

    os.makedirs(rgb_output_dir, exist_ok=True)
    os.makedirs(conf_output_dir, exist_ok=True)
    os.makedirs(binary_output_dir, exist_ok=True)

    # 원본 TIF로부터 공간정보 추출
    with rasterio.open(raster_path) as src:
        crs = src.crs
        tile_transform = window_transform(window, src.transform)

    h, w = conf_map.shape

    # 👉 Confidence map 반전: 건물(1), 배경(0) 방향으로
    conf_map = 1 - conf_map

    # 1. RGB 타일 저장
    rgb_tile_path = os.path.join(rgb_output_dir, f"tile_rgb_{index:04d}.tif")
    with rasterio.open(
        rgb_tile_path, 'w',
        driver='GTiff',
        height=h,
        width=w,
        count=3,
        dtype=rgb_tile.dtype,
        crs=crs,
        transform=tile_transform
    ) as dst:
        dst.write(rgb_tile)

    # 2. Confidence map 저장 (float32)
    conf_path = os.path.join(conf_output_dir, f"tile_conf_{index:04d}.tif")
    with rasterio.open(
        conf_path, 'w',
        driver='GTiff',
        height=h,
        width=w,
        count=1,
        dtype='float32',
        crs=crs,
        transform=tile_transform
    ) as dst:
        dst.write(conf_map.astype(np.float32), 1)

    # 3. 이진 마스크 저장 (uint8)
    binary_map = (conf_map > 0.7).astype(np.uint8)
    binary_path = os.path.join(binary_output_dir, f"tile_binary_{index:04d}.tif")
    with rasterio.open(
        binary_path, 'w',
        driver='GTiff',
        height=h,
        width=w,
        count=1,
        dtype='uint8',
        crs=crs,
        transform=tile_transform
    ) as dst:
        dst.write(binary_map, 1)


# ───── 메인 처리 함수 ─────
def process_tiles(raster_path, windows, indexes, session, config, progress, total_perc, batch_size):
    n = len(windows)
    read_perc = total_perc * 0.15
    infer_perc = total_perc * 0.75
    merge_perc = total_perc * 0.1
    per_tile_merge_perc = merge_perc / n

    # 1. 타일 병렬 생성
    progress.update("Reading tiles", read_perc)
    args_list = [(raster_path, w, indexes, config) for w in windows]
    with Pool() as pool:
        tile_images = pool.map(read_tile, args_list)
    progress.write("Completed tile reading")

    # 2. 추론
    tile_masks = []
    total_batches = len(tile_images) // batch_size + int(len(tile_images) % batch_size != 0)
    per_batch_perc = infer_perc / total_batches

    for i, batch in enumerate(chunked(tile_images, batch_size)):
        progress.update(f"Inference batch {i+1}/{total_batches}", perc=per_batch_perc)
        batch_masks = execute_batch_segmentation(batch, session, config)  # List of (H, W)

        for j, conf_map in enumerate(batch_masks):
            global_index = i * batch_size + j
            window = windows[global_index]

            # RGB 타일 변환: (H, W, 3) → (3, H, W)
            rgb_tile = tile_images[global_index]
            if rgb_tile.shape[-1] == 3:
                rgb_tile = np.transpose(rgb_tile, (2, 0, 1))

            # 저장: RGB, confidence, binary mask
            save_tile_outputs(
                conf_map=conf_map,
                rgb_tile=rgb_tile,
                window=window,
                index=global_index,
                raster_path=raster_path
            )

        tile_masks.extend(batch_masks)
    progress.write("Completed inference")

    # 3. 병합 마스크 생성
    with rasterio.open(raster_path) as src:
        height, width = src.shape
        input_res = round(max(abs(src.transform[0]), abs(src.transform[4])), 4) * 100
        crs = src.crs
        transform = src.transform

    model_res = config['resolution']
    scale_factor = max(1, int(model_res // input_res)) if input_res < model_res else 1
    tiles_overlap = config['tiles_overlap'] / 100.0
    mask = np.zeros((height // scale_factor, width // scale_factor), dtype=np.uint8)

    conf_merged = np.zeros((height // scale_factor, width // scale_factor), dtype=np.float32)
    binary_merged = np.zeros((height // scale_factor, width // scale_factor), dtype=np.uint8)

    for idx, w in enumerate(windows):
        progress.update(f"Merging tile {idx+1}/{n}", perc=per_tile_merge_perc)
        merge_mask(tile_masks[idx], conf_merged, w, width, height, tiles_overlap, scale_factor)
    conf_merged = 1 - conf_merged
    binary_merged = (conf_merged > 0.5).astype(np.uint8)
    conf_merge_path = "data/confidence_map/test_av/conf_map.tif"
    binary_merge_path = "data/binary_result/test_av/binary_map.tif"
    os.makedirs(os.path.dirname(conf_merge_path), exist_ok=True)
    os.makedirs(os.path.dirname(binary_merge_path), exist_ok=True)
    with rasterio.open(
        conf_merge_path, 'w',
        driver='GTiff',
        height=conf_merged.shape[0],
        width=conf_merged.shape[1],
        count=1,
        dtype='float32',
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(conf_merged, 1)

    with rasterio.open(
        binary_merge_path, 'w',
        driver='GTiff',
        height=binary_merged.shape[0],
        width=binary_merged.shape[1],
        count=1,
        dtype='uint8',
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(binary_merged, 1)

    return mask


def preprocess(model_input):
    # 채널 정렬, 정규화
    s = model_input.shape
    if not len(s) in [3, 4]:
        raise Exception(f"Expected input with 3 or 4 dimensions, got: {s}")
    is_batched = len(s) == 4

    # expected: [batch],channel,height,width but could be: [batch],height,width,channel
    if s[-1] in [3, 4] and s[1] > s[-1]:
        if is_batched:
            model_input = np.transpose(model_input, (0, 3, 1, 2))
        else:
            model_input = np.transpose(model_input, (2, 0, 1))

    # add batch dimension (1, c, h, w)
    if not is_batched:
        model_input = np.expand_dims(model_input, axis=0)

    # drop alpha channel
    if model_input.shape[1] == 4:
        model_input = model_input[:, 0:3, :, :]

    if model_input.shape[1] != 3:
        raise Exception(f"Expected input channels to be 3, but got: {model_input.shape[1]}")

    # normalize
    if model_input.dtype == np.uint8:
        return (model_input / 255.0).astype(np.float32)

    if model_input.dtype.kind == 'f':
        min_value = float(model_input.min())
        value_range = float(model_input.max()) - min_value
    else:
        data_range = np.iinfo(model_input.dtype)
        min_value = 0
        value_range = float(data_range.max) - float(data_range.min)

    model_input = model_input.astype(np.float32)
    model_input -= min_value
    model_input /= value_range
    model_input[model_input > 1] = 1
    model_input[model_input < 0] = 0

    return model_input


# 5) 마스크 후처리
def merge_mask(tile_mask, mask, window, width, height, tiles_overlap, scale_factor=1.0):
    # Padding + Central Crop 방식으로 마스크 병합
    w = window
    row_off = int(w.row_off // scale_factor)
    col_off = int(w.col_off // scale_factor)
    tile_w, tile_h = tile_mask.shape

    # 오버랩 비율에 따른 패딩 계산
    pad_x = int(tiles_overlap * tile_w) // 2
    pad_y = int(tiles_overlap * tile_h) // 2

    # 경계 조건 확인 (이미지 가장자리인 경우 패딩 조정)
    pad_l = pad_x if w.col_off > 0 else 0
    pad_r = pad_x if w.col_off + w.width < width else 0
    pad_t = pad_y if w.row_off > 0 else 0
    pad_b = pad_y if w.row_off + w.height < height else 0

    # 중앙 영역만 사용하기 위한 좌표 계산
    central_x_start = pad_l
    central_y_start = pad_t
    central_x_end = tile_w - pad_r
    central_y_end = tile_h - pad_b

    # 타일의 중앙 부분만 추출
    central_tile = tile_mask[central_y_start:central_y_end, central_x_start:central_x_end]

    # 최종 마스크에 중앙 부분만 복사할 위치 계산
    dest_x_start = col_off + pad_l
    dest_y_start = row_off + pad_t
    dest_x_end = dest_x_start + (central_x_end - central_x_start)
    dest_y_end = dest_y_start + (central_y_end - central_y_start)

    # 이미지 경계 확인
    dest_x_end = min(dest_x_end, mask.shape[1])
    dest_y_end = min(dest_y_end, mask.shape[0])

    # 중앙 영역 크기 조정 (경계 조건 처리)
    central_width = dest_x_end - dest_x_start
    central_height = dest_y_end - dest_y_start
    central_tile = central_tile[:central_height, :central_width]

    # 최종 마스크에 중앙 부분 복사 (단순 대체, 가중치 없음)
    mask[dest_y_start:dest_y_end, dest_x_start:dest_x_end] = central_tile


def rect_intersect(rect1, rect2):
    """
    Given two rectangles, compute the intersection rectangle and return
    its coordinates in the coordinate system of both rectangles.

    Each rectangle is represented as (x, y, width, height).

    Returns:
    - (r1_x, r1_y, iw, ih): Intersection in rect1's local coordinates
    - (r2_x, r2_y, iw, ih): Intersection in rect2's local coordinates
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    ix = max(x1, x2)  # Left boundary
    iy = max(y1, y2)  # Top boundary
    ix2 = min(x1 + w1, x2 + w2)  # Right boundary
    iy2 = min(y1 + h1, y2 + h2)  # Bottom boundary

    # Compute intersection
    iw = max(0, ix2 - ix)
    ih = max(0, iy2 - iy)

    # If no intersection
    if iw == 0 or ih == 0:
        return None, None

    # Compute local coordinates
    r1_x = ix - x1
    r1_y = iy - y1
    r2_x = ix - x2
    r2_y = iy - y2

    return (r1_x, r1_y, iw, ih), (r2_x, r2_y, iw, ih)


try:
    from scipy.ndimage import median_filter
except ImportError:
    def median_filter(arr, size=5):
        assert size % 2 == 1, "Kernel size must be an odd number."
        if arr.shape[0] <= size or arr.shape[1] <= size:
            return arr

        pad_size = size // 2
        padded = np.pad(arr, pad_size, mode='edge')
        shape = (arr.shape[0], arr.shape[1], size, size)
        strides = padded.strides + padded.strides
        view = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        return np.median(view, axis=(2, 3)).astype(arr.dtype)


def main():
    args = make_args().parse_args()
    progress = ProgressBar()
    status = StatusManager(args.output)
    with status.task("Loading input files"):
        args = resolve_paths(args)

    # 건물 추론 시작
    with status.task("Loading ONNX Model"):
        progress.write("Start Building Segmentation")
        # 모델 및 설정 불러오기
        progress.update("Loading ONNX Model", perc=5)
        session, config = create_session(get_model_file(args.model), max_threads=args.max_threads)
        config = override_config(
            config,
            conf_threshold=args.conf_threshold,
            resolution=args.resolution,
            classes=args.classes
        )
        progress.write("Completed Load Model")
    # 영상 데이터, 해상도 설정
    with status.task("Loading GeoTIFF"):
        progress.update("Loading GeoTIFF", perc=5)
        raster = load_raster(args.geotiff)
        progress.write("Completed Load GeoTIFF")

    with raster:
        input_res = get_input_resolution(raster)
        height, width, scale_factor, tiles_overlap, windows = compute_tiling_params(raster, config, input_res)
        indexes = determine_indexes(raster)

        with status.task("Processing Tiles"):
            mask = process_tiles(args.geotiff, windows, indexes, session, config, progress=progress, total_perc=55, batch_size=args.batch_size)
            progress.write("Completed Process tiles")


if __name__ == "__main__":
    main()
