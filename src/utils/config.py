import os
import argparse


def make_args():
    parser = argparse.ArgumentParser(description="건물 추론 및 후처리 평가 실험")

    parser.add_argument("--region", type=str, default="test", help="평가 대상 지역 이름")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="confidence threshold (0~1)")

    base_data = os.path.join("..", "data")
    base_eval = os.path.join("..", "evaluation")

    # 각 디렉토리 지정
    parser.add_argument("--orthoimage", type=str, default=os.path.join(base_data, "orthoimage"))
    parser.add_argument("--confidence", type=str, default=os.path.join(base_data, "confidence_map_tiles"))
    parser.add_argument("--groundtruth", type=str, default=os.path.join(base_data, "groundtruth"))
    parser.add_argument("--binary_result_tiles", type=str, default=os.path.join(base_data, "binary_result_tiles"))
    parser.add_argument("--postprocess_result", type=str, default=os.path.join(base_data, "postprocess_result"))
    parser.add_argument("--rgb_tiles", type=str, default=os.path.join(base_data, "rgb_tiles"))
    parser.add_argument("--eval_result", type=str, default=base_eval)

    return parser


def find_file_in(directory, ext=".tif"):
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"{directory} 는 디렉토리가 아닙니다.")

    for fname in os.listdir(directory):
        if fname.lower().endswith(ext):
            return os.path.join(directory, fname)

    raise FileNotFoundError(f"{ext} 파일이 {directory} 내에 없습니다.")
