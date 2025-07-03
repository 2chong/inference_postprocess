import pandas as pd
import numpy as np


def evaluate_summary_table(method_name: str, gt, result, component_df, iou_thresholds=np.arange(0.5, 1.0, 0.05)) -> pd.DataFrame:
    # 1. 기본 통계
    gt_total = len(gt)
    result_total = len(result)

    gt_underseg = gt["Relation"].eq("N:1").sum()
    result_underseg = result["Relation"].eq("N:1").sum()

    gt_underseg_rate = (gt_underseg / gt_total * 100) if gt_total > 0 else 0
    result_underseg_rate = (result_underseg / result_total * 100) if result_total > 0 else 0

    total_result_area = result["area"].sum()

    # 2. AR@[.5:.95] 계산 (GT 기준 재현율)
    ar_scores = []
    for thr in iou_thresholds:
        matched_gt_ids = set()
        for _, row in component_df.iterrows():
            if row["iou"] >= thr:
                matched_gt_ids.update(gt.loc[gt["comp_idx"] == row["comp_idx"]].index)

        recall = len(matched_gt_ids) / gt_total if gt_total > 0 else 0
        ar_scores.append(recall)

    ar_mean = np.mean(ar_scores)

    # 3. 정리
    summary = pd.DataFrame([{
        "method": method_name,
        "GT 객체 수": gt_total,
        "GT 기준 Underseg 객체 수": gt_underseg,
        "Result 객체 수": result_total,
        "Result 기준 Underseg 객체 수": result_underseg,
        "GT 기준 Underseg 비율 (%)": round(gt_underseg_rate, 2),
        "Result 기준 Underseg 비율 (%)": round(result_underseg_rate, 2),
        "AR@[0.5:0.95]": round(ar_mean, 4),
        "result의 면적 (㎡)": round(total_result_area, 2)
    }])

    return summary
