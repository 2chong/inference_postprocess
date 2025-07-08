import pandas as pd
import numpy as np


def evaluate_summary_table(
    method_name: str,
    gt,
    result,
    component_df,
    iou_thresholds=np.arange(0.5, 1.0, 0.05)
) -> pd.DataFrame:
    # 1. 기본 통계
    gt_total = len(gt)
    result_total = len(result)

    gt_underseg = gt["Relation"].eq("N:1").sum()
    result_underseg = result["Relation"].eq("N:1").sum()

    gt_underseg_rate = (gt_underseg / gt_total * 100) if gt_total > 0 else 0
    result_underseg_rate = (result_underseg / result_total * 100) if result_total > 0 else 0

    total_result_area = result["area"].sum()

    # 2. AR@[.5:.95]
    ar_scores = []
    for thr in iou_thresholds:
        matched_gt_ids = set()
        for _, row in component_df.iterrows():
            if row["iou"] >= thr:
                matched_gt_ids.update(gt.loc[gt["comp_idx"] == row["comp_idx"]].index)

        recall = len(matched_gt_ids) / gt_total if gt_total > 0 else 0
        ar_scores.append(recall)

    ar_mean = np.mean(ar_scores)

    # 3. Pixel-level 평가 (geometry의 면적 기준으로 계산)
    TP_area = 0
    FP_area = 0
    FN_area = 0

    for _, row in component_df.iterrows():
        gt_geom = gt.loc[gt["comp_idx"] == row["comp_idx"]].unary_union
        res_geom = result.loc[result["comp_idx"] == row["comp_idx"]].unary_union

        inter = gt_geom.intersection(res_geom).area
        union = gt_geom.union(res_geom).area
        gt_only = gt_geom.area - inter
        res_only = res_geom.area - inter

        TP_area += inter
        FN_area += gt_only
        FP_area += res_only

    recall_pix = TP_area / (TP_area + FN_area) if (TP_area + FN_area) > 0 else 0
    precision_pix = TP_area / (TP_area + FP_area) if (TP_area + FP_area) > 0 else 0
    iou_pix = TP_area / (TP_area + FP_area + FN_area) if (TP_area + FP_area + FN_area) > 0 else 0
    f1_pix = 2 * precision_pix * recall_pix / (precision_pix + recall_pix) if (precision_pix + recall_pix) > 0 else 0

    # 4. 최종 정리
    summary = pd.DataFrame([{
        "method": method_name,
        "GT 객체 수": gt_total,
        "GT 기준 Underseg 객체 수": gt_underseg,
        "Result 객체 수": result_total,
        "Result 기준 Underseg 객체 수": result_underseg,
        "GT 기준 Underseg 비율 (%)": round(gt_underseg_rate, 2),
        "Result 기준 Underseg 비율 (%)": round(result_underseg_rate, 2),
        "AR@[0.5:0.95]": round(ar_mean, 4),
        "면적 (㎡)": round(total_result_area, 2),
        "IoU": round(iou_pix, 4),
        "Recall": round(recall_pix, 4),
        "Precision": round(precision_pix, 4),
        "F1": round(f1_pix, 4)
    }])

    return summary
