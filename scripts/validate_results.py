import os
from pathlib import Path

import pandas as pd


REQUIRED_FILES = [
    "results/raw_metrics.csv",
    "results/summary_stats.csv",
    "results/case_study.csv",
    "results/experimental/raw_metrics_restoration_official.csv",
    "results/sr_bicubic/raw_metrics.csv",
    "results/sr_bicubic/summary_by_dataset_scale.csv",
    "results/sr_bicubic/failure_cases.csv",
    "results/sr_model_comparison/raw_metrics_x4.csv",
    "results/sr_model_comparison/summary_by_method_dataset_x4.csv",
    "results/sr_model_comparison/improvement_over_bicubic_x4.csv",
    "results/sr_model_comparison/case_study_x4.csv",
    "results/sr_model_qalign/raw_metrics_x4.csv",
    "results/sr_model_qalign/summary_by_method_dataset_x4.csv",
    "results/sr_model_qalign/correlations_x4.csv",
    "results/sr_model_qalign/disagreement_cases_x4.csv",
]

REQUIRED_COLUMNS = {
    "results/raw_metrics.csv": {
        "image_name",
        "base_name",
        "degradation",
        "gt_path",
        "pred_path",
        "psnr",
        "ssim",
        "lpips",
        "qalign_score",
    },
    "results/summary_stats.csv": {
        "group",
        "metric_x",
        "metric_y",
        "pearson",
        "spearman",
        "n",
    },
    "results/case_study.csv": {
        "case_type",
        "image_name",
        "base_name",
        "degradation",
        "psnr",
        "ssim",
        "lpips",
        "qalign_score",
    },
    "results/experimental/raw_metrics_restoration_official.csv": {
        "image_name",
        "base_name",
        "method",
        "gt_path",
        "pred_path",
        "crop_border",
        "psnr",
        "ssim",
        "lpips",
        "qalign_score",
    },
    "results/sr_bicubic/raw_metrics.csv": {
        "dataset",
        "scale",
        "image_name",
        "base_name",
        "hr_path",
        "lr_path",
        "psnr_rgb",
        "ssim_rgb",
        "psnr_y",
        "ssim_y",
    },
    "results/sr_bicubic/summary_by_dataset_scale.csv": {
        "dataset",
        "scale",
        "n",
        "mean_psnr_y",
        "mean_ssim_y",
        "mean_psnr_rgb",
        "mean_ssim_rgb",
    },
    "results/sr_bicubic/failure_cases.csv": {
        "case_type",
        "dataset",
        "scale",
        "image_name",
        "base_name",
        "psnr_y",
        "ssim_y",
    },
    "results/sr_model_comparison/raw_metrics_x4.csv": {
        "method",
        "dataset",
        "scale",
        "image_name",
        "base_name",
        "hr_path",
        "lr_path",
        "pred_path",
        "psnr_rgb",
        "ssim_rgb",
        "psnr_y",
        "ssim_y",
    },
    "results/sr_model_comparison/summary_by_method_dataset_x4.csv": {
        "method",
        "dataset",
        "scale",
        "n",
        "mean_psnr_y",
        "mean_ssim_y",
        "mean_psnr_rgb",
        "mean_ssim_rgb",
    },
    "results/sr_model_comparison/improvement_over_bicubic_x4.csv": {
        "method",
        "dataset",
        "scale",
        "n",
        "mean_delta_psnr_y",
        "mean_delta_ssim_y",
        "win_rate_psnr_y",
        "win_rate_ssim_y",
    },
    "results/sr_model_comparison/case_study_x4.csv": {
        "case_type",
        "dataset",
        "scale",
        "base_name",
        "image_name",
        "psnr_y_Bicubic",
        "psnr_y_SwinIR",
        "psnr_y_MambaIR",
        "psnr_y_MambaIRv2",
        "score",
    },
    "results/sr_model_qalign/raw_metrics_x4.csv": {
        "method",
        "dataset",
        "scale",
        "image_name",
        "base_name",
        "hr_path",
        "lr_path",
        "pred_path",
        "error",
        "width",
        "height",
        "crop_border",
        "psnr_rgb",
        "ssim_rgb",
        "psnr_y",
        "ssim_y",
        "lpips",
        "qalign_score",
    },
    "results/sr_model_qalign/summary_by_method_dataset_x4.csv": {
        "method",
        "dataset",
        "scale",
        "n",
        "mean_psnr_y",
        "mean_ssim_y",
        "mean_lpips",
        "mean_qalign",
        "median_lpips",
        "median_qalign",
    },
    "results/sr_model_qalign/correlations_x4.csv": {
        "group",
        "metric_x",
        "metric_y",
        "pearson",
        "spearman",
        "n",
    },
    "results/sr_model_qalign/disagreement_cases_x4.csv": {
        "case_type",
        "method",
        "dataset",
        "scale",
        "image_name",
        "base_name",
        "psnr_y",
        "ssim_y",
        "lpips",
        "qalign_score",
        "score",
    },
}


def validate_file(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Result file is empty: {path}")

    missing = REQUIRED_COLUMNS[path] - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    metric_cols = [
        c
        for c in [
            "psnr",
            "ssim",
            "lpips",
            "qalign_score",
            "psnr_rgb",
            "ssim_rgb",
            "psnr_y",
            "ssim_y",
            "mean_psnr_y",
            "mean_ssim_y",
            "mean_delta_psnr_y",
            "mean_delta_ssim_y",
            "win_rate_psnr_y",
            "win_rate_ssim_y",
            "mean_lpips",
            "mean_qalign",
            "median_lpips",
            "median_qalign",
            "pearson",
            "spearman",
        ]
        if c in df.columns
    ]
    for col in metric_cols:
        if df[col].isna().any():
            raise ValueError(f"{path} contains NaN values in {col}")

    return df


def main():
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    for path in REQUIRED_FILES:
        df = validate_file(path)
        print(f"[OK] {path}: {len(df)} rows")

    raw = pd.read_csv("results/raw_metrics.csv")
    print("\nDegradation counts:")
    print(raw["degradation"].value_counts().sort_index().to_string())

    restoration = pd.read_csv("results/experimental/raw_metrics_restoration_official.csv")
    print("\nRestoration method counts:")
    print(restoration["method"].value_counts().sort_index().to_string())

    sr = pd.read_csv("results/sr_bicubic/raw_metrics.csv")
    print("\nSR bicubic dataset counts:")
    print(sr.groupby(["dataset", "scale"]).size().to_string())

    sr_models = pd.read_csv("results/sr_model_comparison/raw_metrics_x4.csv")
    print("\nSR model-comparison counts:")
    print(sr_models.groupby(["method", "dataset", "scale"]).size().to_string())

    sr_qalign = pd.read_csv("results/sr_model_qalign/raw_metrics_x4.csv")
    errors = sr_qalign["error"].fillna("")
    if (errors != "").any():
        raise ValueError("results/sr_model_qalign/raw_metrics_x4.csv contains Q-Align errors")
    print("\nSR model Q-Align counts:")
    print(sr_qalign.groupby(["method", "dataset", "scale"]).size().to_string())


if __name__ == "__main__":
    main()
