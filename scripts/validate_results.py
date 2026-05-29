import os
from pathlib import Path

import pandas as pd


REQUIRED_FILES = [
    "results/raw_metrics.csv",
    "results/summary_stats.csv",
    "results/case_study.csv",
    "results/experimental/raw_metrics_restoration_official.csv",
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

    metric_cols = [c for c in ["psnr", "ssim", "lpips", "qalign_score"] if c in df.columns]
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


if __name__ == "__main__":
    main()
