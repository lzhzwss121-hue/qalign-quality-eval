import argparse
import os

import pandas as pd
from scipy.stats import pearsonr, spearmanr


def safe_corr(x, y):
    if len(x) < 2:
        return None, None
    pearson_val = pearsonr(x, y)[0]
    spearman_val = spearmanr(x, y)[0]
    return pearson_val, spearman_val


def add_result(rows, group_name, metric_x, metric_y, x, y):
    pearson_val, spearman_val = safe_corr(x, y)
    rows.append({
        "group": group_name,
        "metric_x": metric_x,
        "metric_y": metric_y,
        "pearson": pearson_val,
        "spearman": spearman_val,
        "n": len(x),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="./results/raw_metrics.csv")
    parser.add_argument("--output_csv", type=str, default="./results/summary_stats.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    df = pd.read_csv(args.input_csv)

    required_cols = ["degradation", "psnr", "ssim", "lpips", "qalign_score"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    rows = []

    metric_pairs = [
        ("psnr", "qalign_score"),
        ("ssim", "qalign_score"),
        ("lpips", "qalign_score"),
        ("psnr", "lpips"),
        ("ssim", "lpips"),
    ]

    # overall
    for mx, my in metric_pairs:
        sub = df[[mx, my]].dropna()
        add_result(rows, "overall", mx, my, sub[mx], sub[my])

    # by degradation
    for degradation, gdf in df.groupby("degradation"):
        for mx, my in metric_pairs:
            sub = gdf[[mx, my]].dropna()
            add_result(rows, degradation, mx, my, sub[mx], sub[my])

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output_csv, index=False)

    print("Saved summary stats to:", args.output_csv)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
