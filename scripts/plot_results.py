import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", type=str, default="./results/raw_metrics.csv")
    parser.add_argument("--summary_csv", type=str, default="./results/summary_stats.csv")
    parser.add_argument("--fig_dir", type=str, default="./results/figures")
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    df = pd.read_csv(args.raw_csv)
    summary_df = pd.read_csv(args.summary_csv)

    sns.set_theme(style="whitegrid", context="talk")

    # 1. PSNR vs Q-Align
    plt.figure(figsize=(9, 7), dpi=200)
    sns.scatterplot(
        data=df,
        x="psnr",
        y="qalign_score",
        hue="degradation",
        s=110,
        alpha=0.85,
        edgecolor="black",
    )
    plt.title("PSNR vs. Q-Align Score")
    plt.xlabel("PSNR (dB)")
    plt.ylabel("Q-Align Score")
    plt.tight_layout()
    plt.savefig(os.path.join(args.fig_dir, "scatter_psnr_qalign.png"), bbox_inches="tight")
    plt.close()

    # 2. SSIM vs Q-Align
    plt.figure(figsize=(9, 7), dpi=200)
    sns.scatterplot(
        data=df,
        x="ssim",
        y="qalign_score",
        hue="degradation",
        s=110,
        alpha=0.85,
        edgecolor="black",
    )
    plt.title("SSIM vs. Q-Align Score")
    plt.xlabel("SSIM")
    plt.ylabel("Q-Align Score")
    plt.tight_layout()
    plt.savefig(os.path.join(args.fig_dir, "scatter_ssim_qalign.png"), bbox_inches="tight")
    plt.close()

    # 3. LPIPS vs Q-Align
    plt.figure(figsize=(9, 7), dpi=200)
    sns.scatterplot(
        data=df,
        x="lpips",
        y="qalign_score",
        hue="degradation",
        s=110,
        alpha=0.85,
        edgecolor="black",
    )
    plt.title("LPIPS vs. Q-Align Score")
    plt.xlabel("LPIPS (lower is better)")
    plt.ylabel("Q-Align Score")
    plt.tight_layout()
    plt.savefig(os.path.join(args.fig_dir, "scatter_lpips_qalign.png"), bbox_inches="tight")
    plt.close()

    # 4. distribution of qalign score
    plt.figure(figsize=(9, 7), dpi=200)
    sns.boxplot(data=df, x="degradation", y="qalign_score")
    sns.stripplot(data=df, x="degradation", y="qalign_score", color="black", alpha=0.5, size=4)
    plt.title("Q-Align Score Distribution by Degradation")
    plt.xlabel("Degradation")
    plt.ylabel("Q-Align Score")
    plt.tight_layout()
    plt.savefig(os.path.join(args.fig_dir, "box_qalign_by_degradation.png"), bbox_inches="tight")
    plt.close()

    # 5. correlation bar plot
    corr_df = summary_df[
        (summary_df["group"] != "overall") &
        (summary_df["metric_y"] == "qalign_score") &
        (summary_df["metric_x"].isin(["psnr", "ssim", "lpips"]))
    ].copy()

    corr_long = corr_df.melt(
        id_vars=["group", "metric_x", "metric_y", "n"],
        value_vars=["pearson", "spearman"],
        var_name="corr_type",
        value_name="corr_value"
    )

    plt.figure(figsize=(11, 7), dpi=200)
    sns.barplot(
        data=corr_long,
        x="group",
        y="corr_value",
        hue="metric_x"
    )
    plt.title("Correlation with Q-Align across Degradation Types")
    plt.xlabel("Degradation")
    plt.ylabel("Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(args.fig_dir, "bar_correlation_with_qalign.png"), bbox_inches="tight")
    plt.close()

    print(f"Figures saved to: {args.fig_dir}")


if __name__ == "__main__":
    main()
