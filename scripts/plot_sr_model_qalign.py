import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = ["Bicubic", "SwinIR", "MambaIR", "MambaIRv2"]
DATASET_ORDER = ["Set5", "Set14", "B100", "Urban100", "Manga109"]
METHOD_COLORS = {
    "Bicubic": "#8A8F98",
    "SwinIR": "#4C78A8",
    "MambaIR": "#F58518",
    "MambaIRv2": "#54A24B",
}


def ordered(values: pd.Series, preferred: list[str]) -> list[str]:
    present = list(values.drop_duplicates())
    return [item for item in preferred if item in present] + [item for item in present if item not in preferred]


def plot_grouped_bars(summary: pd.DataFrame, metric: str, ylabel: str, output_path: Path):
    datasets = ordered(summary["dataset"], DATASET_ORDER)
    methods = ordered(summary["method"], METHOD_ORDER)
    x = range(len(datasets))
    width = 0.18

    plt.figure(figsize=(9, 5))
    for idx, method in enumerate(methods):
        values = []
        for dataset in datasets:
            row = summary[(summary["dataset"] == dataset) & (summary["method"] == method)]
            values.append(float(row[metric].iloc[0]) if not row.empty else 0.0)
        offsets = [v + (idx - (len(methods) - 1) / 2) * width for v in x]
        plt.bar(offsets, values, width=width, label=method, color=METHOD_COLORS.get(method))

    plt.xticks(list(x), datasets, rotation=15)
    plt.xlabel("Dataset")
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(frameon=False, ncols=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_correlation_bars(correlations: pd.DataFrame, output_path: Path):
    overall = correlations[correlations["group"] == "all"].copy()
    overall["metric_x"] = overall["metric_x"].map({
        "psnr_y": "PSNR-Y",
        "ssim_y": "SSIM-Y",
        "lpips": "LPIPS",
    })

    x = range(len(overall))
    width = 0.34
    plt.figure(figsize=(7, 4.5))
    plt.bar([v - width / 2 for v in x], overall["pearson"], width=width, label="Pearson", color="#4C78A8")
    plt.bar([v + width / 2 for v in x], overall["spearman"], width=width, label="Spearman", color="#F58518")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(list(x), overall["metric_x"])
    plt.ylabel("Correlation with Q-Align")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_lpips_qalign_scatter(raw: pd.DataFrame, output_path: Path):
    valid = raw[raw["error"].fillna("") == ""].copy()
    plt.figure(figsize=(7, 5))
    for method in ordered(valid["method"], METHOD_ORDER):
        group = valid[valid["method"] == method]
        plt.scatter(
            group["lpips"],
            group["qalign_score"],
            s=14,
            alpha=0.55,
            label=method,
            color=METHOD_COLORS.get(method),
            edgecolors="none",
        )
    plt.xlabel("LPIPS lower is better")
    plt.ylabel("Q-Align score higher is better")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, ncols=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", type=str, default="./results/sr_model_qalign/raw_metrics_x4.csv")
    parser.add_argument("--summary_csv", type=str, default="./results/sr_model_qalign/summary_by_method_dataset_x4.csv")
    parser.add_argument("--correlation_csv", type=str, default="./results/sr_model_qalign/correlations_x4.csv")
    parser.add_argument("--fig_dir", type=str, default="./results/figures")
    args = parser.parse_args()

    raw = pd.read_csv(args.raw_csv)
    summary = pd.read_csv(args.summary_csv)
    correlations = pd.read_csv(args.correlation_csv)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_grouped_bars(summary, "mean_qalign", "Mean Q-Align score", fig_dir / "sr_model_x4_qalign_by_dataset.png")
    plot_grouped_bars(summary, "mean_lpips", "Mean LPIPS lower is better", fig_dir / "sr_model_x4_lpips_by_dataset.png")
    plot_correlation_bars(correlations, fig_dir / "sr_model_x4_qalign_correlations.png")
    plot_lpips_qalign_scatter(raw, fig_dir / "sr_model_x4_lpips_vs_qalign.png")

    print(f"Saved SR Q-Align figures to {fig_dir}")


if __name__ == "__main__":
    main()
