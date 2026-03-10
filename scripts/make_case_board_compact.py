import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd


TARGET_CASES = [
    "PSNR_high_QAlign_low",
    "PSNR_low_QAlign_high",
    "LPIPS_good_QAlign_low",
    "LPIPS_bad_QAlign_high",
]


def load_img(path):
    return mpimg.imread(path)


def pick_one_per_case(df):
    selected = []
    for case_type in TARGET_CASES:
        sub = df[df["case_type"] == case_type].copy()
        if len(sub) == 0:
            continue

        # 选每类第一条；后面如果你想手动指定，我们再改
        selected.append(sub.iloc[0])

    if len(selected) == 0:
        raise RuntimeError("No matching cases found.")
    return pd.DataFrame(selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_csv", type=str, default="./results/case_study.csv")
    parser.add_argument("--output_png", type=str, default="./results/figures/case_study_board_compact.png")
    args = parser.parse_args()

    df = pd.read_csv(args.case_csv)
    df = pick_one_per_case(df)

    n_rows = len(df)
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.2 * n_rows), dpi=220)

    if n_rows == 1:
        axes = [axes]

    for i, (_, row) in enumerate(df.iterrows()):
        gt_img = load_img(row["gt_path"])
        pred_img = load_img(row["pred_path"])

        # GT
        ax0 = axes[i][0]
        ax0.imshow(gt_img)
        ax0.set_title("Ground Truth", fontsize=11)
        ax0.axis("off")

        # degraded/pred
        ax1 = axes[i][1]
        ax1.imshow(pred_img)
        ax1.set_title(f"Compared Image ({row['degradation']})", fontsize=11)
        ax1.axis("off")

        # text
        ax2 = axes[i][2]
        ax2.axis("off")

        text = (
            f"Case: {row['case_type']}\n\n"
            f"Image: {row['image_name']}\n"
            f"Degradation: {row['degradation']}\n\n"
            f"PSNR     : {row['psnr']:.4f}\n"
            f"SSIM     : {row['ssim']:.4f}\n"
            f"LPIPS    : {row['lpips']:.4f}\n"
            f"Q-Align  : {row['qalign_score']:.4f}"
        )

        ax2.text(
            0.02, 0.98, text,
            va="top", ha="left",
            fontsize=12,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7", edgecolor="gray")
        )

    fig.suptitle("Compact Case Study Board: Metric Disagreement Examples", fontsize=17, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.992])

    os.makedirs(os.path.dirname(args.output_png), exist_ok=True)
    plt.savefig(args.output_png, bbox_inches="tight")
    plt.close()

    print(f"Saved compact case board to: {args.output_png}")


if __name__ == "__main__":
    main()
