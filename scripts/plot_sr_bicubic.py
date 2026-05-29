import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw


def plot_metric(summary: pd.DataFrame, metric: str, ylabel: str, output_path: Path):
    plt.figure(figsize=(8, 5))
    for dataset, group in summary.groupby("dataset"):
        group = group.sort_values("scale")
        plt.plot(group["scale"], group[metric], marker="o", linewidth=2, label=dataset)
    plt.xlabel("Bicubic scale")
    plt.ylabel(ylabel)
    plt.xticks(sorted(summary["scale"].unique()))
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_x4_bar(summary: pd.DataFrame, output_path: Path):
    x4 = summary[summary["scale"] == 4].sort_values("mean_psnr_y", ascending=False)
    plt.figure(figsize=(8, 5))
    plt.bar(x4["dataset"], x4["mean_psnr_y"], color="#4C78A8")
    plt.ylabel("Mean Y-channel PSNR (dB)")
    plt.xlabel("Dataset")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def resize_for_cell(path: str, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, "white")
        x = (size[0] - image.width) // 2
        y = (size[1] - image.height) // 2
        canvas.paste(image, (x, y))
        return canvas


def bicubic_output_for_cell(lr_path: str, hr_path: str, size: tuple[int, int]) -> Image.Image:
    with Image.open(hr_path) as hr_image:
        target_size = hr_image.size
    with Image.open(lr_path) as lr_image:
        pred = lr_image.convert("RGB").resize(target_size, Image.Resampling.BICUBIC)
        pred.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, "white")
        x = (size[0] - pred.width) // 2
        y = (size[1] - pred.height) // 2
        canvas.paste(pred, (x, y))
        return canvas


def make_failure_board(failure_df: pd.DataFrame, output_path: Path, max_cases: int):
    cases = failure_df[
        (failure_df["scale"] == 4)
        & (failure_df["case_type"] == "lowest_psnr_y")
        & (failure_df["error"].fillna("") == "")
    ].sort_values(["dataset", "psnr_y"]).head(max_cases)

    if cases.empty:
        return

    cell_w, cell_h = 220, 170
    label_h = 56
    columns = 2
    rows = len(cases)
    board = Image.new("RGB", (columns * cell_w, rows * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(board)

    for row_idx, row in enumerate(cases.itertuples(index=False)):
        y0 = row_idx * (cell_h + label_h)
        hr = resize_for_cell(row.hr_path, (cell_w, cell_h))
        lr = bicubic_output_for_cell(row.lr_path, row.hr_path, (cell_w, cell_h))
        board.paste(hr, (0, y0))
        board.paste(lr, (cell_w, y0))
        label = (
            f"{row.dataset} X{row.scale} | {row.image_name}\n"
            f"Y-PSNR {row.psnr_y:.2f} dB | Y-SSIM {row.ssim_y:.3f}"
        )
        draw.text((8, y0 + cell_h + 6), "HR reference", fill=(0, 0, 0))
        draw.text((cell_w + 8, y0 + cell_h + 6), "Bicubic output", fill=(0, 0, 0))
        draw.text((8, y0 + cell_h + 26), label, fill=(60, 60, 60))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=str, default="./results/sr_bicubic/summary_by_dataset_scale.csv")
    parser.add_argument("--failure_csv", type=str, default="./results/sr_bicubic/failure_cases.csv")
    parser.add_argument("--fig_dir", type=str, default="./results/figures")
    parser.add_argument("--max_cases", type=int, default=8)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary_csv)
    failure = pd.read_csv(args.failure_csv)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(summary, "mean_psnr_y", "Mean Y-channel PSNR (dB)", fig_dir / "sr_bicubic_psnr_y_by_scale.png")
    plot_metric(summary, "mean_ssim_y", "Mean Y-channel SSIM", fig_dir / "sr_bicubic_ssim_y_by_scale.png")
    plot_x4_bar(summary, fig_dir / "sr_bicubic_x4_psnr_y_by_dataset.png")
    make_failure_board(failure, fig_dir / "sr_bicubic_x4_failure_board.png", args.max_cases)

    print(f"Saved SR bicubic figures to {fig_dir}")


if __name__ == "__main__":
    main()
