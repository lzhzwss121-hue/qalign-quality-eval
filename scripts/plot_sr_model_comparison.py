import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw


METHOD_ORDER = ["Bicubic", "SwinIR", "MambaIR", "MambaIRv2"]
METHOD_COLORS = {
    "Bicubic": "#8A8F98",
    "SwinIR": "#4C78A8",
    "MambaIR": "#F58518",
    "MambaIRv2": "#54A24B",
}


def plot_grouped_bars(summary: pd.DataFrame, metric: str, ylabel: str, output_path: Path):
    datasets = list(summary["dataset"].drop_duplicates())
    methods = [m for m in METHOD_ORDER if m in set(summary["method"])]
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
    plt.ylabel(ylabel)
    plt.xlabel("Dataset")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(frameon=False, ncols=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_improvement(improvement: pd.DataFrame, metric: str, ylabel: str, output_path: Path):
    datasets = list(improvement["dataset"].drop_duplicates())
    methods = [m for m in METHOD_ORDER if m != "Bicubic" and m in set(improvement["method"])]
    x = range(len(datasets))
    width = 0.22

    plt.figure(figsize=(9, 5))
    for idx, method in enumerate(methods):
        values = []
        for dataset in datasets:
            row = improvement[(improvement["dataset"] == dataset) & (improvement["method"] == method)]
            values.append(float(row[metric].iloc[0]) if not row.empty else 0.0)
        offsets = [v + (idx - (len(methods) - 1) / 2) * width for v in x]
        plt.bar(offsets, values, width=width, label=method, color=METHOD_COLORS.get(method))

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(list(x), datasets, rotation=15)
    plt.ylabel(ylabel)
    plt.xlabel("Dataset")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(frameon=False, ncols=3)
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


def bicubic_for_cell(lr_path: str, hr_path: str, size: tuple[int, int]) -> Image.Image:
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


def make_qualitative_board(raw: pd.DataFrame, cases: pd.DataFrame, output_path: Path, max_cases: int):
    if cases.empty:
        return
    selected = cases[cases["case_type"] == "largest_mambairv2_gain_over_swinir"].head(max_cases)
    if selected.empty:
        selected = cases.head(max_cases)

    columns = ["HR", "Bicubic", "SwinIR", "MambaIR", "MambaIRv2"]
    cell_w, cell_h = 190, 150
    label_h = 70
    board = Image.new("RGB", (len(columns) * cell_w, len(selected) * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(board)

    for row_idx, row in enumerate(selected.itertuples(index=False)):
        y0 = row_idx * (cell_h + label_h)
        key = (raw["dataset"] == row.dataset) & (raw["scale"] == row.scale) & (raw["base_name"] == row.base_name)
        group = raw[key]
        paths = {r.method: r.pred_path for r in group.itertuples(index=False)}
        hr_path = str(group["hr_path"].iloc[0])
        lr_path = str(group["lr_path"].iloc[0])

        for col_idx, column in enumerate(columns):
            x0 = col_idx * cell_w
            if column == "HR":
                image = resize_for_cell(hr_path, (cell_w, cell_h))
            elif column == "Bicubic":
                image = bicubic_for_cell(lr_path, hr_path, (cell_w, cell_h))
            else:
                image = resize_for_cell(paths[column], (cell_w, cell_h))
            board.paste(image, (x0, y0))
            draw.text((x0 + 6, y0 + cell_h + 6), column, fill=(0, 0, 0))

        label = (
            f"{row.dataset} X{row.scale} | {row.image_name}\n"
            f"MambaIRv2-SwinIR gain: {row.score:.2f} dB"
        )
        draw.text((6, y0 + cell_h + 30), label, fill=(50, 50, 50))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", type=str, default="./results/sr_model_comparison/raw_metrics_x4.csv")
    parser.add_argument("--summary_csv", type=str, default="./results/sr_model_comparison/summary_by_method_dataset_x4.csv")
    parser.add_argument("--improvement_csv", type=str, default="./results/sr_model_comparison/improvement_over_bicubic_x4.csv")
    parser.add_argument("--case_csv", type=str, default="./results/sr_model_comparison/case_study_x4.csv")
    parser.add_argument("--fig_dir", type=str, default="./results/figures")
    parser.add_argument("--max_cases", type=int, default=6)
    args = parser.parse_args()

    raw = pd.read_csv(args.raw_csv)
    summary = pd.read_csv(args.summary_csv)
    improvement = pd.read_csv(args.improvement_csv)
    cases = pd.read_csv(args.case_csv)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_grouped_bars(summary, "mean_psnr_y", "Mean Y-channel PSNR (dB)", fig_dir / "sr_model_x4_psnr_y_by_dataset.png")
    plot_grouped_bars(summary, "mean_ssim_y", "Mean Y-channel SSIM", fig_dir / "sr_model_x4_ssim_y_by_dataset.png")
    plot_improvement(
        improvement,
        "mean_delta_psnr_y",
        "Mean Y-PSNR gain over bicubic (dB)",
        fig_dir / "sr_model_x4_delta_psnr_y_over_bicubic.png",
    )
    plot_improvement(
        improvement,
        "win_rate_psnr_y",
        "Win rate over bicubic by Y-PSNR",
        fig_dir / "sr_model_x4_win_rate_over_bicubic.png",
    )
    make_qualitative_board(raw, cases, fig_dir / "sr_model_x4_qualitative_board.png", args.max_cases)

    print(f"Saved SR model comparison figures to {fig_dir}")


if __name__ == "__main__":
    main()
