import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def normalize_stem(path: Path, scale: int, method: str | None = None) -> str:
    stem = path.stem
    if method:
        stem = re.sub(rf"_{re.escape(method)}$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_(SwinIR|MambaIRv2|MambaIR|Bicubic)$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(rf"(_LRBI_x{scale}|_lrbi_x{scale}|_x{scale}|x{scale}|_lr|_LR|_sr|_SR)$", "", stem)
    return stem.lower()


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def crop_border(image: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return image
    height, width = image.shape[:2]
    if height <= border * 2 or width <= border * 2:
        return image
    return image[border:-border, border:-border, ...]


def rgb_to_y(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float64)
    return 16.0 + (65.481 * image[:, :, 0] + 128.553 * image[:, :, 1] + 24.966 * image[:, :, 2]) / 255.0


def calculate_psnr(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(data_range / math.sqrt(mse))


def calculate_ssim(a: np.ndarray, b: np.ndarray, data_range: float, channel_axis=None) -> float:
    return float(structural_similarity(a, b, data_range=data_range, channel_axis=channel_axis))


def align_reference_and_prediction(hr: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_height = min(hr.shape[0], pred.shape[0])
    target_width = min(hr.shape[1], pred.shape[1])
    if target_height <= 0 or target_width <= 0:
        raise ValueError(f"Invalid image shapes: hr={hr.shape[:2]}, pred={pred.shape[:2]}")
    return hr[:target_height, :target_width, :], pred[:target_height, :target_width, :]


def evaluate_pair(hr_path: Path, pred_path: Path, crop: int) -> dict:
    hr = load_rgb(hr_path)
    pred = load_rgb(pred_path)
    hr, pred = align_reference_and_prediction(hr, pred)

    hr_rgb = crop_border(hr, crop)
    pred_rgb = crop_border(pred, crop)
    hr_y = rgb_to_y(hr_rgb)
    pred_y = rgb_to_y(pred_rgb)

    return {
        "width": hr.shape[1],
        "height": hr.shape[0],
        "crop_border": crop,
        "psnr_rgb": calculate_psnr(hr_rgb, pred_rgb, 255.0),
        "ssim_rgb": calculate_ssim(hr_rgb, pred_rgb, 255.0, channel_axis=2),
        "psnr_y": calculate_psnr(hr_y, pred_y, 255.0),
        "ssim_y": calculate_ssim(hr_y, pred_y, 255.0),
    }


def parse_method_dirs(values: list[str]) -> dict[str, Path]:
    method_dirs = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid method spec: {value}. Expected METHOD=/path/to/outputs")
        method, folder = value.split("=", 1)
        method = method.strip()
        if not method:
            raise ValueError(f"Missing method name in spec: {value}")
        method_dirs[method] = Path(folder).expanduser().resolve()
    return method_dirs


def build_hr_index(dataset_root: Path, datasets: list[str]) -> dict[str, dict[str, Path]]:
    index = {}
    for dataset in datasets:
        hr_dir = dataset_root / dataset / "HR"
        hr_paths = list_images(hr_dir)
        if not hr_paths:
            raise FileNotFoundError(f"No HR images found in {hr_dir}")
        index[dataset] = {normalize_stem(path, scale=0): path for path in hr_paths}
    return index


def build_lr_index(dataset_root: Path, datasets: list[str], scale: int) -> dict[tuple[str, str], Path]:
    index = {}
    for dataset in datasets:
        lr_dir = dataset_root / dataset / "LR_bicubic" / f"X{scale}"
        for path in list_images(lr_dir):
            index[(dataset, normalize_stem(path, scale))] = path
    return index


def evaluate_method_outputs(
    method: str,
    output_root: Path,
    hr_index: dict[str, dict[str, Path]],
    lr_index: dict[tuple[str, str], Path],
    datasets: list[str],
    scale: int,
    crop: int,
) -> list[dict]:
    rows = []
    for dataset in datasets:
        pred_dir = output_root / dataset
        pred_paths = list_images(pred_dir)
        pred_index = {normalize_stem(path, scale, method): path for path in pred_paths}
        if not pred_paths:
            for base_name, hr_path in hr_index[dataset].items():
                rows.append({
                    "method": method,
                    "dataset": dataset,
                    "scale": scale,
                    "image_name": hr_path.name,
                    "base_name": base_name,
                    "hr_path": str(hr_path),
                    "lr_path": str(lr_index.get((dataset, base_name), "")),
                    "pred_path": "",
                    "error": "missing_prediction_dir_or_images",
                })
            continue

        for base_name, hr_path in tqdm(hr_index[dataset].items(), desc=f"{method} {dataset} X{scale}", leave=False):
            pred_path = pred_index.get(base_name)
            if pred_path is None:
                rows.append({
                    "method": method,
                    "dataset": dataset,
                    "scale": scale,
                    "image_name": hr_path.name,
                    "base_name": base_name,
                    "hr_path": str(hr_path),
                    "lr_path": str(lr_index.get((dataset, base_name), "")),
                    "pred_path": "",
                    "error": "missing_prediction_match",
                })
                continue
            metrics = evaluate_pair(hr_path, pred_path, crop)
            rows.append({
                "method": method,
                "dataset": dataset,
                "scale": scale,
                "image_name": hr_path.name,
                "base_name": base_name,
                "hr_path": str(hr_path),
                "lr_path": str(lr_index.get((dataset, base_name), "")),
                "pred_path": str(pred_path),
                "error": "",
                **metrics,
            })
    return rows


def load_bicubic_rows(path: Path, scale: int, datasets: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing bicubic CSV: {path}")
    df = pd.read_csv(path)
    df = df[(df["scale"] == scale) & (df["dataset"].isin(datasets))].copy()
    df.insert(0, "method", "Bicubic")
    df["pred_path"] = ""
    required = [
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
    ]
    return df[required]


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    valid = raw_df[raw_df["error"].fillna("") == ""].copy()
    grouped = valid.groupby(["method", "dataset", "scale"], as_index=False)
    summary = grouped.agg(
        n=("image_name", "count"),
        mean_psnr_y=("psnr_y", "mean"),
        mean_ssim_y=("ssim_y", "mean"),
        mean_psnr_rgb=("psnr_rgb", "mean"),
        mean_ssim_rgb=("ssim_rgb", "mean"),
        median_psnr_y=("psnr_y", "median"),
        median_ssim_y=("ssim_y", "median"),
        min_psnr_y=("psnr_y", "min"),
        max_psnr_y=("psnr_y", "max"),
    )
    return summary.sort_values(["scale", "dataset", "method"])


def calculate_improvement(raw_df: pd.DataFrame) -> pd.DataFrame:
    valid = raw_df[raw_df["error"].fillna("") == ""].copy()
    key_cols = ["dataset", "scale", "base_name"]
    baseline = valid[valid["method"] == "Bicubic"][key_cols + ["psnr_y", "ssim_y"]]
    baseline = baseline.rename(columns={"psnr_y": "bicubic_psnr_y", "ssim_y": "bicubic_ssim_y"})
    compared = valid[valid["method"] != "Bicubic"].merge(baseline, on=key_cols, how="inner")
    compared["delta_psnr_y"] = compared["psnr_y"] - compared["bicubic_psnr_y"]
    compared["delta_ssim_y"] = compared["ssim_y"] - compared["bicubic_ssim_y"]
    grouped = compared.groupby(["method", "dataset", "scale"], as_index=False)
    result = grouped.agg(
        n=("base_name", "count"),
        mean_delta_psnr_y=("delta_psnr_y", "mean"),
        mean_delta_ssim_y=("delta_ssim_y", "mean"),
        median_delta_psnr_y=("delta_psnr_y", "median"),
        win_rate_psnr_y=("delta_psnr_y", lambda values: float((values > 0).mean())),
        win_rate_ssim_y=("delta_ssim_y", lambda values: float((values > 0).mean())),
    )
    return result.sort_values(["scale", "dataset", "method"])


def select_cases(raw_df: pd.DataFrame, cases_per_type: int) -> pd.DataFrame:
    valid = raw_df[raw_df["error"].fillna("") == ""].copy()
    pivot = valid.pivot_table(
        index=["dataset", "scale", "base_name", "image_name", "hr_path", "lr_path"],
        columns="method",
        values=["psnr_y", "ssim_y"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{method}" for metric, method in pivot.columns]
    pivot = pivot.reset_index()

    rows = []
    if {"psnr_y_MambaIRv2", "psnr_y_SwinIR"}.issubset(pivot.columns):
        df = pivot.copy()
        df["case_type"] = "largest_mambairv2_gain_over_swinir"
        df["score"] = df["psnr_y_MambaIRv2"] - df["psnr_y_SwinIR"]
        rows.append(df.nlargest(cases_per_type, "score"))

        df = pivot.copy()
        df["case_type"] = "smallest_mambairv2_gain_over_swinir"
        df["score"] = df["psnr_y_MambaIRv2"] - df["psnr_y_SwinIR"]
        rows.append(df.nsmallest(cases_per_type, "score"))

    if {"psnr_y_MambaIRv2", "psnr_y_Bicubic"}.issubset(pivot.columns):
        df = pivot.copy()
        df["case_type"] = "largest_mambairv2_gain_over_bicubic"
        df["score"] = df["psnr_y_MambaIRv2"] - df["psnr_y_Bicubic"]
        rows.append(df.nlargest(cases_per_type, "score"))

    if "psnr_y_MambaIRv2" in pivot.columns:
        df = pivot.copy()
        df["case_type"] = "lowest_mambairv2_psnr_y"
        df["score"] = df["psnr_y_MambaIRv2"]
        rows.append(df.nsmallest(cases_per_type, "score"))

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["case_type", "dataset", "base_name"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="../datasets/imageSR")
    parser.add_argument("--method_dirs", nargs="+", required=True, help="METHOD=/path/to/visualization")
    parser.add_argument("--bicubic_csv", type=str, default="./results/sr_bicubic/raw_metrics.csv")
    parser.add_argument("--datasets", nargs="+", default=["Set5", "Set14", "B100", "Urban100", "Manga109"])
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--crop_border", type=str, default="scale", help="'scale' or an integer")
    parser.add_argument("--output_dir", type=str, default="./results/sr_model_comparison")
    parser.add_argument("--cases_per_type", type=int, default=8)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crop = args.scale if args.crop_border == "scale" else int(args.crop_border)

    method_dirs = parse_method_dirs(args.method_dirs)
    hr_index = build_hr_index(dataset_root, args.datasets)
    lr_index = build_lr_index(dataset_root, args.datasets, args.scale)

    frames = [load_bicubic_rows(Path(args.bicubic_csv), args.scale, args.datasets)]
    for method, output_root in method_dirs.items():
        rows = evaluate_method_outputs(method, output_root, hr_index, lr_index, args.datasets, args.scale, crop)
        frames.append(pd.DataFrame(rows))

    raw_df = pd.concat(frames, ignore_index=True)
    summary_df = summarize(raw_df)
    improvement_df = calculate_improvement(raw_df)
    cases_df = select_cases(raw_df, args.cases_per_type)

    raw_path = output_dir / f"raw_metrics_x{args.scale}.csv"
    summary_path = output_dir / f"summary_by_method_dataset_x{args.scale}.csv"
    improvement_path = output_dir / f"improvement_over_bicubic_x{args.scale}.csv"
    cases_path = output_dir / f"case_study_x{args.scale}.csv"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    improvement_df.to_csv(improvement_path, index=False)
    cases_df.to_csv(cases_path, index=False)

    print(f"Saved raw model metrics to {raw_path}")
    print(f"Saved model summary to {summary_path}")
    print(f"Saved improvements to {improvement_path}")
    print(f"Saved case study rows to {cases_path}")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
