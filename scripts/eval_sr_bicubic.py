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


def normalize_stem(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"(_LRBI_x\d+|_lrbi_x\d+|_x\d+|x\d+|_lr|_LR|_sr|_SR)$", "", stem)
    return stem.lower()


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def resize_bicubic(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(image)
    return np.asarray(pil.resize(size, Image.Resampling.BICUBIC))


def mod_crop_to_lr_scale(hr: np.ndarray, lr: np.ndarray, scale: int) -> np.ndarray:
    target_height = lr.shape[0] * scale
    target_width = lr.shape[1] * scale
    if hr.shape[0] < target_height or hr.shape[1] < target_width:
        raise ValueError(
            f"HR image is smaller than LR*scale target: "
            f"hr={hr.shape[:2]}, target={(target_height, target_width)}"
        )
    return hr[:target_height, :target_width, :]


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


def find_lr_image(lr_images: dict[str, Path], hr_path: Path) -> Path | None:
    return lr_images.get(normalize_stem(hr_path))


def evaluate_pair(hr_path: Path, lr_path: Path, scale: int, crop: int) -> dict:
    hr = load_rgb(hr_path)
    lr = load_rgb(lr_path)
    hr = mod_crop_to_lr_scale(hr, lr, scale)
    pred = resize_bicubic(lr, (hr.shape[1], hr.shape[0]))

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


def evaluate_dataset(dataset_root: Path, dataset: str, scale: int, crop_border_mode: str) -> list[dict]:
    dataset_dir = dataset_root / dataset
    hr_dir = dataset_dir / "HR"
    lr_dir = dataset_dir / "LR_bicubic" / f"X{scale}"

    hr_images = list_images(hr_dir)
    lr_images = {normalize_stem(path): path for path in list_images(lr_dir)}

    if not hr_images:
        raise FileNotFoundError(f"No HR images found in {hr_dir}")
    if not lr_images:
        raise FileNotFoundError(f"No LR images found in {lr_dir}")

    crop = scale if crop_border_mode == "scale" else int(crop_border_mode)
    rows = []

    for hr_path in tqdm(hr_images, desc=f"{dataset} X{scale}", leave=False):
        lr_path = find_lr_image(lr_images, hr_path)
        if lr_path is None:
            rows.append({
                "dataset": dataset,
                "scale": scale,
                "image_name": hr_path.name,
                "base_name": normalize_stem(hr_path),
                "hr_path": str(hr_path),
                "lr_path": None,
                "error": "missing_lr_match",
            })
            continue

        metrics = evaluate_pair(hr_path, lr_path, scale, crop)
        rows.append({
            "dataset": dataset,
            "scale": scale,
            "image_name": hr_path.name,
            "base_name": normalize_stem(hr_path),
            "hr_path": str(hr_path),
            "lr_path": str(lr_path),
            "error": "",
            **metrics,
        })

    return rows


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    valid = raw_df[raw_df["error"].fillna("") == ""].copy()
    grouped = valid.groupby(["dataset", "scale"], as_index=False)
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
    return summary.sort_values(["dataset", "scale"])


def select_failure_cases(raw_df: pd.DataFrame, cases_per_group: int) -> pd.DataFrame:
    valid = raw_df[raw_df["error"].fillna("") == ""].copy()
    rows = []
    for (dataset, scale), group in valid.groupby(["dataset", "scale"]):
        worst_psnr = group.nsmallest(cases_per_group, "psnr_y").copy()
        worst_psnr["case_type"] = "lowest_psnr_y"
        worst_ssim = group.nsmallest(cases_per_group, "ssim_y").copy()
        worst_ssim["case_type"] = "lowest_ssim_y"
        rows.extend([worst_psnr, worst_ssim])
    return pd.concat(rows, ignore_index=True).sort_values(["dataset", "scale", "case_type", "psnr_y"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="../datasets/imageSR")
    parser.add_argument("--datasets", nargs="+", default=["Set5", "Set14", "B100", "Urban100", "Manga109"])
    parser.add_argument("--scales", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--crop_border", type=str, default="scale", help="'scale' or an integer")
    parser.add_argument("--output_dir", type=str, default="./results/sr_bicubic")
    parser.add_argument("--cases_per_group", type=int, default=2)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for dataset in args.datasets:
        for scale in args.scales:
            all_rows.extend(evaluate_dataset(dataset_root, dataset, scale, args.crop_border))

    raw_df = pd.DataFrame(all_rows)
    summary_df = summarize(raw_df)
    failure_df = select_failure_cases(raw_df, args.cases_per_group)

    raw_path = output_dir / "raw_metrics.csv"
    summary_path = output_dir / "summary_by_dataset_scale.csv"
    failure_path = output_dir / "failure_cases.csv"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    failure_df.to_csv(failure_path, index=False)

    print(f"Saved raw metrics to {raw_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved failure cases to {failure_path}")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
