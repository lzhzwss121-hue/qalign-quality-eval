import argparse
import math
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM


KEY_COLUMNS = ["method", "dataset", "scale", "base_name"]


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def resize_bicubic(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(image)
    return np.asarray(pil.resize(size, Image.Resampling.BICUBIC))


def crop_border(image: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return image
    height, width = image.shape[:2]
    if height <= border * 2 or width <= border * 2:
        return image
    return image[border:-border, border:-border, :]


def align_reference_and_prediction(hr: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_height = min(hr.shape[0], pred.shape[0])
    target_width = min(hr.shape[1], pred.shape[1])
    if target_height <= 0 or target_width <= 0:
        raise ValueError(f"Invalid image shapes: hr={hr.shape[:2]}, pred={pred.shape[:2]}")
    return hr[:target_height, :target_width, :], pred[:target_height, :target_width, :]


def build_prediction(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    hr = load_rgb(row["hr_path"])
    if row["method"] == "Bicubic":
        lr = load_rgb(row["lr_path"])
        target_size = (int(row["width"]), int(row["height"]))
        hr = hr[: target_size[1], : target_size[0], :]
        pred = resize_bicubic(lr, target_size)
        return hr, pred

    pred_path = row.get("pred_path", "")
    if not isinstance(pred_path, str) or not pred_path:
        raise ValueError("Missing prediction path for non-bicubic row")
    pred = load_rgb(pred_path)
    return align_reference_and_prediction(hr, pred)


def to_lpips_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(image).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device)


def choose_device(preferred: str) -> torch.device:
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preferred)


def load_qalign_model(model_name: str, device: torch.device):
    print(f"Loading Q-Align model: {model_name}")
    kwargs = {"trust_remote_code": True}
    if device.type == "cuda":
        kwargs.update({"torch_dtype": torch.float16, "device_map": "auto"})
    else:
        kwargs.update({"torch_dtype": torch.float32})

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    return model


def score_qalign(model, image: np.ndarray) -> float:
    pil = Image.fromarray(image).convert("RGB")
    score = model.score([pil], task_="quality", input_="image")
    if isinstance(score, list):
        score = score[0]
    if torch.is_tensor(score):
        return float(score.item())
    return float(score)


def score_lpips(model, hr: np.ndarray, pred: np.ndarray, crop: int, device: torch.device) -> float:
    hr_crop = crop_border(hr, crop)
    pred_crop = crop_border(pred, crop)
    with torch.no_grad():
        hr_tensor = to_lpips_tensor(hr_crop, device)
        pred_tensor = to_lpips_tensor(pred_crop, device)
        return float(model(hr_tensor, pred_tensor).item())


def row_key(row: pd.Series) -> tuple:
    return tuple(row[col] for col in KEY_COLUMNS)


def load_completed(output_csv: Path) -> set[tuple]:
    if not output_csv.exists():
        return set()
    done = pd.read_csv(output_csv)
    if done.empty:
        return set()
    return {tuple(row[col] for col in KEY_COLUMNS) for _, row in done.iterrows()}


def append_row(output_csv: Path, row: dict):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    frame.to_csv(output_csv, mode="a", index=False, header=not output_csv.exists())


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    valid = raw[raw["error"].fillna("") == ""].copy()
    grouped = valid.groupby(["method", "dataset", "scale"], as_index=False)
    return grouped.agg(
        n=("base_name", "count"),
        mean_psnr_y=("psnr_y", "mean"),
        mean_ssim_y=("ssim_y", "mean"),
        mean_lpips=("lpips", "mean"),
        mean_qalign=("qalign_score", "mean"),
        median_lpips=("lpips", "median"),
        median_qalign=("qalign_score", "median"),
    ).sort_values(["scale", "dataset", "method"])


def correlations(raw: pd.DataFrame) -> pd.DataFrame:
    valid = raw[raw["error"].fillna("") == ""].copy()
    rows = []
    groups = [("all", valid)]
    groups.extend((dataset, group) for dataset, group in valid.groupby("dataset"))
    for group_name, group in groups:
        for metric in ["psnr_y", "ssim_y", "lpips"]:
            if len(group) < 3:
                continue
            rows.append({
                "group": group_name,
                "metric_x": metric,
                "metric_y": "qalign_score",
                "pearson": group[metric].corr(group["qalign_score"], method="pearson"),
                "spearman": group[metric].corr(group["qalign_score"], method="spearman"),
                "n": len(group),
            })
    return pd.DataFrame(rows)


def disagreement_cases(raw: pd.DataFrame, cases_per_type: int) -> pd.DataFrame:
    valid = raw[raw["error"].fillna("") == ""].copy()
    rows = []
    if valid.empty:
        return pd.DataFrame()

    z = valid.copy()
    for col in ["psnr_y", "ssim_y", "lpips", "qalign_score"]:
        std = z[col].std()
        z[f"{col}_z"] = 0.0 if std == 0 or math.isnan(std) else (z[col] - z[col].mean()) / std

    high_psnr_low_qalign = z.copy()
    high_psnr_low_qalign["case_type"] = "high_psnr_low_qalign"
    high_psnr_low_qalign["score"] = high_psnr_low_qalign["psnr_y_z"] - high_psnr_low_qalign["qalign_score_z"]
    rows.append(high_psnr_low_qalign.nlargest(cases_per_type, "score"))

    low_psnr_high_qalign = z.copy()
    low_psnr_high_qalign["case_type"] = "low_psnr_high_qalign"
    low_psnr_high_qalign["score"] = low_psnr_high_qalign["qalign_score_z"] - low_psnr_high_qalign["psnr_y_z"]
    rows.append(low_psnr_high_qalign.nlargest(cases_per_type, "score"))

    low_lpips_low_qalign = z.copy()
    low_lpips_low_qalign["case_type"] = "low_lpips_low_qalign"
    low_lpips_low_qalign["score"] = -low_lpips_low_qalign["lpips_z"] - low_lpips_low_qalign["qalign_score_z"]
    rows.append(low_lpips_low_qalign.nlargest(cases_per_type, "score"))

    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="./results/sr_model_comparison/raw_metrics_x4.csv")
    parser.add_argument("--output_csv", type=str, default="./results/sr_model_qalign/raw_metrics_x4.csv")
    parser.add_argument("--summary_csv", type=str, default="./results/sr_model_qalign/summary_by_method_dataset_x4.csv")
    parser.add_argument("--correlation_csv", type=str, default="./results/sr_model_qalign/correlations_x4.csv")
    parser.add_argument("--case_csv", type=str, default="./results/sr_model_qalign/disagreement_cases_x4.csv")
    parser.add_argument("--qalign_model", type=str, default="q-future/one-align")
    parser.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cases_per_type", type=int, default=12)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    raw = pd.read_csv(input_csv)
    raw = raw[raw["error"].fillna("") == ""].copy()
    if args.methods:
        raw = raw[raw["method"].isin(args.methods)].copy()
    if args.datasets:
        raw = raw[raw["dataset"].isin(args.datasets)].copy()

    if args.resume:
        completed = load_completed(output_csv)
        raw = raw[[row_key(row) not in completed for _, row in raw.iterrows()]].copy()
        print(f"Skipping {len(completed)} completed rows from {output_csv}")
    elif output_csv.exists():
        output_csv.unlink()

    if args.max_images is not None:
        raw = raw.head(args.max_images).copy()

    if raw.empty:
        print("No rows to process.")
    else:
        device = choose_device(args.device)
        print(f"Using device: {device}")
        qalign_model = load_qalign_model(args.qalign_model, device)
        lpips_model = lpips.LPIPS(net=args.lpips_net).to(device)
        lpips_model.eval()

        for _, row in tqdm(raw.iterrows(), total=len(raw), desc="Q-Align/LPIPS"):
            result = row.to_dict()
            try:
                hr, pred = build_prediction(row)
                crop = int(row["crop_border"])
                result["lpips"] = score_lpips(lpips_model, hr, pred, crop, device)
                result["qalign_score"] = score_qalign(qalign_model, pred)
                result["error"] = ""
            except Exception as exc:
                result["lpips"] = np.nan
                result["qalign_score"] = np.nan
                result["error"] = str(exc)
            append_row(output_csv, result)

    full = pd.read_csv(output_csv)
    summary = summarize(full)
    corr = correlations(full)
    cases = disagreement_cases(full, args.cases_per_type)

    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    corr.to_csv(args.correlation_csv, index=False)
    cases.to_csv(args.case_csv, index=False)

    print(f"Saved raw Q-Align metrics to {output_csv}")
    print(f"Saved summary to {args.summary_csv}")
    print(f"Saved correlations to {args.correlation_csv}")
    print(f"Saved disagreement cases to {args.case_csv}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
