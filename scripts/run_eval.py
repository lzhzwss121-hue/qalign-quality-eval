import os
import re
import math
import argparse
from pathlib import Path

import cv2
import lpips
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from skimage.metrics import structural_similarity as ssim


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMG_EXTENSIONS)


def normalize_name(filename: str) -> str:
    """
    Normalize file names for robust matching.
    Examples:
        zebrax4.png   -> zebra
        zebra_x4.png  -> zebra
        zebra_LR.png  -> zebra
    """
    stem = Path(filename).stem
    stem = re.sub(r'(_x\d+|x\d+|_lr|_LR|_sr|_SR)$', '', stem)
    return stem.lower()


def load_image_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_if_needed(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.shape[:2] != gt.shape[:2]:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
    return pred


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    return ssim(img1, img2, channel_axis=2, data_range=255)


def to_lpips_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert RGB uint8 [H, W, C] in [0,255]
    -> torch tensor [1, 3, H, W] in [-1,1]
    """
    tensor = torch.from_numpy(img).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device)


def build_hr_index(hr_dir: str) -> dict:
    hr_index = {}
    for fname in os.listdir(hr_dir):
        if is_image_file(fname):
            key = normalize_name(fname)
            hr_index[key] = os.path.join(hr_dir, fname)
    return hr_index


def get_qalign_model(model_name: str):
    print(f"Loading Q-Align model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model


def score_qalign(model, image_path: str) -> float:
    img_pil = Image.open(image_path).convert("RGB")
    score = model.score([img_pil], task_="quality", input_="image")
    if isinstance(score, list):
        score = score[0]
    if torch.is_tensor(score):
        return float(score.item())
    return float(score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/images")
    parser.add_argument("--output_csv", type=str, default="./results/raw_metrics.csv")
    parser.add_argument(
        "--degradations",
        nargs="+",
        default=["bicubic", "blur", "noise"],
        help="Folders under data_root"
    )
    parser.add_argument("--qalign_model", type=str, default="q-future/one-align")
    parser.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])
    args = parser.parse_args()

    hr_dir = os.path.join(args.data_root, "hr")
    if not os.path.isdir(hr_dir):
        raise FileNotFoundError(f"HR directory not found: {hr_dir}")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    hr_index = build_hr_index(hr_dir)
    if len(hr_index) == 0:
        raise RuntimeError(f"No HR images found in {hr_dir}")

    # Load models
    qalign_model = get_qalign_model(args.qalign_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net=args.lpips_net).to(device)
    lpips_model.eval()

    rows = []
    error_rows = []

    for degradation in args.degradations:
        pred_dir = os.path.join(args.data_root, degradation)
        if not os.path.isdir(pred_dir):
            print(f"[Skip] Missing directory: {pred_dir}")
            continue

        image_files = [f for f in os.listdir(pred_dir) if is_image_file(f)]
        if len(image_files) == 0:
            print(f"[Skip] No images in: {pred_dir}")
            continue

        print(f"\n=== Processing {degradation}: {len(image_files)} images ===")

        for fname in tqdm(sorted(image_files), desc=degradation):
            pred_path = os.path.join(pred_dir, fname)
            key = normalize_name(fname)

            gt_path = hr_index.get(key)
            if gt_path is None:
                msg = f"No HR match found for {fname}"
                print(f"[Warn] {msg}")
                error_rows.append({"image_name": fname, "degradation": degradation, "error": msg})
                continue

            try:
                gt = load_image_rgb(gt_path)
                pred = load_image_rgb(pred_path)
                pred = resize_if_needed(pred, gt)

                psnr_val = calculate_psnr(gt, pred)
                ssim_val = calculate_ssim(gt, pred)

                with torch.no_grad():
                    gt_tensor = to_lpips_tensor(gt, device)
                    pred_tensor = to_lpips_tensor(pred, device)
                    lpips_val = float(lpips_model(gt_tensor, pred_tensor).item())

                qalign_val = score_qalign(qalign_model, pred_path)

                rows.append({
                    "image_name": fname,
                    "base_name": key,
                    "degradation": degradation,
                    "gt_path": gt_path,
                    "pred_path": pred_path,
                    "width": gt.shape[1],
                    "height": gt.shape[0],
                    "psnr": round(psnr_val, 6),
                    "ssim": round(ssim_val, 6),
                    "lpips": round(lpips_val, 6),
                    "qalign_score": round(qalign_val, 6),
                })

            except Exception as e:
                msg = str(e)
                print(f"[Error] {fname}: {msg}")
                error_rows.append({"image_name": fname, "degradation": degradation, "error": msg})

    if len(rows) == 0:
        raise RuntimeError("No valid results collected. Please check paths and image matching.")

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)

    print("\nSaved metrics to:", args.output_csv)
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nPer-degradation mean:")
    print(df.groupby("degradation")[["psnr", "ssim", "lpips", "qalign_score"]].mean())

    if len(error_rows) > 0:
        error_csv = args.output_csv.replace(".csv", "_errors.csv")
        pd.DataFrame(error_rows).to_csv(error_csv, index=False)
        print(f"\nSaved error log to: {error_csv}")


if __name__ == "__main__":
    main()
