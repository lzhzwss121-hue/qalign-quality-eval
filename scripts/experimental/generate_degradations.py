import os
import argparse
from pathlib import Path

import cv2
import numpy as np


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMG_EXTENSIONS)


def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_gaussian_blur(img: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return blurred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr_dir", type=str, default="./data/images/hr")
    parser.add_argument("--blur_dir", type=str, default="./data/images/blur")
    parser.add_argument("--noise_dir", type=str, default="./data/images/noise")
    parser.add_argument("--blur_ksize", type=int, default=7)
    parser.add_argument("--blur_sigma", type=float, default=1.6)
    parser.add_argument("--noise_sigma", type=float, default=15.0)
    args = parser.parse_args()

    os.makedirs(args.blur_dir, exist_ok=True)
    os.makedirs(args.noise_dir, exist_ok=True)

    image_files = [f for f in os.listdir(args.hr_dir) if is_image_file(f)]
    if len(image_files) == 0:
        raise RuntimeError(f"No images found in {args.hr_dir}")

    print(f"Found {len(image_files)} HR images")

    for fname in sorted(image_files):
        hr_path = os.path.join(args.hr_dir, fname)
        img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[Skip] Failed to read: {hr_path}")
            continue

        blur_img = add_gaussian_blur(img, args.blur_ksize, args.blur_sigma)
        noise_img = add_gaussian_noise(img, args.noise_sigma)

        blur_path = os.path.join(args.blur_dir, fname)
        noise_path = os.path.join(args.noise_dir, fname)

        cv2.imwrite(blur_path, blur_img)
        cv2.imwrite(noise_path, noise_img)

    print(f"Saved blur images to: {args.blur_dir}")
    print(f"Saved noise images to: {args.noise_dir}")


if __name__ == "__main__":
    main()
