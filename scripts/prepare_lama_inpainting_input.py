#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def collect_images(images_dir: Path):
    return sorted([p for p in images_dir.iterdir() if p.is_file()])


def frame_id_from_image(path: Path) -> str:
    stem = path.stem
    if stem.startswith("frame_"):
        return stem.split("frame_", 1)[1]
    return stem


def expected_mask_path(masks_dir: Path, frame_id: str) -> Path:
    return masks_dir / f"{frame_id}_obj000.png"


def binarize_mask(src: Path, dst: Path):
    mask = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {src}")
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    cv2.imwrite(str(dst), binary)


def binarize_and_dilate_mask(src: Path, dst: Path, dilate_pixels: int):
    mask = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {src}")

    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if dilate_pixels > 0:
        kernel_size = dilate_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.dilate(binary, kernel, iterations=1)

    cv2.imwrite(str(dst), binary)


def main():
    parser = argparse.ArgumentParser(description="Prepare LaMa input directory from images and masks.")
    parser.add_argument("--images", required=True, help="Image directory")
    parser.add_argument("--masks", required=True, help="Mask directory")
    parser.add_argument("--outdir", required=True, help="LaMa input directory")
    parser.add_argument("--prefix", required=True, help="Prefix to attach, e.g. left or right")
    parser.add_argument(
        "--dilate_pixels",
        type=int,
        default=0,
        help="Dilate binary masks by this many pixels before writing mask001 files",
    )
    args = parser.parse_args()

    images_dir = Path(args.images)
    masks_dir = Path(args.masks)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(images_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    for image_path in image_paths:
        frame_id = frame_id_from_image(image_path)
        mask_path = expected_mask_path(masks_dir, frame_id)
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for frame {frame_id}: {mask_path}")

        image_out = outdir / f"{args.prefix}_{frame_id}{image_path.suffix.lower()}"
        mask_out = outdir / f"{args.prefix}_{frame_id}_mask001.png"

        shutil.copy2(image_path, image_out)
        binarize_and_dilate_mask(mask_path, mask_out, args.dilate_pixels)


if __name__ == "__main__":
    main()
