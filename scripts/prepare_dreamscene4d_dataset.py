#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def collect_images(path: Path) -> list[Path]:
    files = [p for p in sorted(path.iterdir()) if p.is_file()]
    return files


def save_rgb_png(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)


def save_binary_mask_png(src: Path, dst: Path) -> None:
    mask = Image.open(src).convert("L")
    # Keep masks strictly binary for DreamScene4D.
    mask = mask.point(lambda p: 255 if p > 0 else 0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mask.save(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DreamScene4D JPEGImages/Annotations dataset.")
    parser.add_argument("--image_dir", required=True, help="Source frame directory.")
    parser.add_argument("--mask_dir", required=True, help="Source mask directory.")
    parser.add_argument("--output_root", required=True, help="DreamScene4D data root.")
    parser.add_argument("--video_name", required=True, help="Target video name under JPEGImages/Annotations.")
    parser.add_argument("--object_id", default="001", help="DreamScene4D object id directory name.")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_root = Path(args.output_root)

    images = collect_images(image_dir)
    masks = sorted(mask_dir.glob("*.png"))

    if len(images) == 0:
        raise SystemExit(f"No images found in {image_dir}")
    if len(images) != len(masks):
        raise SystemExit(f"Image/mask count mismatch: {len(images)} images vs {len(masks)} masks")

    jpeg_dir = output_root / "JPEGImages" / args.video_name
    anno_dir = output_root / "Annotations" / args.video_name / args.object_id
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    anno_dir.mkdir(parents=True, exist_ok=True)

    for idx, (img_src, mask_src) in enumerate(zip(images, masks)):
        name = f"{idx:05d}.png"
        save_rgb_png(img_src, jpeg_dir / name)
        save_binary_mask_png(mask_src, anno_dir / name)

    print(f"Prepared {len(images)} frames")
    print(f"JPEGImages: {jpeg_dir}")
    print(f"Annotations: {anno_dir}")


if __name__ == "__main__":
    main()
