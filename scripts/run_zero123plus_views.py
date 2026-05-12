#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def expand2square(img: Image.Image, background_color: tuple[int, int, int]) -> Image.Image:
    width, height = img.size
    if width == height:
        return img
    side = max(width, height)
    result = Image.new("RGB", (side, side), background_color)
    result.paste(img, ((side - width) // 2, (side - height) // 2))
    return result


def load_mask(mask_path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    if mask.size != size:
        raise ValueError(f"Mask size {mask.size} does not match image size {size}.")
    return np.array(mask) > 0


def apply_mask_background(
    img: Image.Image, mask_bin: np.ndarray, background: tuple[int, int, int]
) -> Image.Image:
    rgb = np.array(img.convert("RGB"))
    rgb[~mask_bin] = np.asarray(background, dtype=np.uint8)
    return Image.fromarray(rgb)


def split_zero123plus_grid(grid: Image.Image) -> list[Image.Image]:
    # Official Zero123++ layout is 2 columns x 3 rows, each 320x320.
    tile = 320
    cols, rows = 2, 3
    expected = (tile * cols, tile * rows)
    if grid.size != expected:
        raise ValueError(f"Expected Zero123++ grid size {expected}, got {grid.size}.")
    views: list[Image.Image] = []
    for r in range(rows):
        for c in range(cols):
            left = c * tile
            top = r * tile
            views.append(grid.crop((left, top, left + tile, top + tile)))
    return views


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mask-path", type=Path, default=None)
    parser.add_argument("--background", choices=["white", "black", "green"], default="white")
    parser.add_argument("--steps", type=int, default=36)
    parser.add_argument("--model-id", default="sudo-ai/zero123plus-v1.2")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    bg_map = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "green": (0, 255, 0),
    }
    bg = bg_map[args.background]

    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
    import torch

    args.output_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(args.input).convert("RGB")
    if args.mask_path is not None:
        mask_bin = load_mask(args.mask_path, img.size)
        img = apply_mask_background(img, mask_bin, bg)

    square = expand2square(img, bg)
    square.save(args.output_dir / "input_square.png")

    pipeline = DiffusionPipeline.from_pretrained(
        args.model_id,
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16 if "cuda" in args.device else torch.float32,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing="trailing"
    )
    pipeline.to(args.device)

    result = pipeline(square, num_inference_steps=args.steps).images[0]
    result.save(args.output_dir / "zero123plus_grid.png")

    azimuths = [30, 90, 150, 210, 270, 330]
    views = split_zero123plus_grid(result)
    for idx, (az, view) in enumerate(zip(azimuths, views)):
        view.save(args.output_dir / f"view_{idx:02d}_az{az:03d}.png")


if __name__ == "__main__":
    main()
