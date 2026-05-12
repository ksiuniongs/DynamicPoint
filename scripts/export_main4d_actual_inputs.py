#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import center_crop


def load_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.float32) / 255.0


def load_mask(path):
    mask = Image.open(path)
    mask = np.array(mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask.astype(np.float32) / 255.0)[..., None]


def save_rgb(tensor, path):
    arr = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_mask(tensor, path):
    arr = tensor.detach().cpu().squeeze().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_overlay(img_t, mask_t, path):
    img = img_t.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    mask = mask_t.detach().cpu().squeeze().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    overlay = img.copy()
    overlay[..., 1] = np.maximum(overlay[..., 1], (mask * 255).astype(np.uint8))
    overlay[..., 2] = np.minimum(overlay[..., 2], 255 - (mask * 255).astype(np.uint8))
    Image.fromarray(overlay).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--stage1-global-motion", required=True)
    parser.add_argument("--ref-size", type=int, default=256)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    image_paths = sorted([p for p in Path(args.images).iterdir() if p.is_file()])
    mask_paths = sorted([p for p in Path(args.masks).iterdir() if p.is_file()])
    assert len(image_paths) == len(mask_paths), "image/mask count mismatch"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    actual_dir = outdir / "actual_inputs"
    actual_dir.mkdir(exist_ok=True)
    world_dir = outdir / "world_inputs_720p"
    world_dir.mkdir(exist_ok=True)

    with open(args.stage1_global_motion, "rb") as f:
        stage1_global_motion = pickle.load(f)
    input_scale0 = float(stage1_global_motion["scale"].squeeze().detach().cpu().item())

    sample = load_rgb(image_paths[0])
    height, width = sample.shape[:2]
    resize_factor = 720 / max(width, height) if max(width, height) > 720 else 1.0
    H_ = int(height * resize_factor)
    W_ = int(width * resize_factor)

    summary = []

    for idx, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        input_img = load_rgb(img_path)
        input_mask = load_mask(mask_path)
        input_depth = np.zeros_like(input_mask, dtype=np.float32)
        input_depth_mask = np.zeros_like(input_mask, dtype=np.float32)

        input_img_torch = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0)
        input_mask_torch = torch.from_numpy(input_mask).permute(2, 0, 1).unsqueeze(0)
        input_depth_torch = torch.from_numpy(input_depth).permute(2, 0, 1).unsqueeze(0)
        input_depth_mask_torch = torch.from_numpy(input_depth_mask).permute(2, 0, 1).unsqueeze(0)

        # World-frame inputs resized to 720p, matching main_4d.
        input_img_world = F.interpolate(input_img_torch, (H_, W_), mode="bilinear", align_corners=False)
        input_mask_world = F.interpolate(input_mask_torch, (H_, W_), mode="nearest")

        save_rgb(input_img_world, world_dir / f"{idx:04d}.png")
        save_mask(input_mask_world, world_dir / f"{idx:04d}_mask.png")
        save_overlay(input_img_world, input_mask_world, world_dir / f"{idx:04d}_overlay.png")

        N, C, H, W = input_mask_torch.shape
        mask = input_mask_torch > 0.5
        nonzero_idxes = torch.nonzero(mask[0, 0])

        frame_summary = {"frame": idx, "source_image": str(img_path), "source_mask": str(mask_path)}

        if len(nonzero_idxes) > 0:
            min_x = nonzero_idxes[:, 1].min()
            max_x = nonzero_idxes[:, 1].max()
            min_y = nonzero_idxes[:, 0].min()
            max_y = nonzero_idxes[:, 0].max()
            cx = (max_x + min_x) / 2
            cx_norm = (cx / W) * 2 - 1
            cy = (max_y + min_y) / 2
            cy_norm = (cy / H) * 2 - 1
            width_box = (max_x - min_x) / W
            height_box = (max_y - min_y) / H
            scale_x = width_box / 0.975
            scale_y = height_box / 0.975
            max_scale = max(scale_x, scale_y)
            scale = max(max_scale, input_scale0)

            theta = torch.tensor([[[scale, 0.0, cx_norm], [0.0, scale, cy_norm]]], dtype=torch.float32)
            resize_factor_local = args.ref_size / min(H, W)
            grid = F.affine_grid(theta, (N, C, int(H * resize_factor_local), int(W * resize_factor_local)), align_corners=True)

            input_img_torch[:, :, 0] = 1.0
            input_img_torch[:, :, -1] = 1.0
            input_img_torch[:, :, :, 0] = 1.0
            input_img_torch[:, :, :, -1] = 1.0

            input_img_actual = F.grid_sample(input_img_torch, grid, align_corners=True, padding_mode="border")
            input_mask_actual = F.grid_sample(input_mask_torch, grid, align_corners=True)
            input_depth_actual = F.grid_sample(input_depth_torch, grid, mode="nearest", align_corners=True)
            input_depth_mask_actual = F.grid_sample(input_depth_mask_torch, grid, mode="nearest", align_corners=True)

            input_img_actual = center_crop(input_img_actual, args.ref_size)
            input_mask_actual = center_crop(input_mask_actual, args.ref_size)
            input_depth_actual = center_crop(input_depth_actual, args.ref_size)
            input_depth_mask_actual = center_crop(input_depth_mask_actual, args.ref_size)

            frame_summary.update(
                {
                    "has_mask": True,
                    "cx_norm": float(cx_norm),
                    "cy_norm": float(cy_norm),
                    "scale": float(scale),
                }
            )
        else:
            input_img_actual = torch.zeros((1, 3, args.ref_size, args.ref_size), dtype=torch.float32)
            input_mask_actual = torch.zeros((1, 1, args.ref_size, args.ref_size), dtype=torch.float32)
            input_depth_actual = torch.zeros((1, 1, args.ref_size, args.ref_size), dtype=torch.float32)
            input_depth_mask_actual = torch.zeros((1, 1, args.ref_size, args.ref_size), dtype=torch.float32)
            frame_summary["has_mask"] = False

        save_rgb(input_img_actual, actual_dir / f"{idx:04d}.png")
        save_mask(input_mask_actual, actual_dir / f"{idx:04d}_mask.png")
        save_overlay(input_img_actual, input_mask_actual, actual_dir / f"{idx:04d}_overlay.png")
        summary.append(frame_summary)

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
