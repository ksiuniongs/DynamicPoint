#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def load_mask(mask_path):
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask.astype(np.float32) / 255.0) > 0.5


def compute_bbox_stats(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    bw = max_x - min_x + 1
    bh = max_y - min_y + 1
    return {
        "cx_px": float(cx),
        "cy_px": float(cy),
        "cx_norm": float((cx / w) * 2 - 1),
        "cy_norm": float((cy / h) * 2 - 1),
        "width_px": int(bw),
        "height_px": int(bh),
        "scale_norm": float(max((bw / w) / 0.975, (bh / h) / 0.975)),
    }


def flow_to_rgb(flow):
    fx = flow[0]
    fy = flow[1]
    mag = np.sqrt(fx ** 2 + fy ** 2)
    ang = np.arctan2(fy, fx)
    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.float32)
    hsv[..., 0] = (ang + np.pi) / (2 * np.pi)
    hsv[..., 1] = 1.0
    hsv[..., 2] = np.clip(mag / (np.percentile(mag, 95) + 1e-6), 0, 1)
    import colorsys
    rgb = np.zeros_like(hsv)
    for y in range(hsv.shape[0]):
        for x in range(hsv.shape[1]):
            rgb[y, x] = colorsys.hsv_to_rgb(*hsv[y, x])
    return (rgb * 255).astype(np.uint8)


def save_bbox_preview(image_path, stats, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    if stats is not None:
        cx = stats["cx_px"]
        cy = stats["cy_px"]
        bw = stats["width_px"]
        bh = stats["height_px"]
        x0 = cx - bw / 2
        y0 = cy - bh / 2
        x1 = cx + bw / 2
        y1 = cy + bh / 2
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=4)
        draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(0, 255, 0))
    img.save(out_path)


def plot_mask_stats(stats_list, out_path):
    centers_x = [s["cx_norm"] if s else np.nan for s in stats_list]
    centers_y = [s["cy_norm"] if s else np.nan for s in stats_list]
    scales = [s["scale_norm"] if s else np.nan for s in stats_list]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(centers_x, marker="o")
    axes[0].set_ylabel("cx_norm")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(centers_y, marker="o")
    axes[1].set_ylabel("cy_norm")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(scales, marker="o")
    axes[2].set_ylabel("scale_norm")
    axes[2].set_xlabel("frame")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle("Mask trajectory diagnostics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_motion(global_motion, out_path):
    trans = global_motion["translation"].detach().cpu().numpy()
    scale = global_motion["scale"].detach().cpu().numpy()

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(trans[:, 0], marker="o")
    axes[0].set_ylabel("tx")
    axes[1].plot(trans[:, 1], marker="o")
    axes[1].set_ylabel("ty")
    axes[2].plot(trans[:, 2], marker="o")
    axes[2].set_ylabel("tz")
    axes[3].plot(scale, marker="o")
    axes[3].set_ylabel("scale")
    axes[3].set_xlabel("frame")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("4D residual global motion")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def export_flow_previews(dreamscene_root, image_paths, mask_paths, out_dir, max_pairs):
    sys.path.insert(0, dreamscene_root)
    import torch
    from gmflow.gmflow.gmflow import GMFlow
    from utils.flow_utils import run_flow_on_images

    device = "cuda"
    imgs = []
    masks = []
    for image_path, mask_path in zip(image_paths, mask_paths):
        img = np.array(Image.open(image_path).convert("RGB")).astype(np.float32) / 255.0
        mask = load_mask(mask_path).astype(np.float32)[..., None]
        masked = img * mask + (1 - mask)
        tensor = torch.from_numpy(masked).permute(2, 0, 1).unsqueeze(0)
        imgs.append(tensor)
        masks.append(mask)
    images = torch.cat(imgs, dim=0)

    flow_predictor = GMFlow(
        feature_channels=128,
        num_scales=1,
        upsample_factor=8,
        num_head=1,
        attention_type='swin',
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        attn_splits_list=[2],
        corr_radius_list=[-1],
        prop_radius_list=[-1],
    )
    checkpoint = torch.load(os.path.join(dreamscene_root, "gmflow/pretrained/gmflow_kitti-285701a8.pth"))
    weights = checkpoint["model"] if "model" in checkpoint else checkpoint
    flow_predictor.load_state_dict(weights)
    flow_predictor.eval().to(device)

    with torch.no_grad():
        fwd_flows, _, fwd_valids, _ = run_flow_on_images(flow_predictor, images)

    num_pairs = min(max_pairs, len(fwd_flows))
    for i in range(num_pairs):
        flow = fwd_flows[i].detach().cpu().numpy()
        valid = fwd_valids[i].squeeze(0).detach().cpu().numpy()
        rgb = flow_to_rgb(flow)
        rgb[valid < 0.5] = 0
        Image.fromarray(rgb).save(os.path.join(out_dir, f"flow_pair_{i:04d}_{i+1:04d}.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--global-motion", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--dreamscene-root", default="/mnt/d/develop/4D/submodules/dreamscene4d")
    parser.add_argument("--max-flow-pairs", type=int, default=6)
    parser.add_argument("--skip-flow", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(str(p) for p in Path(args.images).glob("*"))
    mask_paths = sorted(str(p) for p in Path(args.masks).glob("*"))

    stats_list = []
    for idx, mask_path in enumerate(mask_paths):
        stats = compute_bbox_stats(load_mask(mask_path))
        stats_list.append(stats)
        if idx < 3:
            save_bbox_preview(image_paths[idx], stats, outdir / f"bbox_preview_{idx:04d}.png")

    plot_mask_stats(stats_list, outdir / "mask_trajectory.png")

    with open(args.global_motion, "rb") as f:
        global_motion = pickle.load(f)
    plot_motion(global_motion, outdir / "residual_motion.png")

    summary = {
        "num_frames": len(image_paths),
        "has_base_transform": "base_translation" in global_motion,
        "global_motion_keys": sorted(global_motion.keys()),
        "mask_stats_first_frame": stats_list[0],
        "mask_stats_last_frame": stats_list[-1],
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not args.skip_flow:
        export_flow_previews(args.dreamscene_root, image_paths, mask_paths, str(outdir), args.max_flow_pairs)


if __name__ == "__main__":
    main()
