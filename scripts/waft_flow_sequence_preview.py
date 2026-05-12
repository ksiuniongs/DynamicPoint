#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from waft_single_flow import load_image


def load_mask(path):
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 127).astype(np.uint8)


def compute_union_roi(mask0, mask1, margin_ratio=0.15):
    ys0, xs0 = np.nonzero(mask0)
    ys1, xs1 = np.nonzero(mask1)
    xs = np.concatenate([xs0, xs1]) if len(xs0) + len(xs1) > 0 else np.array([], dtype=np.int64)
    ys = np.concatenate([ys0, ys1]) if len(ys0) + len(ys1) > 0 else np.array([], dtype=np.int64)
    h, w = mask0.shape
    if len(xs) == 0:
        return 0, 0, w, h
    x0 = xs.min()
    x1 = xs.max() + 1
    y0 = ys.min()
    y1 = ys.max() + 1
    bw = x1 - x0
    bh = y1 - y0
    mx = max(8, int(round(bw * margin_ratio)))
    my = max(8, int(round(bh * margin_ratio)))
    x0 = max(0, x0 - mx)
    y0 = max(0, y0 - my)
    x1 = min(w, x1 + mx)
    y1 = min(h, y1 + my)
    return int(x0), int(y0), int(x1), int(y1)


def flow_to_rgb(flow):
    fx = flow[..., 0]
    fy = flow[..., 1]
    mag = np.sqrt(fx ** 2 + fy ** 2)
    ang = np.arctan2(fy, fx)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    hsv[..., 0] = (ang + np.pi) / (2 * np.pi)
    hsv[..., 1] = 1.0
    hsv[..., 2] = np.clip(mag / (np.percentile(mag, 95) + 1e-6), 0, 1)
    rgb = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return rgb


def build_waft_wrapper(waft_root, waft_cfg, waft_ckpt, scale):
    import os
    import sys

    waft_root = Path(waft_root)
    os.chdir(waft_root)
    if str(waft_root) not in sys.path:
        sys.path.insert(0, str(waft_root))

    from config.parser import json_to_args
    from inference_tools import InferenceWrapper
    from model import fetch_model
    from utils.utils import load_ckpt

    cfg = json_to_args(str(waft_cfg))
    cfg.cfg = str(waft_cfg)
    cfg.ckpt = str(waft_ckpt)
    cfg.scale = scale

    model = fetch_model(cfg)
    load_ckpt(model, cfg.ckpt)
    model = model.cuda().eval()
    wrapper = InferenceWrapper(
        model,
        scale=cfg.scale,
        train_size=cfg.image_size,
        pad_to_train_size=False,
        tiling=False,
    )
    return wrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--waft-root", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--scale", type=float, default=0.0)
    args = parser.parse_args()

    import torch

    image_paths = sorted([p for p in Path(args.images).iterdir() if p.is_file()])
    mask_paths = sorted([p for p in Path(args.masks).iterdir() if p.is_file()])
    assert len(image_paths) == len(mask_paths), "image/mask count mismatch"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    wrapper = build_waft_wrapper(args.waft_root, args.cfg, args.ckpt, args.scale)

    preview_frames = []
    roi_meta = []

    for i in range(len(image_paths) - 1):
        image0_np, image0_t = load_image(image_paths[i])
        image1_np, image1_t = load_image(image_paths[i + 1])
        mask0 = load_mask(mask_paths[i])
        mask1 = load_mask(mask_paths[i + 1])
        x0, y0, x1, y1 = compute_union_roi(mask0, mask1)
        roi_meta.append({"pair": [i, i + 1], "roi": [x0, y0, x1, y1]})

        crop0_np = image0_np[y0:y1, x0:x1]
        crop1_np = image1_np[y0:y1, x0:x1]
        crop0_t = image0_t[:, :, y0:y1, x0:x1].cuda()
        crop1_t = image1_t[:, :, y0:y1, x0:x1].cuda()

        with torch.no_grad():
            output = wrapper.calc_flow(crop0_t, crop1_t)
        flow = output["flow"][-1][0].permute(1, 2, 0).detach().cpu().numpy()
        flow_bgr = flow_to_rgb(flow)
        overlay = cv2.addWeighted(cv2.cvtColor(crop1_np, cv2.COLOR_RGB2BGR), 0.45, flow_bgr, 0.55, 0)

        cv2.putText(
            overlay,
            f"{i:04d}->{i+1:04d}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        np.save(outdir / f"flow_pair_{i:04d}_{i+1:04d}.npy", flow)
        cv2.imwrite(str(outdir / f"flow_pair_{i:04d}_{i+1:04d}.png"), flow_bgr)
        cv2.imwrite(str(outdir / f"flow_pair_{i:04d}_{i+1:04d}_overlay.png"), overlay)
        preview_frames.append(overlay)

    if preview_frames:
        max_h = max(frame.shape[0] for frame in preview_frames)
        max_w = max(frame.shape[1] for frame in preview_frames)
        video_path = outdir / "waft_preview.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            6.0,
            (max_w, max_h),
        )
        for frame in preview_frames:
            canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            canvas[: frame.shape[0], : frame.shape[1]] = frame
            writer.write(canvas)
        writer.release()

    with open(outdir / "roi_pairs.json", "w", encoding="utf-8") as f:
        json.dump(roi_meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
