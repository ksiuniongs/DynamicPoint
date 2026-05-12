#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    return image, tensor


def flow_to_rgb(flow):
    fx = flow[..., 0]
    fy = flow[..., 1]
    mag = np.sqrt(fx ** 2 + fy ** 2)
    ang = np.arctan2(fy, fx)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    hsv[..., 0] = (ang + np.pi) / (2 * np.pi)
    hsv[..., 1] = 1.0
    hsv[..., 2] = np.clip(mag / (np.percentile(mag, 95) + 1e-6), 0, 1)
    return cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def build_waft_wrapper(waft_root, waft_cfg, waft_ckpt, scale):
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


def label_frame(frame, text):
    frame = frame.copy()
    cv2.putText(
        frame,
        text,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--waft-root", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--actual-inputs", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--scale", type=float, default=0.0)
    parser.add_argument("--fps", type=float, default=6.0)
    args = parser.parse_args()

    inputs_dir = Path(args.actual_inputs)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in inputs_dir.glob("*.png") if not p.name.endswith("_mask.png") and not p.name.endswith("_overlay.png")])
    wrapper = build_waft_wrapper(args.waft_root, args.cfg, args.ckpt, args.scale)

    video_frames = []
    metadata = []

    if image_paths:
        first_np, _ = load_image(image_paths[0])
        first_bgr = cv2.cvtColor(first_np, cv2.COLOR_RGB2BGR)
        first_bgr = label_frame(first_bgr, "0000 (input)")
        video_frames.append(first_bgr)

    for i in range(len(image_paths) - 1):
        image0_np, image0_t = load_image(image_paths[i])
        image1_np, image1_t = load_image(image_paths[i + 1])

        with torch.no_grad():
            output = wrapper.calc_flow(image0_t.cuda(), image1_t.cuda())

        flow = output["flow"][-1][0].permute(1, 2, 0).detach().cpu().numpy()
        flow_bgr = flow_to_rgb(flow)
        overlay = cv2.addWeighted(cv2.cvtColor(image1_np, cv2.COLOR_RGB2BGR), 0.45, flow_bgr, 0.55, 0)
        flow_bgr = label_frame(flow_bgr, f"{i:04d}->{i+1:04d} flow")
        overlay = label_frame(overlay, f"{i:04d}->{i+1:04d} overlay")

        np.save(outdir / f"flow_pair_{i:04d}_{i+1:04d}.npy", flow)
        cv2.imwrite(str(outdir / f"flow_pair_{i:04d}_{i+1:04d}.png"), flow_bgr)
        cv2.imwrite(str(outdir / f"flow_pair_{i:04d}_{i+1:04d}_overlay.png"), overlay)

        video_frames.append(overlay)
        metadata.append({"pair": [i, i + 1]})

    if video_frames:
        h, w = video_frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(outdir / "waft_actual_preview.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (w, h),
        )
        for frame in video_frames:
            writer.write(frame)
        writer.release()

    with open(outdir / "pairs.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
