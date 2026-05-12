#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    return image, tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--waft-root", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--image1", required=True)
    parser.add_argument("--image2", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--scale", type=float, default=0.0)
    args = parser.parse_args()

    waft_root = Path(args.waft_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    os.chdir(waft_root)
    sys.path.insert(0, str(waft_root))
    from config.parser import json_to_args
    from inference_tools import InferenceWrapper
    from model import fetch_model
    from utils.flow_viz import flow_to_image
    from utils.utils import load_ckpt

    cfg = json_to_args(args.cfg)
    cfg.cfg = args.cfg
    cfg.ckpt = args.ckpt
    cfg.scale = args.scale

    _, image1 = load_image(Path(args.image1))
    image2_np, image2 = load_image(Path(args.image2))

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

    with torch.no_grad():
        output = wrapper.calc_flow(image1.cuda(), image2.cuda())

    flow = output["flow"][-1][0].permute(1, 2, 0).detach().cpu().numpy()
    flow_bgr = flow_to_image(flow, convert_to_bgr=True)

    overlay = cv2.addWeighted(
        cv2.cvtColor(image2_np, cv2.COLOR_RGB2BGR),
        0.45,
        flow_bgr,
        0.55,
        0,
    )

    np.save(outdir / "flow.npy", flow)
    cv2.imwrite(str(outdir / "flow.png"), flow_bgr)
    cv2.imwrite(str(outdir / "flow_overlay.png"), overlay)


if __name__ == "__main__":
    main()
