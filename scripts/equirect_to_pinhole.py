#!/usr/bin/env python3
import argparse
import math
import os
from glob import glob

import numpy as np
from PIL import Image


def parse_size(s):
    if "x" not in s:
        raise ValueError("size must be like 800x800")
    w, h = s.lower().split("x", 1)
    return int(w), int(h)


def parse_list(s):
    return [float(x) for x in s.split(",") if x.strip() != ""]


def build_rotation(yaw_deg, pitch_deg):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    return ry @ rx


def sample_bilinear(img, u, v):
    h, w, c = img.shape
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = (u0 + 1) % w
    v1 = np.clip(v0 + 1, 0, h - 1)
    u0 = u0 % w
    v0 = np.clip(v0, 0, h - 1)

    du = (u - u0).astype(np.float32)
    dv = (v - v0).astype(np.float32)

    c00 = img[v0, u0]
    c10 = img[v0, u1]
    c01 = img[v1, u0]
    c11 = img[v1, u1]

    c0 = c00 * (1 - du)[..., None] + c10 * du[..., None]
    c1 = c01 * (1 - du)[..., None] + c11 * du[..., None]
    return c0 * (1 - dv)[..., None] + c1 * dv[..., None]


def project_equirect_to_pinhole(img, out_w, out_h, fov_deg, yaw_deg, pitch_deg):
    h, w, _ = img.shape
    fov = math.radians(fov_deg)
    fx = (out_w / 2.0) / math.tan(fov / 2.0)
    fy = fx
    cx = (out_w - 1) / 2.0
    cy = (out_h - 1) / 2.0

    xs = (np.arange(out_w, dtype=np.float32) - cx) / fx
    ys = (np.arange(out_h, dtype=np.float32) - cy) / fy
    xv, yv = np.meshgrid(xs, ys)
    zv = np.ones_like(xv)

    dirs = np.stack([xv, yv, zv], axis=-1)
    norm = np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs = dirs / np.maximum(norm, 1e-8)

    rot = build_rotation(yaw_deg, pitch_deg)
    dirs = dirs @ rot.T

    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    lon = np.arctan2(x, z)
    lat = np.arcsin(np.clip(y, -1.0, 1.0))

    u = (lon / (2 * math.pi) + 0.5) * w
    v = (0.5 - lat / math.pi) * h

    out = sample_bilinear(img, u, v)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Convert equirectangular frames to pinhole views.")
    ap.add_argument("--input_dir", required=True, help="Directory with equirectangular frames")
    ap.add_argument("--output_dir", required=True, help="Output directory for pinhole images")
    ap.add_argument("--yaw", default="0,90,180,270", help="Comma-separated yaw degrees")
    ap.add_argument("--pitch", default="0", help="Comma-separated pitch degrees")
    ap.add_argument("--fov", type=float, default=90.0, help="Horizontal FOV in degrees")
    ap.add_argument("--size", default="800x800", help="Output size WxH")
    ap.add_argument("--ext", default="jpg", help="Output extension: jpg or png")
    args = ap.parse_args()

    out_w, out_h = parse_size(args.size)
    yaw_list = parse_list(args.yaw)
    pitch_list = parse_list(args.pitch)

    os.makedirs(args.output_dir, exist_ok=True)
    inputs = sorted(glob(os.path.join(args.input_dir, "*")))
    if not inputs:
        raise SystemExit("no input frames found")

    for i, path in enumerate(inputs):
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        base = f"frame_{i:06d}"
        for yaw in yaw_list:
            for pitch in pitch_list:
                out = project_equirect_to_pinhole(arr, out_w, out_h, args.fov, yaw, pitch)
                name = f"{base}_yaw_{int(yaw):03d}_pitch_{int(pitch):03d}.{args.ext}"
                Image.fromarray(out).save(os.path.join(args.output_dir, name), quality=95)


if __name__ == "__main__":
    main()
