#!/usr/bin/env python3
import argparse
import math
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


SCGS_VIEW_PAIRS = [
    (-60.0, 0.0),
    (-30.0, 0.0),
    (0.0, 0.0),
    (30.0, 0.0),
    (60.0, 0.0),
    (0.0, -10.0),
    (0.0, 10.0),
]


def parse_size(s):
    if "x" not in s:
        raise ValueError("size must be like 1080x1080")
    w, h = s.lower().split("x", 1)
    return int(w), int(h)


def angle_tag(value):
    value = int(round(value))
    if value < 0:
        return f"m{abs(value):03d}"
    return f"{value:03d}"


def build_rotation(yaw_deg, pitch_deg):
    yaw = math.radians(yaw_deg)
    # SCGS uses up/down virtual views; keep positive pitch = look up.
    pitch = math.radians(-pitch_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    ry = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float32,
    )
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]],
        dtype=np.float32,
    )
    return ry @ rx


def sample_bilinear(img, u, v):
    h, w, _ = img.shape
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

    c0 = c00 * (1.0 - du)[..., None] + c10 * du[..., None]
    c1 = c01 * (1.0 - du)[..., None] + c11 * du[..., None]
    return c0 * (1.0 - dv)[..., None] + c1 * dv[..., None]


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

    # Image y grows downward, but camera-space y should point upward.
    dirs = np.stack([xv, -yv, np.ones_like(xv)], axis=-1)
    dirs /= np.maximum(np.linalg.norm(dirs, axis=-1, keepdims=True), 1e-8)

    rot = build_rotation(yaw_deg, pitch_deg)
    dirs = dirs @ rot.T

    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    lon = np.arctan2(x, z)
    lat = np.arcsin(np.clip(y, -1.0, 1.0))

    u = (lon / (2 * math.pi) + 0.5) * w
    v = (0.5 - lat / math.pi) * h

    out = sample_bilinear(img, u, v)
    return np.clip(out, 0, 255).astype(np.uint8)


def extract_frames(video_path, frames_dir, start_time, duration):
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        start_time,
        "-i",
        str(video_path),
        "-t",
        duration,
        "-vsync",
        "0",
        "-start_number",
        "0",
        str(frames_dir / "frame_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)


def render_views(frames_dir, views_dir, size, fov):
    out_w, out_h = parse_size(size)
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise SystemExit("no extracted frames found")

    views_dir.mkdir(parents=True, exist_ok=True)

    for frame_path in frame_paths:
        base = frame_path.stem
        arr = np.array(Image.open(frame_path).convert("RGB"), dtype=np.uint8)

        for yaw, pitch in SCGS_VIEW_PAIRS:
            view_dir = views_dir / f"yaw_{angle_tag(yaw)}_pitch_{angle_tag(pitch)}"
            view_dir.mkdir(parents=True, exist_ok=True)
            out = project_equirect_to_pinhole(arr, out_w, out_h, fov, yaw, pitch)
            out_name = f"{base}_yaw_{angle_tag(yaw)}_pitch_{angle_tag(pitch)}.jpg"
            Image.fromarray(out).save(view_dir / out_name, quality=95)


def main():
    ap = argparse.ArgumentParser(description="Extract SCGS-style virtual views from a 360 video.")
    ap.add_argument("--video", required=True, help="Input 360 video")
    ap.add_argument("--output_root", required=True, help="Output root directory")
    ap.add_argument("--start", default="00:08:00", help="Start time, e.g. 00:08:00")
    ap.add_argument("--duration", default="10", help="Duration in seconds")
    ap.add_argument("--size", default="1080x1080", help="Pinhole view size WxH")
    ap.add_argument("--fov", type=float, default=90.0, help="Horizontal FOV in degrees")
    ap.add_argument("--views_dir_name", default="views_scgs", help="Subdirectory name for rendered views")
    ap.add_argument("--skip_extract", action="store_true", help="Reuse existing frames")
    args = ap.parse_args()

    output_root = Path(args.output_root)
    frames_dir = output_root / "frames"
    views_dir = output_root / args.views_dir_name

    if not args.skip_extract:
        extract_frames(Path(args.video), frames_dir, args.start, args.duration)
    render_views(frames_dir, views_dir, args.size, args.fov)


if __name__ == "__main__":
    main()
