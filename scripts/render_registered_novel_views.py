#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData

C0 = 0.28209479177387814


def load_ply_xyz_rgb(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    rgb = None
    alpha = None
    size = None
    names = set(v.dtype.names or [])
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.float64) / 255.0
    elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        sh = np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]).astype(np.float64)
        rgb = np.clip(sh * C0 + 0.5, 0.0, 1.0)
    if "opacity" in names:
        opacity = np.asarray(v["opacity"], dtype=np.float64)
        alpha = 1.0 / (1.0 + np.exp(-opacity))
    if {"scale_0", "scale_1", "scale_2"}.issubset(names):
        scales = np.column_stack([v["scale_0"], v["scale_1"], v["scale_2"]]).astype(np.float64)
        size = np.exp(scales).mean(axis=1)
    return xyz, rgb, alpha, size


def subsample(
    points: np.ndarray,
    colors: np.ndarray | None,
    alpha: np.ndarray | None,
    size: np.ndarray | None,
    limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if len(points) <= limit:
        return points, colors, alpha, size
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=limit, replace=False)
    idx.sort()
    colors_out = None if colors is None else colors[idx]
    alpha_out = None if alpha is None else alpha[idx]
    size_out = None if size is None else size[idx]
    return points[idx], colors_out, alpha_out, size_out


def set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_frame(
    bg_points: np.ndarray,
    bg_colors: np.ndarray | None,
    bg_alpha: np.ndarray | None,
    bg_size: np.ndarray | None,
    dyn_points: np.ndarray,
    out_path: Path,
    elev: float,
    azim: float,
    title: str,
) -> None:
    fig = plt.figure(figsize=(8, 8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    if bg_colors is None:
        bg_colors = np.tile(np.array([[0.62, 0.68, 0.74]]), (len(bg_points), 1))
    bg_scatter_size = 0.8
    if bg_size is not None:
        norm = np.clip(bg_size / np.percentile(bg_size, 90), 0.05, 2.5)
        bg_scatter_size = 0.6 + 4.0 * norm
    bg_color_input = bg_colors
    bg_scatter_alpha = 0.28
    if bg_alpha is not None:
        bg_color_input = np.concatenate([bg_colors, np.clip(bg_alpha[:, None], 0.06, 0.85)], axis=1)
        bg_scatter_alpha = None
    ax.scatter(
        bg_points[:, 0],
        bg_points[:, 1],
        bg_points[:, 2],
        c=bg_color_input,
        s=bg_scatter_size,
        alpha=bg_scatter_alpha,
        edgecolors="none",
        linewidths=0,
        depthshade=False,
    )
    ax.scatter(
        dyn_points[:, 0],
        dyn_points[:, 1],
        dyn_points[:, 2],
        c="#d62728",
        s=1.8,
        alpha=0.9,
        linewidths=0,
        depthshade=False,
    )

    combo = np.vstack([bg_points, dyn_points])
    set_equal_axes(ax, combo)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(title, fontsize=10, pad=10)
    ax.grid(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def make_contact_sheet(frame_paths: list[Path], out_path: Path, cols: int = 3) -> None:
    from PIL import Image

    imgs = [Image.open(p).convert("RGB") for p in frame_paths]
    if not imgs:
        return
    w, h = imgs[0].size
    rows = math.ceil(len(imgs) / cols)
    canvas = Image.new("RGB", (cols * w, rows * h), color="white")
    for i, img in enumerate(imgs):
        canvas.paste(img, ((i % cols) * w, (i // cols) * h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render quick novel-view previews for registered static/dynamic point clouds.")
    parser.add_argument("--static_ply", required=True)
    parser.add_argument("--dynamic_ply", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bg_limit", type=int, default=6000)
    parser.add_argument("--dyn_limit", type=int, default=4000)
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--elev", type=float, default=18.0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    bg_points, bg_colors, bg_alpha, bg_size = load_ply_xyz_rgb(Path(args.static_ply))
    dyn_points, _, _, _ = load_ply_xyz_rgb(Path(args.dynamic_ply))
    bg_points, bg_colors, bg_alpha, bg_size = subsample(bg_points, bg_colors, bg_alpha, bg_size, args.bg_limit, seed=1)
    dyn_points, _, _, _ = subsample(dyn_points, None, None, None, args.dyn_limit, seed=2)

    frame_paths: list[Path] = []
    for i in range(args.num_views):
        azim = (360.0 / args.num_views) * i + 25.0
        out_path = frames_dir / f"view_{i:03d}.png"
        render_frame(
            bg_points,
            bg_colors,
            bg_alpha,
            bg_size,
            dyn_points,
            out_path,
            elev=args.elev,
            azim=azim,
            title=f"Novel View {i} | azim={azim:.1f}",
        )
        frame_paths.append(out_path)

    make_contact_sheet(frame_paths, out_dir / "novel_view_contact_sheet.png", cols=3)

    mp4_path = out_dir / "novel_view_orbit.mp4"
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "4",
        "-i",
        str(frames_dir / "view_%03d.png"),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(mp4_path),
    ]
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(mp4_path)
    print(out_dir / "novel_view_contact_sheet.png")


if __name__ == "__main__":
    main()
