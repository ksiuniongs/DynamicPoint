#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


def rotmat_to_qvec(rmat: np.ndarray) -> np.ndarray:
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rmat.flat
    k = np.array(
        [
            [rxx - ryy - rzz, 0.0, 0.0, 0.0],
            [ryx + rxy, ryy - rxx - rzz, 0.0, 0.0],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0.0],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ],
        dtype=np.float64,
    ) / 3.0
    eigvals, eigvecs = np.linalg.eigh(k)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1.0
    return qvec


def read_ply_xyz_rgb(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    verts = ply["vertex"]
    xyz = np.column_stack([verts["x"], verts["y"], verts["z"]]).astype(np.float32)
    rgb = np.column_stack([verts["red"], verts["green"], verts["blue"]]).astype(np.uint8)
    return xyz, rgb


def write_point_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    normals = np.zeros_like(xyz, dtype=np.float32)
    arr = np.empty(xyz.shape[0], dtype=dtype)
    arr["x"] = xyz[:, 0]
    arr["y"] = xyz[:, 1]
    arr["z"] = xyz[:, 2]
    arr["nx"] = normals[:, 0]
    arr["ny"] = normals[:, 1]
    arr["nz"] = normals[:, 2]
    arr["red"] = rgb[:, 0]
    arr["green"] = rgb[:, 1]
    arr["blue"] = rgb[:, 2]
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


def sample_points(
    ply_paths: List[Path],
    sample_per_frame: int,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, int]]]:
    rng = np.random.default_rng(seed)
    xyz_chunks = []
    rgb_chunks = []
    stats = []

    for ply_path in ply_paths:
        xyz, rgb = read_ply_xyz_rgb(ply_path)
        keep = len(xyz)
        if sample_per_frame > 0 and keep > sample_per_frame:
            idx = rng.choice(keep, size=sample_per_frame, replace=False)
            xyz = xyz[idx]
            rgb = rgb[idx]
        xyz_chunks.append(xyz)
        rgb_chunks.append(rgb)
        stats.append(
            {
                "file": ply_path.name,
                "original_points": int(keep),
                "kept_points": int(len(xyz)),
            }
        )

    merged_xyz = np.concatenate(xyz_chunks, axis=0)
    merged_rgb = np.concatenate(rgb_chunks, axis=0)
    if max_points > 0 and len(merged_xyz) > max_points:
        idx = rng.choice(len(merged_xyz), size=max_points, replace=False)
        merged_xyz = merged_xyz[idx]
        merged_rgb = merged_rgb[idx]

    return merged_xyz, merged_rgb, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a COLMAP-style GS dataset from UniSH exports.")
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--camera_npz", required=True)
    ap.add_argument("--scene_ply_dir", default="")
    ap.add_argument("--input_ply", default="")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--sample_per_frame", type=int, default=3000)
    ap.add_argument("--max_points", type=int, default=80000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    camera_npz = Path(args.camera_npz)
    scene_ply_dir = Path(args.scene_ply_dir) if args.scene_ply_dir else None
    input_ply = Path(args.input_ply) if args.input_ply else None
    output_dir = Path(args.output_dir)
    images_out = output_dir / "images"
    sparse_out = output_dir / "sparse" / "0"
    images_out.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("*"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    camera_data = np.load(camera_npz)
    extrinsics = camera_data["extrinsics"].astype(np.float64)
    intrinsics = camera_data["intrinsics"].astype(np.float64)
    if len(frame_paths) != len(extrinsics):
        raise ValueError(
            f"Frame count {len(frame_paths)} does not match camera count {len(extrinsics)}"
        )

    width, height = Image.open(frame_paths[0]).size

    for src in frame_paths:
        shutil.copy2(src, images_out / src.name)

    with open(sparse_out / "cameras.txt", "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(frame_paths)}\n")
        for idx, k in enumerate(intrinsics, start=1):
            fx = float(k[0, 0])
            fy = float(k[1, 1])
            cx = float(k[0, 2])
            cy = float(k[1, 2])
            f.write(f"{idx} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

    with open(sparse_out / "images.txt", "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(frame_paths)}\n")
        for idx, (frame_path, w2c) in enumerate(zip(frame_paths, extrinsics), start=1):
            r = w2c[:3, :3]
            t = w2c[:3, 3]
            q = rotmat_to_qvec(r)
            f.write(
                f"{idx} {q[0]} {q[1]} {q[2]} {q[3]} "
                f"{t[0]} {t[1]} {t[2]} {idx} {frame_path.name}\n"
            )
            f.write("0.0 0.0 -1\n")

    if input_ply is not None:
        if not input_ply.exists():
            raise FileNotFoundError(f"Input ply not found: {input_ply}")
        xyz, rgb = read_ply_xyz_rgb(input_ply)
        point_stats = [
            {
                "file": input_ply.name,
                "original_points": int(len(xyz)),
                "kept_points": int(len(xyz)),
            }
        ]
        if args.max_points > 0 and len(xyz) > args.max_points:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(xyz), size=args.max_points, replace=False)
            xyz = xyz[idx]
            rgb = rgb[idx]
            point_stats[0]["kept_points"] = int(len(xyz))
    else:
        if scene_ply_dir is None:
            raise ValueError("Provide either --scene_ply_dir or --input_ply")
        ply_paths = sorted(scene_ply_dir.glob("*.ply"))
        if not ply_paths:
            raise FileNotFoundError(f"No scene PLY files found in {scene_ply_dir}")
        xyz, rgb, point_stats = sample_points(
            ply_paths, args.sample_per_frame, args.max_points, args.seed
        )
    write_point_ply(sparse_out / "points3D.ply", xyz, rgb)

    summary = {
        "frames_dir": str(frames_dir),
        "camera_npz": str(camera_npz),
        "scene_ply_dir": str(scene_ply_dir) if scene_ply_dir is not None else "",
        "input_ply": str(input_ply) if input_ply is not None else "",
        "num_frames": len(frame_paths),
        "image_size": [width, height],
        "sample_per_frame": args.sample_per_frame,
        "max_points": args.max_points,
        "merged_points": int(len(xyz)),
        "point_sampling": point_stats,
    }
    with open(output_dir / "unish_dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
