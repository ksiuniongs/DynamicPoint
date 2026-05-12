#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement


def read_ply_xyz_rgb(path: Path):
    ply = PlyData.read(str(path))
    verts = ply["vertex"]
    xyz = np.column_stack([verts["x"], verts["y"], verts["z"]]).astype(np.float32)
    rgb = np.column_stack([verts["red"], verts["green"], verts["blue"]]).astype(np.uint8)
    return xyz, rgb


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
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
    arr = np.empty(xyz.shape[0], dtype=dtype)
    arr["x"] = xyz[:, 0]
    arr["y"] = xyz[:, 1]
    arr["z"] = xyz[:, 2]
    arr["nx"] = 0
    arr["ny"] = 0
    arr["nz"] = 0
    arr["red"] = rgb[:, 0]
    arr["green"] = rgb[:, 1]
    arr["blue"] = rgb[:, 2]
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fuse and clean UniSH scene-only point clouds.")
    ap.add_argument("--scene_ply_dir", required=True)
    ap.add_argument("--output_ply", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--voxel_size", type=float, default=0.03)
    ap.add_argument("--nb_neighbors", type=int, default=20)
    ap.add_argument("--std_ratio", type=float, default=2.0)
    ap.add_argument("--max_points", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ply_paths = sorted(Path(args.scene_ply_dir).glob("*.ply"))
    if not ply_paths:
        raise FileNotFoundError(f"No PLY files found in {args.scene_ply_dir}")

    xyz_all = []
    rgb_all = []
    per_file = []
    for path in ply_paths:
        xyz, rgb = read_ply_xyz_rgb(path)
        xyz_all.append(xyz)
        rgb_all.append(rgb)
        per_file.append({"file": path.name, "points": int(len(xyz))})

    xyz = np.concatenate(xyz_all, axis=0)
    rgb = np.concatenate(rgb_all, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)

    if args.voxel_size > 0:
        pcd = pcd.voxel_down_sample(args.voxel_size)
    if args.nb_neighbors > 0:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=args.nb_neighbors,
            std_ratio=args.std_ratio,
        )

    xyz_out = np.asarray(pcd.points).astype(np.float32)
    rgb_out = np.clip(np.asarray(pcd.colors) * 255.0, 0, 255).astype(np.uint8)

    if args.max_points > 0 and len(xyz_out) > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(xyz_out), size=args.max_points, replace=False)
        xyz_out = xyz_out[idx]
        rgb_out = rgb_out[idx]

    output_ply = Path(args.output_ply)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    write_ply(output_ply, xyz_out, rgb_out)

    summary = {
        "scene_ply_dir": args.scene_ply_dir,
        "num_input_files": len(ply_paths),
        "raw_points": int(len(xyz)),
        "final_points": int(len(xyz_out)),
        "voxel_size": args.voxel_size,
        "nb_neighbors": args.nb_neighbors,
        "std_ratio": args.std_ratio,
        "max_points": args.max_points,
        "per_file": per_file,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
