#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qz * qx + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qz * qx - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def umeyama(src: np.ndarray, dst: np.ndarray):
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = (dst_centered.T @ src_centered) / src.shape[0]
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1.0
    rot = u @ s @ vt
    var_src = np.mean(np.sum(src_centered ** 2, axis=1))
    scale = np.trace(np.diag(d) @ s) / max(var_src, 1e-12)
    trans = dst_mean - scale * (rot @ src_mean)
    return scale, rot, trans


def read_colmap_images(images_txt: Path):
    cams = {}
    with open(images_txt, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        image_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])), dtype=np.float64)
        tvec = np.array(list(map(float, parts[5:8])), dtype=np.float64)
        name = parts[9]
        rot = qvec2rotmat(qvec)
        center = -rot.T @ tvec
        cams[name] = {"id": image_id, "center": center, "rot": rot, "t": tvec}
        idx += 1
    return cams


def read_colmap_points(points_txt: Path):
    pts = []
    with open(points_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pts.append(
                (
                    np.array(list(map(float, parts[1:4])), dtype=np.float64),
                    np.array(list(map(int, parts[4:7])), dtype=np.uint8),
                )
            )
    xyz = np.stack([p[0] for p in pts], axis=0)
    rgb = np.stack([p[1] for p in pts], axis=0)
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
    ap = argparse.ArgumentParser(description="Align a COLMAP sparse cloud into UniSH world coordinates.")
    ap.add_argument("--camera_npz", required=True)
    ap.add_argument("--colmap_images_txt", required=True)
    ap.add_argument("--colmap_points_txt", required=True)
    ap.add_argument("--output_ply", required=True)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    unish = np.load(args.camera_npz)["extrinsics"].astype(np.float64)
    unish_centers = {}
    for idx, w2c in enumerate(unish):
        r = w2c[:3, :3]
        t = w2c[:3, 3]
        unish_centers[f"frame_{idx:06d}.jpg"] = -r.T @ t

    colmap_cams = read_colmap_images(Path(args.colmap_images_txt))
    common = sorted(set(unish_centers) & set(colmap_cams))
    if not common:
        raise ValueError("No matching image names between UniSH and COLMAP")

    src = np.stack([colmap_cams[name]["center"] for name in common], axis=0)
    dst = np.stack([unish_centers[name] for name in common], axis=0)
    scale, rot, trans = umeyama(src, dst)

    xyz, rgb = read_colmap_points(Path(args.colmap_points_txt))
    xyz_aligned = (scale * (rot @ xyz.T)).T + trans[None, :]

    output_ply = Path(args.output_ply)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    write_ply(output_ply, xyz_aligned.astype(np.float32), rgb)

    aligned_cam = (scale * (rot @ src.T)).T + trans[None, :]
    err = np.linalg.norm(aligned_cam - dst, axis=1)
    summary = {
        "num_common_cameras": len(common),
        "scale": float(scale),
        "rotation": rot.tolist(),
        "translation": trans.tolist(),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean_error": float(err.mean()),
        "max_error": float(err.max()),
        "output_ply": str(output_ply),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
