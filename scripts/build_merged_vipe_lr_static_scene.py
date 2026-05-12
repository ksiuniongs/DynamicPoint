#!/usr/bin/env python3
import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge left/right VIPE monocular reconstructions into one static COLMAP-style scene."
    )
    parser.add_argument("--right_pose_npz", required=True)
    parser.add_argument("--left_pose_npz", required=True)
    parser.add_argument("--right_intrinsics_npz", required=True)
    parser.add_argument("--left_intrinsics_npz", required=True)
    parser.add_argument("--right_points3d_txt", required=True)
    parser.add_argument("--left_points3d_txt", required=True)
    parser.add_argument("--lama_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def umeyama_alignment(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    cov = (dst_centered.T @ src_centered) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_src = np.mean(np.sum(src_centered ** 2, axis=1))
    scale = np.trace(np.diag(D) @ S) / var_src
    t = dst_mean - scale * (R @ src_mean)
    return scale, R, t


def c2w_to_colmap_pose(c2w):
    w2c = np.linalg.inv(c2w)
    quat_xyzw = Rotation.from_matrix(w2c[:3, :3]).as_quat()
    qwxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    trans = w2c[:3, 3]
    return qwxyz, trans


def parse_points3d_txt(path):
    points = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split()
            point = {
                "xyz": np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64),
                "rgb": (int(parts[4]), int(parts[5]), int(parts[6])),
                "error": float(parts[7]),
                "track": [int(x) for x in parts[8:]],
            }
            points.append(point)
    return points


def shift_track_ids(track, image_id_offset):
    shifted = []
    for idx in range(0, len(track), 2):
        image_id = track[idx] + image_id_offset
        point2d_idx = track[idx + 1]
        shifted.extend([image_id, point2d_idx])
    return shifted


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    sparse_dir = output_dir / "sparse" / "0"
    images_dir = output_dir / "images"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    right_pose = np.load(args.right_pose_npz)
    left_pose = np.load(args.left_pose_npz)
    right_poses = right_pose["data"].astype(np.float64)
    left_poses = left_pose["data"].astype(np.float64)
    right_inds = right_pose["inds"].astype(np.int64)
    left_inds = left_pose["inds"].astype(np.int64)

    if not np.array_equal(right_inds, left_inds):
        raise ValueError("Left/right VIPE frame indices do not match.")

    right_centers = right_poses[:, :3, 3]
    left_centers = left_poses[:, :3, 3]

    scale, global_rot, trans = umeyama_alignment(left_centers, right_centers)

    left_poses_aligned = left_poses.copy()
    left_poses_aligned[:, :3, :3] = global_rot[None, :, :] @ left_poses[:, :3, :3]
    left_poses_aligned[:, :3, 3] = (
        scale * (global_rot @ left_poses[:, :3, 3].T).T + trans[None, :]
    )

    aligned_centers = left_poses_aligned[:, :3, 3]
    center_rmse = float(np.sqrt(np.mean(np.sum((aligned_centers - right_centers) ** 2, axis=1))))
    orientation_delta = []
    for l_pose, r_pose in zip(left_poses_aligned, right_poses):
        rel = Rotation.from_matrix(l_pose[:3, :3]).inv() * Rotation.from_matrix(r_pose[:3, :3])
        orientation_delta.append(rel.as_rotvec())
    orientation_delta = np.stack(orientation_delta, axis=0)
    mean_orientation_delta_deg = float(np.linalg.norm(orientation_delta.mean(axis=0)) * 180.0 / math.pi)

    right_intr = np.load(args.right_intrinsics_npz)["data"][0].astype(np.float64)
    left_intr = np.load(args.left_intrinsics_npz)["data"][0].astype(np.float64)

    sample_image = Image.open(sorted(Path(args.lama_dir).glob("right_*_mask001.png"))[0])
    width, height = sample_image.size

    with open(sparse_dir / "cameras.txt", "w", encoding="utf-8") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(
            f"1 PINHOLE {width} {height} {right_intr[0]:.6f} {right_intr[1]:.6f} {right_intr[2]:.6f} {right_intr[3]:.6f}\n"
        )
        f.write(
            f"2 PINHOLE {width} {height} {left_intr[0]:.6f} {left_intr[1]:.6f} {left_intr[2]:.6f} {left_intr[3]:.6f}\n"
        )

    with open(sparse_dir / "images.txt", "w", encoding="utf-8") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        image_id = 1
        for idx, pose in zip(right_inds, right_poses):
            quat, tvec = c2w_to_colmap_pose(pose)
            name = f"right_frame_{idx:06d}.jpg"
            f.write(
                f"{image_id} {quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} {quat[3]:.9f} "
                f"{tvec[0]:.9f} {tvec[1]:.9f} {tvec[2]:.9f} 1 {name}\n\n"
            )
            image_id += 1
        for idx, pose in zip(left_inds, left_poses_aligned):
            quat, tvec = c2w_to_colmap_pose(pose)
            name = f"left_frame_{idx:06d}.jpg"
            f.write(
                f"{image_id} {quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} {quat[3]:.9f} "
                f"{tvec[0]:.9f} {tvec[1]:.9f} {tvec[2]:.9f} 2 {name}\n\n"
            )
            image_id += 1

    right_points = parse_points3d_txt(args.right_points3d_txt)
    left_points = parse_points3d_txt(args.left_points3d_txt)
    with open(sparse_dir / "points3D.txt", "w", encoding="utf-8") as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        point_id = 1
        for point in right_points:
            track = point["track"] if point["track"] else [1, 0]
            f.write(
                f"{point_id} {point['xyz'][0]:.6f} {point['xyz'][1]:.6f} {point['xyz'][2]:.6f} "
                f"{point['rgb'][0]} {point['rgb'][1]} {point['rgb'][2]} {point['error']:.6f} "
                + " ".join(str(x) for x in track)
                + "\n"
            )
            point_id += 1
        for point in left_points:
            xyz = scale * (global_rot @ point["xyz"]) + trans
            track = shift_track_ids(point["track"], 20) if point["track"] else [21, 0]
            f.write(
                f"{point_id} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} "
                f"{point['rgb'][0]} {point['rgb'][1]} {point['rgb'][2]} {point['error']:.6f} "
                + " ".join(str(x) for x in track)
                + "\n"
            )
            point_id += 1

    right_files = sorted(Path(args.lama_dir).glob("right_*_mask001.png"))
    left_files = sorted(Path(args.lama_dir).glob("left_*_mask001.png"))
    for src in right_files:
        frame_id = src.stem.split("_")[1]
        dst = images_dir / f"right_frame_{frame_id}.jpg"
        with Image.open(src) as img:
            img.convert("RGB").save(dst, quality=95)
    for src in left_files:
        frame_id = src.stem.split("_")[1]
        dst = images_dir / f"left_frame_{frame_id}.jpg"
        with Image.open(src) as img:
            img.convert("RGB").save(dst, quality=95)

    with open(output_dir / "vipe_lr_alignment.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "scale_left_to_right": scale,
                "rotation_left_to_right": global_rot.tolist(),
                "translation_left_to_right": trans.tolist(),
                "center_rmse": center_rmse,
                "mean_orientation_delta_deg": mean_orientation_delta_deg,
                "num_right_images": len(right_files),
                "num_left_images": len(left_files),
            },
            f,
            indent=2,
        )

    print(f"Built merged scene at {output_dir}")
    print(f"Center RMSE after alignment: {center_rmse:.6f}")
    print(f"Mean left/right orientation delta magnitude: {mean_orientation_delta_deg:.3f} deg")


if __name__ == "__main__":
    main()
