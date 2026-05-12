#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a DreamScene4D model as a per-frame PLY sequence.")
    parser.add_argument("--dynamic_pkl", required=True)
    parser.add_argument("--dynamic_motion_pkl", required=True)
    parser.add_argument("--base_transform_json", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefix", default="frame")
    return parser.parse_args()


def load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def to_numpy(value) -> np.ndarray:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_transform(path: str | None) -> tuple[float, np.ndarray, np.ndarray]:
    if path is None:
        return 1.0, np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    data = json.loads(Path(path).read_text())
    scale = float(data["optimized_scale"])
    rotvec = np.asarray(data["optimized_rotvec"], dtype=np.float64)
    trans = np.asarray(data["optimized_translation"], dtype=np.float64)
    return scale, rotvec, trans


def write_frame_ply(
    out_path: Path,
    xyz: np.ndarray,
    normals: np.ndarray,
    f_dc: np.ndarray,
    opacity: np.ndarray,
    scaling: np.ndarray,
    rotation_wxyz: np.ndarray,
) -> None:
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]
    verts = np.empty(xyz.shape[0], dtype=dtype)
    verts["x"] = xyz[:, 0].astype(np.float32)
    verts["y"] = xyz[:, 1].astype(np.float32)
    verts["z"] = xyz[:, 2].astype(np.float32)
    verts["nx"] = normals[:, 0].astype(np.float32)
    verts["ny"] = normals[:, 1].astype(np.float32)
    verts["nz"] = normals[:, 2].astype(np.float32)
    verts["f_dc_0"] = f_dc[:, 0].astype(np.float32)
    verts["f_dc_1"] = f_dc[:, 1].astype(np.float32)
    verts["f_dc_2"] = f_dc[:, 2].astype(np.float32)
    verts["opacity"] = opacity[:, 0].astype(np.float32)
    verts["scale_0"] = scaling[:, 0].astype(np.float32)
    verts["scale_1"] = scaling[:, 1].astype(np.float32)
    verts["scale_2"] = scaling[:, 2].astype(np.float32)
    verts["rot_0"] = rotation_wxyz[:, 0].astype(np.float32)
    verts["rot_1"] = rotation_wxyz[:, 1].astype(np.float32)
    verts["rot_2"] = rotation_wxyz[:, 2].astype(np.float32)
    verts["rot_3"] = rotation_wxyz[:, 3].astype(np.float32)
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(out_path)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = load_pickle(Path(args.dynamic_pkl))
    motion = load_pickle(Path(args.dynamic_motion_pkl))

    xyz0 = to_numpy(state["xyz"]).astype(np.float64)
    scaling0 = to_numpy(state["scaling"]).astype(np.float64)
    rotation0 = to_numpy(state["rotation"]).astype(np.float64)  # wxyz
    features_dc = to_numpy(state["features_dc"]).astype(np.float64).reshape(-1, 3)
    opacity = to_numpy(state["opacity"]).astype(np.float64)
    normals = np.zeros_like(xyz0, dtype=np.float64)

    translation = to_numpy(motion["translation"]).astype(np.float64)
    global_scale = to_numpy(motion["scale"]).astype(np.float64).reshape(-1)

    align_scale, align_rotvec, align_trans = load_transform(args.base_transform_json)
    align_rot = Rotation.from_rotvec(align_rotvec)
    align_quat_xyzw = align_rot.as_quat()
    base_quat_xyzw = np.column_stack([rotation0[:, 1], rotation0[:, 2], rotation0[:, 3], rotation0[:, 0]])
    aligned_quat_xyzw = (Rotation.from_quat(np.repeat(align_quat_xyzw[None, :], base_quat_xyzw.shape[0], axis=0)) * Rotation.from_quat(base_quat_xyzw)).as_quat()
    aligned_quat_wxyz = np.column_stack([aligned_quat_xyzw[:, 3], aligned_quat_xyzw[:, 0], aligned_quat_xyzw[:, 1], aligned_quat_xyzw[:, 2]])

    meta = {
        "num_frames": int(translation.shape[0]),
        "num_points": int(xyz0.shape[0]),
        "base_transform_json": args.base_transform_json,
        "prefix": args.prefix,
    }
    (out_dir / "sequence_meta.json").write_text(json.dumps(meta, indent=2))

    for frame_idx in range(translation.shape[0]):
        frame_scale = float(global_scale[frame_idx])
        xyz = xyz0 * frame_scale + translation[frame_idx][None, :]
        xyz = (align_scale * (align_rot.as_matrix() @ xyz.T)).T + align_trans[None, :]

        scaling = scaling0 + np.log(max(frame_scale, 1e-8))
        scaling = scaling + np.log(max(align_scale, 1e-8))

        out_path = out_dir / f"{args.prefix}_{frame_idx:06d}.ply"
        write_frame_ply(out_path, xyz, normals, features_dc, opacity, scaling, aligned_quat_wxyz)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
