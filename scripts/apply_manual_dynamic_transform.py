#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


def load_dynamic_model(dynamic_pkl: Path, dynamic_motion_pkl: Path) -> np.ndarray:
    import torch

    with dynamic_pkl.open("rb") as f:
        state = pickle.load(f)
    with dynamic_motion_pkl.open("rb") as f:
        motion = pickle.load(f)

    xyz = state["xyz"].detach().cpu().numpy().astype(np.float64)
    translation = motion["translation"].detach().cpu().numpy().astype(np.float64)
    scale = motion["scale"].detach().cpu().numpy().astype(np.float64).reshape(-1, 1, 1)
    return xyz[None, :, :] * scale + translation[:, None, :]


def transform_points(points: np.ndarray, scale: float, rot_xyz_deg: np.ndarray, trans: np.ndarray) -> np.ndarray:
    rot = Rotation.from_euler("xyz", rot_xyz_deg, degrees=True).as_matrix()
    return (scale * (rot @ points.T)).T + trans[None, :]


def export_xyz_ply(out_path: Path, points: np.ndarray) -> None:
    verts = np.empty(points.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    verts["x"] = points[:, 0].astype(np.float32)
    verts["y"] = points[:, 1].astype(np.float32)
    verts["z"] = points[:, 2].astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(out_path)


def transform_splat_ply(in_path: Path, out_path: Path, scale: float, rot_xyz_deg: np.ndarray, trans: np.ndarray) -> None:
    ply = PlyData.read(str(in_path))
    vertex = ply["vertex"].data
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)
    rot = Rotation.from_euler("xyz", rot_xyz_deg, degrees=True)
    xyz_t = transform_points(xyz, scale, rot_xyz_deg, trans)

    data = np.array(vertex, copy=True)
    data["x"] = xyz_t[:, 0].astype(data["x"].dtype)
    data["y"] = xyz_t[:, 1].astype(data["y"].dtype)
    data["z"] = xyz_t[:, 2].astype(data["z"].dtype)

    # Rotate splat quaternion and add uniform scale to Gaussian scales when present.
    quat_fields = ["rot_0", "rot_1", "rot_2", "rot_3"]
    scale_fields = ["scale_0", "scale_1", "scale_2"]
    names = set(vertex.dtype.names or [])
    if all(f in names for f in quat_fields):
        q_in = np.column_stack([data["rot_1"], data["rot_2"], data["rot_3"], data["rot_0"]]).astype(np.float64)
        q_rot = rot.as_quat()
        q_out = (Rotation.from_quat(np.repeat(q_rot[None, :], q_in.shape[0], axis=0)) * Rotation.from_quat(q_in)).as_quat()
        data["rot_0"] = q_out[:, 3].astype(data["rot_0"].dtype)
        data["rot_1"] = q_out[:, 0].astype(data["rot_1"].dtype)
        data["rot_2"] = q_out[:, 1].astype(data["rot_2"].dtype)
        data["rot_3"] = q_out[:, 2].astype(data["rot_3"].dtype)
    if all(f in names for f in scale_fields):
        log_scale = np.log(scale)
        for field in scale_fields:
            data[field] = (data[field].astype(np.float64) + log_scale).astype(data[field].dtype)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a manual Sim(3) transform to DreamScene4D outputs.")
    parser.add_argument("--dynamic_pkl", required=True)
    parser.add_argument("--dynamic_motion_pkl", required=True)
    parser.add_argument("--dynamic_ply", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--rotation_deg_xyz", default=None, help="rx,ry,rz in degrees")
    parser.add_argument("--translation_xyz", default=None, help="tx,ty,tz")
    parser.add_argument("--base_transform_json", default=None)
    parser.add_argument("--delta_scale", type=float, default=1.0)
    parser.add_argument("--delta_rotation_deg_xyz", default="0,0,0")
    parser.add_argument("--delta_translation_xyz", default="0,0,0")
    return parser.parse_args()


def parse_vec3(text: str | None, default: np.ndarray) -> np.ndarray:
    if text is None:
        return default.copy()
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 values, got: {text}")
    return np.asarray(vals, dtype=np.float64)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_scale = 1.0
    base_rot_xyz_deg = np.zeros(3, dtype=np.float64)
    base_trans = np.zeros(3, dtype=np.float64)
    if args.base_transform_json:
        data = json.loads(Path(args.base_transform_json).read_text())
        base_scale = float(data["optimized_scale"])
        base_rot_xyz_deg = Rotation.from_rotvec(np.asarray(data["optimized_rotvec"], dtype=np.float64)).as_euler("xyz", degrees=True)
        base_trans = np.asarray(data["optimized_translation"], dtype=np.float64)

    scale = float(args.scale) if args.scale is not None else base_scale
    rot_xyz_deg = parse_vec3(args.rotation_deg_xyz, base_rot_xyz_deg)
    trans = parse_vec3(args.translation_xyz, base_trans)

    scale *= float(args.delta_scale)
    rot_xyz_deg = rot_xyz_deg + parse_vec3(args.delta_rotation_deg_xyz, np.zeros(3, dtype=np.float64))
    trans = trans + parse_vec3(args.delta_translation_xyz, np.zeros(3, dtype=np.float64))

    frames_xyz = load_dynamic_model(Path(args.dynamic_pkl), Path(args.dynamic_motion_pkl))
    if args.frame_idx < 0 or args.frame_idx >= frames_xyz.shape[0]:
        raise IndexError(f"frame_idx {args.frame_idx} out of range [0, {frames_xyz.shape[0] - 1}]")

    aligned_frame = transform_points(frames_xyz[args.frame_idx], scale, rot_xyz_deg, trans)
    export_xyz_ply(output_dir / f"aligned_dynamic_frame_{args.frame_idx:06d}.ply", aligned_frame)
    transform_splat_ply(Path(args.dynamic_ply), output_dir / "aligned_dynamic_snapshot_splat.ply", scale, rot_xyz_deg, trans)

    summary = {
        "frame_idx": args.frame_idx,
        "scale": scale,
        "rotation_deg_xyz": rot_xyz_deg.tolist(),
        "translation_xyz": trans.tolist(),
        "base_transform_json": args.base_transform_json,
        "delta_scale": args.delta_scale,
        "delta_rotation_deg_xyz": parse_vec3(args.delta_rotation_deg_xyz, np.zeros(3, dtype=np.float64)).tolist(),
        "delta_translation_xyz": parse_vec3(args.delta_translation_xyz, np.zeros(3, dtype=np.float64)).tolist(),
    }
    (output_dir / "manual_transform.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
