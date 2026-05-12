#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


def read_vertex(path: Path):
    return PlyData.read(str(path))["vertex"].data


def normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return q / n


def quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    out = np.empty_like(q1)
    out[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    out[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    out[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    out[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return normalize_quat_wxyz(out)


def build_dtype(static_names: tuple[str, ...]) -> np.dtype:
    return np.dtype([(name, "f4") for name in static_names])


def vertex_to_dict(vtx, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    out = {}
    for name in names:
        out[name] = np.asarray(vtx[name], dtype=np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge static and aligned dynamic gaussian splats into one PLY.")
    parser.add_argument("--static_ply", required=True)
    parser.add_argument("--dynamic_ply", required=True)
    parser.add_argument("--transform_json", required=True)
    parser.add_argument("--output_ply", required=True)
    args = parser.parse_args()

    static_v = read_vertex(Path(args.static_ply))
    dynamic_v = read_vertex(Path(args.dynamic_ply))
    static_names = static_v.dtype.names
    if static_names is None:
        raise RuntimeError("Static PLY has no vertex fields")

    info = json.loads(Path(args.transform_json).read_text())
    global_scale = float(info["optimized_scale"])
    global_rot = Rotation.from_rotvec(np.asarray(info["optimized_rotvec"], dtype=np.float64))
    global_t = np.asarray(info["optimized_translation"], dtype=np.float64)
    global_q_xyzw = global_rot.as_quat()
    global_q_wxyz = np.array([global_q_xyzw[3], global_q_xyzw[0], global_q_xyzw[1], global_q_xyzw[2]], dtype=np.float64)

    out_dtype = build_dtype(static_names)
    static_out = np.empty(len(static_v), dtype=out_dtype)
    for name in static_names:
        static_out[name] = np.asarray(static_v[name], dtype=np.float32)

    dynamic_out = np.zeros(len(dynamic_v), dtype=out_dtype)
    dyn_names = set(dynamic_v.dtype.names or [])
    for name in static_names:
        if name in dyn_names:
            dynamic_out[name] = np.asarray(dynamic_v[name], dtype=np.float32)

    dyn_xyz = np.column_stack(
        [
            np.asarray(dynamic_v["x"], dtype=np.float64),
            np.asarray(dynamic_v["y"], dtype=np.float64),
            np.asarray(dynamic_v["z"], dtype=np.float64),
        ]
    )
    dyn_xyz = (global_scale * global_rot.apply(dyn_xyz)) + global_t[None, :]
    dynamic_out["x"] = dyn_xyz[:, 0].astype(np.float32)
    dynamic_out["y"] = dyn_xyz[:, 1].astype(np.float32)
    dynamic_out["z"] = dyn_xyz[:, 2].astype(np.float32)

    for k in ["scale_0", "scale_1", "scale_2"]:
        dynamic_out[k] = (np.asarray(dynamic_v[k], dtype=np.float64) + np.log(global_scale)).astype(np.float32)

    local_q = np.column_stack(
        [
            np.asarray(dynamic_v["rot_0"], dtype=np.float64),
            np.asarray(dynamic_v["rot_1"], dtype=np.float64),
            np.asarray(dynamic_v["rot_2"], dtype=np.float64),
            np.asarray(dynamic_v["rot_3"], dtype=np.float64),
        ]
    )
    local_q = normalize_quat_wxyz(local_q)
    global_q = np.repeat(global_q_wxyz[None, :], len(local_q), axis=0)
    merged_q = quat_multiply_wxyz(global_q, local_q)
    dynamic_out["rot_0"] = merged_q[:, 0].astype(np.float32)
    dynamic_out["rot_1"] = merged_q[:, 1].astype(np.float32)
    dynamic_out["rot_2"] = merged_q[:, 2].astype(np.float32)
    dynamic_out["rot_3"] = merged_q[:, 3].astype(np.float32)

    # Missing SH residual channels in DreamScene4D output are filled with zeros.
    merged = np.concatenate([static_out, dynamic_out], axis=0)
    out_path = Path(args.output_ply)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(merged, "vertex")], text=False).write(str(out_path))
    print(out_path)
    print(f"static={len(static_out)} dynamic={len(dynamic_out)} merged={len(merged)}")


if __name__ == "__main__":
    main()
