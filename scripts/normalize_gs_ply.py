#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def build_vertex_dtype(src_names: tuple[str, ...]) -> np.dtype:
    dtype_fields = []
    for name in src_names:
        dtype_fields.append((name, "f4"))
    return np.dtype(dtype_fields)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Gaussian-splat PLY into object-centric coordinates.")
    parser.add_argument("--input", required=True, help="Input GS ply")
    parser.add_argument("--output", required=True, help="Output normalized GS ply")
    parser.add_argument(
        "--center-mode",
        choices=["mean", "bbox"],
        default="mean",
        help="How to choose translation center",
    )
    parser.add_argument(
        "--target-max-extent",
        type=float,
        default=1.0,
        help="Scale normalized object so its max bbox extent matches this value",
    )
    args = parser.parse_args()

    src = PlyData.read(args.input)
    vertex = src["vertex"]
    names = vertex.data.dtype.names
    required = [
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"Missing required GS fields: {missing}")

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = xyz.mean(axis=0) if args.center_mode == "mean" else (mins + maxs) * 0.5
    extents = maxs - mins
    max_extent = float(extents.max())
    scale = 1.0 if max_extent <= 1e-8 else float(args.target_max_extent) / max_extent

    out_dtype = np.dtype(
        [
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
    )
    out = np.empty(vertex.count, dtype=out_dtype)

    xyz_norm = (xyz - center[None, :]) * scale
    out["x"] = xyz_norm[:, 0]
    out["y"] = xyz_norm[:, 1]
    out["z"] = xyz_norm[:, 2]
    out["nx"] = 0.0
    out["ny"] = 0.0
    out["nz"] = 0.0
    out["f_dc_0"] = np.asarray(vertex["f_dc_0"], dtype=np.float32)
    out["f_dc_1"] = np.asarray(vertex["f_dc_1"], dtype=np.float32)
    out["f_dc_2"] = np.asarray(vertex["f_dc_2"], dtype=np.float32)
    out["opacity"] = np.asarray(vertex["opacity"], dtype=np.float32)

    log_scale_delta = np.log(scale).astype(np.float32)
    out["scale_0"] = np.asarray(vertex["scale_0"], dtype=np.float32) + log_scale_delta
    out["scale_1"] = np.asarray(vertex["scale_1"], dtype=np.float32) + log_scale_delta
    out["scale_2"] = np.asarray(vertex["scale_2"], dtype=np.float32) + log_scale_delta
    out["rot_0"] = np.asarray(vertex["rot_0"], dtype=np.float32)
    out["rot_1"] = np.asarray(vertex["rot_1"], dtype=np.float32)
    out["rot_2"] = np.asarray(vertex["rot_2"], dtype=np.float32)
    out["rot_3"] = np.asarray(vertex["rot_3"], dtype=np.float32)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(out, "vertex")], text=False).write(out_path)

    print(f"saved {out_path}")
    print(f"center_mode={args.center_mode}")
    print(f"center={center.tolist()}")
    print(f"input_extents={extents.tolist()}")
    print(f"scale={scale}")


if __name__ == "__main__":
    main()
