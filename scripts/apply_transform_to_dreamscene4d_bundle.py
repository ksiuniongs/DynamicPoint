#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply an external Sim(3) placement transform to a DreamScene4D bundle."
    )
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--transform-json", required=True)
    parser.add_argument("--output-run-dir", required=True)
    parser.add_argument("--source-save-name", default=None)
    parser.add_argument("--output-save-name", default=None)
    return parser.parse_args()


def infer_save_name(run_dir: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    candidates = sorted(run_dir.glob("*_4d_model.ply"))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one *_4d_model.ply in {run_dir}, found {len(candidates)}")
    return candidates[0].name[: -len("_4d_model.ply")]


def to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def quat_wxyz_to_rotation(quat_wxyz: np.ndarray) -> Rotation:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    quat_xyzw = np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
        dtype=np.float64,
    )
    return Rotation.from_quat(quat_xyzw)


def rotation_to_quat_wxyz(rotation: Rotation) -> np.ndarray:
    quat_xyzw = rotation.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


def copy_bundle_files(
    source_run_dir: Path,
    output_run_dir: Path,
    source_save_name: str,
    output_save_name: str,
) -> None:
    output_run_dir.mkdir(parents=True, exist_ok=True)
    (output_run_dir / "gaussians").mkdir(parents=True, exist_ok=True)

    top_level_suffixes = [
        "_4d_model.ply",
        "_deformation.pth",
        "_deformation_table.pth",
        "_deformation_accum.pth",
    ]
    gaussian_suffixes = [
        "_4d.pkl",
    ]

    for suffix in top_level_suffixes:
        src = source_run_dir / f"{source_save_name}{suffix}"
        dst = output_run_dir / f"{output_save_name}{suffix}"
        shutil.copy2(src, dst)

    for suffix in gaussian_suffixes:
        src = source_run_dir / "gaussians" / f"{source_save_name}{suffix}"
        dst = output_run_dir / "gaussians" / f"{output_save_name}{suffix}"
        shutil.copy2(src, dst)


def transform_snapshot_ply(
    source_ply: Path,
    output_ply: Path,
    scale: float,
    rotation: Rotation,
    translation: np.ndarray,
) -> None:
    ply = PlyData.read(str(source_ply))
    vertex = ply["vertex"].data
    data = np.array(vertex, copy=True)

    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)
    xyz_t = (scale * rotation.apply(xyz)) + translation[None, :]
    data["x"] = xyz_t[:, 0].astype(data["x"].dtype)
    data["y"] = xyz_t[:, 1].astype(data["y"].dtype)
    data["z"] = xyz_t[:, 2].astype(data["z"].dtype)

    names = set(vertex.dtype.names or [])
    quat_fields = ["rot_0", "rot_1", "rot_2", "rot_3"]
    if all(field in names for field in quat_fields):
        local_q_xyzw = np.column_stack(
            [data["rot_1"], data["rot_2"], data["rot_3"], data["rot_0"]]
        ).astype(np.float64)
        local_rot = Rotation.from_quat(local_q_xyzw)
        world_rot = rotation * local_rot
        world_q = world_rot.as_quat()
        data["rot_0"] = world_q[:, 3].astype(data["rot_0"].dtype)
        data["rot_1"] = world_q[:, 0].astype(data["rot_1"].dtype)
        data["rot_2"] = world_q[:, 1].astype(data["rot_2"].dtype)
        data["rot_3"] = world_q[:, 2].astype(data["rot_3"].dtype)

    scale_fields = ["scale_0", "scale_1", "scale_2"]
    if all(field in names for field in scale_fields):
        log_scale = np.log(scale)
        for field in scale_fields:
            data[field] = (data[field].astype(np.float64) + log_scale).astype(data[field].dtype)

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(output_ply))


def main() -> None:
    args = parse_args()
    source_run_dir = Path(args.source_run_dir).resolve()
    output_run_dir = Path(args.output_run_dir).resolve()
    transform_json = Path(args.transform_json).resolve()

    source_save_name = infer_save_name(source_run_dir, args.source_save_name)
    output_save_name = args.output_save_name or f"{source_save_name}_dynpoint_aligned"

    with open(source_run_dir / "gaussians" / f"{source_save_name}_4d_global_motion.pkl", "rb") as f:
        motion = pickle.load(f)
    transform_info = json.loads(transform_json.read_text())

    ext_scale = float(transform_info["optimized_scale"])
    ext_rotation = Rotation.from_rotvec(np.asarray(transform_info["optimized_rotvec"], dtype=np.float64))
    ext_translation = np.asarray(transform_info["optimized_translation"], dtype=np.float64)

    old_base_scale = float(to_numpy(motion.get("base_scale", np.ones(1))).reshape(-1)[0])
    old_base_rotation = quat_wxyz_to_rotation(
        to_numpy(motion.get("base_rotation", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)))
    )
    old_translation = to_numpy(motion["translation"]).astype(np.float64)
    old_scale = to_numpy(motion["scale"]).astype(np.float64)

    # Treat the external transform as the new absolute bundle pose.
    # Preserve the learned relative global trajectory by remapping it from the old
    # base frame into the new frame.
    translation_ratio = ext_scale / max(old_base_scale, 1e-8)
    remapped_translation = translation_ratio * ext_rotation.apply(
        old_base_rotation.inv().apply(old_translation)
    )

    new_motion = {
        "translation": torch.from_numpy(remapped_translation.astype(np.float32)),
        "scale": torch.from_numpy(old_scale.astype(np.float32)),
        "base_translation": torch.from_numpy(ext_translation.astype(np.float32)),
        "base_scale": torch.tensor([ext_scale], dtype=torch.float32),
        "base_rotation": torch.from_numpy(rotation_to_quat_wxyz(ext_rotation)),
        "base_rotation_center": torch.zeros(3, dtype=torch.float32),
    }

    copy_bundle_files(
        source_run_dir=source_run_dir,
        output_run_dir=output_run_dir,
        source_save_name=source_save_name,
        output_save_name=output_save_name,
    )

    with open(output_run_dir / "gaussians" / f"{output_save_name}_4d_global_motion.pkl", "wb") as f:
        pickle.dump(new_motion, f)

    transform_snapshot_ply(
        source_run_dir / f"{source_save_name}_4d_model.ply",
        output_run_dir / f"{output_save_name}_aligned_snapshot.ply",
        scale=ext_scale,
        rotation=ext_rotation,
        translation=ext_translation,
    )

    summary = {
        "source_run_dir": str(source_run_dir),
        "source_save_name": source_save_name,
        "output_run_dir": str(output_run_dir),
        "output_save_name": output_save_name,
        "transform_json": str(transform_json),
        "mode": "replace_base_pose_preserve_relative_motion",
        "old_base_scale": old_base_scale,
        "new_base_scale": ext_scale,
        "new_base_translation": ext_translation.tolist(),
        "new_base_rotation_wxyz": rotation_to_quat_wxyz(ext_rotation).tolist(),
        "translation_ratio": translation_ratio,
        "object_source_center": transform_info.get("object_source_center"),
    }
    (output_run_dir / "bundle_transform_applied_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
