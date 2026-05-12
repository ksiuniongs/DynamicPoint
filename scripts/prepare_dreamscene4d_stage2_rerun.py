#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a fresh DreamScene4D stage2 rerun workspace from a transformed target pose."
    )
    parser.add_argument("--source-stage1-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--transform-json", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--dreamscene-root", required=True)
    parser.add_argument("--source-save-name", default=None)
    parser.add_argument("--new-save-name", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1000)
    return parser.parse_args()


def infer_stage1_save_name(stage1_dir: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    candidates = sorted(stage1_dir.glob("*_model.ply"))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one *_model.ply in {stage1_dir}, found {len(candidates)}")
    return candidates[0].name[: -len("_model.ply")]


def rotation_to_quat_wxyz(rotation: Rotation) -> np.ndarray:
    quat_xyzw = rotation.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def write_stage2_config(
    config_path: Path,
    dreamscene_root: Path,
    workspace_root: Path,
    new_save_name: str,
    batch_size: int,
    iters: int,
) -> None:
    image_dir = workspace_root / "data" / "JPEGImages" / "ut_x5_right"
    mask_dir = workspace_root / "data" / "Annotations" / "ut_x5_right" / "001"
    stage1_dir = workspace_root / "stage1_inputs"
    outdir = workspace_root / "stage2_run"
    visdir = outdir / "vis"
    config_text = f"""### Input
input: {image_dir}
input_mask: {mask_dir}
cam_pose:
prompt:
mesh:
elevation: 0
ref_size: 256
density_thresh: 0.5

### Output
visdir: {visdir}
outdir: {outdir}
mesh_format: obj
save_path: {new_save_name}

### Training
gmflow_path: './gmflow/pretrained/gmflow_kitti-285701a8.pth'
flow_backend: gmflow
waft_root: /mnt/d/develop/4D/submodules/WAFT
waft_cfg: /mnt/d/develop/4D/submodules/WAFT/config/a1/tar-c-t.json
waft_ckpt: /mnt/d/develop/4D/submodules/WAFT/ckpts/a1/adaptation.pth
waft_scale: 0
lambda_sd: 0
stable_zero123: False
mvdream: False
lambda_zero123: 1
lambda_svd: 0
zero123_backend: zero123
hf_cache_dir: /mnt/d/develop/master_thesis/.hf_cache
batch_size: {batch_size}
iters: {iters}
iters_refine: 50
radius: 2
fovy: 49.1
load:
train_geo: False
invert_bg_prob: 0.
n_views: 4
t_max: 0.5
resize_square: False
depth_loss: False
stage1_from: {stage1_dir}
resume_4d: False

### GUI
force_cuda_rast: False
H: 512
W: 512
render_bg: False

### Gaussian splatting
num_pts: 5000
sh_degree: 0
position_lr_init: 0.001
position_lr_final: 0.00002
position_lr_delay_mult: 0.02
position_lr_max_steps: 500
feature_lr: 0.01
opacity_lr: 0.05
scaling_lr: 0.005
rotation_lr: 0.005
percent_dense: 0.1
density_start_iter: 30000
density_end_iter: 30000
densification_interval: 100
opacity_reset_interval: 70000
densify_grad_threshold: 0.05
optimize_sh: False

### deformation field
deformation_lr_init: 0.00064
deformation_lr_final: 0.00064
deformation_lr_delay_mult: 0.01
grid_lr_init: 0.0064
grid_lr_final: 0.0064

### Textured Mesh
geom_lr: 0.0001
texture_lr: 0.2
"""
    config_path.write_text(config_text)


def main() -> None:
    args = parse_args()
    source_stage1_dir = Path(args.source_stage1_dir).resolve()
    image_dir = Path(args.image_dir).resolve()
    mask_dir = Path(args.mask_dir).resolve()
    transform_json = Path(args.transform_json).resolve()
    workspace_root = Path(args.workspace_root).resolve()
    dreamscene_root = Path(args.dreamscene_root).resolve()
    source_save_name = infer_stage1_save_name(source_stage1_dir, args.source_save_name)
    new_save_name = args.new_save_name

    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    (workspace_root / "stage1_inputs" / "gaussians").mkdir(parents=True, exist_ok=True)
    (workspace_root / "data" / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (workspace_root / "data" / "Annotations").mkdir(parents=True, exist_ok=True)
    (workspace_root / "stage2_run" / "vis").mkdir(parents=True, exist_ok=True)
    (workspace_root / "stage2_run" / "gaussians").mkdir(parents=True, exist_ok=True)

    copy_tree(image_dir, workspace_root / "data" / "JPEGImages" / image_dir.name)
    copy_tree(mask_dir.parent, workspace_root / "data" / "Annotations" / mask_dir.parent.name)

    shutil.copy2(
        source_stage1_dir / f"{source_save_name}_model.ply",
        workspace_root / "stage1_inputs" / f"{new_save_name}_model.ply",
    )
    shutil.copy2(
        source_stage1_dir / "gaussians" / f"{source_save_name}.pkl",
        workspace_root / "stage1_inputs" / "gaussians" / f"{new_save_name}.pkl",
    )

    transform_info = json.loads(transform_json.read_text())
    ext_scale = float(transform_info["optimized_scale"])
    ext_rotation = Rotation.from_rotvec(np.asarray(transform_info["optimized_rotvec"], dtype=np.float64))
    ext_translation = np.asarray(transform_info["optimized_translation"], dtype=np.float64)

    stage1_motion = {
        "translation": torch.tensor(ext_translation, dtype=torch.float32),
        "scale": torch.tensor([ext_scale], dtype=torch.float32),
        "rotation": torch.tensor(rotation_to_quat_wxyz(ext_rotation), dtype=torch.float32),
        "rotation_center": torch.zeros(3, dtype=torch.float32),
    }

    for suffix in ["_global_motion.pkl", "_calibrated_pose.pkl"]:
        with open(
            workspace_root / "stage1_inputs" / "gaussians" / f"{new_save_name}{suffix}",
            "wb",
        ) as f:
            pickle.dump(stage1_motion, f)

    config_path = workspace_root / f"{new_save_name}_stage2.yaml"
    write_stage2_config(
        config_path=config_path,
        dreamscene_root=dreamscene_root,
        workspace_root=workspace_root,
        new_save_name=new_save_name,
        batch_size=args.batch_size,
        iters=args.iters,
    )

    summary = {
        "workspace_root": str(workspace_root),
        "stage1_inputs": str(workspace_root / "stage1_inputs"),
        "copied_image_dir": str(workspace_root / "data" / "JPEGImages" / image_dir.name),
        "copied_mask_dir": str(workspace_root / "data" / "Annotations" / mask_dir.parent.name / mask_dir.name),
        "stage2_outdir": str(workspace_root / "stage2_run"),
        "config_path": str(config_path),
        "source_stage1_dir": str(source_stage1_dir),
        "source_save_name": source_save_name,
        "new_save_name": new_save_name,
        "transform_json": str(transform_json),
    }
    (workspace_root / "prepare_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
