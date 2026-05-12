#!/usr/bin/env python3
import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path("/mnt/d/develop/master_thesis/external/UniSH")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unish.utils.inference_utils import (  # noqa: E402
    generate_mixed_geometries_in_memory,
    load_model,
    process_video,
    run_inference,
    save_camera_parameters_per_frame,
    save_human_point_clouds,
    save_scene_only_point_clouds,
    save_smpl_meshes_per_frame,
)


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("export_unish_geometry")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "export_geometry.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def main() -> None:
    ap = argparse.ArgumentParser(description="Export UniSH geometry without visualization rendering")
    ap.add_argument("--video_path", required=True, help="Input video or image directory")
    ap.add_argument("--output_dir", required=True, help="Output directory")
    ap.add_argument("--body_models_path", required=True, help="Path to SMPL body models")
    ap.add_argument("--checkpoint", default="checkpoints/unish_release.safetensors")
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--original_fps", type=float, default=30.0)
    ap.add_argument("--target_size", type=int, default=518)
    ap.add_argument("--chunk_size", type=int, default=8)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--human_idx", type=int, default=0)
    ap.add_argument("--bbox_scale", type=float, default=1.0)
    ap.add_argument("--conf_thres", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--yolo_ckpt", default="ckpts/yolo11n.pt")
    ap.add_argument("--sam2_model", default="facebook/sam2-hiera-large")
    ap.add_argument("--start_idx", type=int, default=None)
    ap.add_argument("--end_idx", type=int, default=None)
    args = ap.parse_args()

    setup_seed(args.seed)
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(args.gpu_id)
        logger.info(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, using CPU")

    logger.info("Loading UniSH model...")
    model = load_model(args.checkpoint).to(device)
    model.eval()

    logger.info("Processing input frames...")
    data_dict = process_video(
        args.video_path,
        args.fps,
        args.human_idx,
        args.target_size,
        bbox_scale=args.bbox_scale,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        original_fps=args.original_fps,
        yolo_ckpt=args.yolo_ckpt,
        sam2_model=args.sam2_model,
    )

    logger.info("Running UniSH inference...")
    results = run_inference(model, data_dict, device, args.chunk_size)
    logger.info("Generating mixed geometries in memory...")
    viz_scene_point_clouds, _, viz_scene_only_point_clouds, _ = generate_mixed_geometries_in_memory(
        results,
        args.body_models_path,
        fps=args.fps,
        conf_thres=args.conf_thres,
    )

    logger.info("Saving scene-only point clouds...")
    save_scene_only_point_clouds(viz_scene_only_point_clouds, str(output_dir), results["seq_name"])

    logger.info("Saving human-only point clouds...")
    save_human_point_clouds(
        viz_scene_point_clouds,
        viz_scene_only_point_clouds,
        str(output_dir),
        results["seq_name"],
        results,
    )

    logger.info("Saving SMPL meshes...")
    save_smpl_meshes_per_frame(results, str(output_dir), args.body_models_path)

    logger.info("Saving camera parameters...")
    save_camera_parameters_per_frame(results, str(output_dir), results["seq_name"])

    logger.info("Geometry export completed.")


if __name__ == "__main__":
    main()
