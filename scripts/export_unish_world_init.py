#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement

REPO_ROOT = Path("/mnt/d/develop/master_thesis/external/UniSH")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unish.utils.inference_utils import load_model, process_video  # noqa: E402


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


def resize_mask(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    mask_img = o3d.geometry.Image(mask.astype(np.uint8) * 255)
    resized = np.asarray(mask_img)
    if resized.shape != (h, w):
        import cv2

        resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a cleaned UniSH world-point initialization.")
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--output_ply", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--checkpoint", default="checkpoints/unish_release.safetensors")
    ap.add_argument("--conf_thres", type=float, default=0.35)
    ap.add_argument("--voxel_size", type=float, default=0.03)
    ap.add_argument("--nb_neighbors", type=int, default=20)
    ap.add_argument("--std_ratio", type=float, default=2.0)
    ap.add_argument("--target_size", type=int, default=518)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--original_fps", type=float, default=6.0)
    ap.add_argument("--chunk_size", type=int, default=8)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--human_idx", type=int, default=0)
    ap.add_argument("--bbox_scale", type=float, default=1.0)
    ap.add_argument("--yolo_ckpt", default="ckpts/yolo11n.pt")
    ap.add_argument("--sam2_model", default="facebook/sam2-hiera-large")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    model = load_model(args.checkpoint).to(device)
    model.eval()
    data_dict = process_video(
        args.frames_dir,
        args.fps,
        args.human_idx,
        args.target_size,
        bbox_scale=args.bbox_scale,
        original_fps=args.original_fps,
        yolo_ckpt=args.yolo_ckpt,
        sam2_model=args.sam2_model,
    )

    images = data_dict["images"]
    human_patches = data_dict["human_patches"]
    bbox_info = data_dict["bbox_info"]
    human_masks = data_dict["human_masks"]

    world_xyz = []
    world_rgb = []
    per_frame = []

    total_frames = images.shape[0]
    for start in range(0, total_frames, args.chunk_size):
        end = min(start + args.chunk_size, total_frames)
        input_dict = {
            "images": images[start:end].unsqueeze(0).to(device),
            "human_patches": human_patches[args.human_idx, start:end].unsqueeze(0).to(device=device, dtype=torch.bfloat16),
            "bbox_info": bbox_info[args.human_idx, start:end].unsqueeze(0).to(device),
        }
        with torch.no_grad():
            pred = model.inference(input_dict)

        points = pred["world_points"].squeeze(0).cpu().numpy()
        conf = pred["point_conf"].squeeze(0).cpu().numpy()
        rgbs = images[start:end].permute(0, 2, 3, 1).cpu().numpy()
        if rgbs.max() > 1.0:
            rgbs = rgbs / 255.0

        for local_idx in range(end - start):
            frame_idx = start + local_idx
            pts = points[local_idx]
            cf = conf[local_idx]
            rgb = rgbs[local_idx]
            mask = cf > args.conf_thres

            human_mask = human_masks[args.human_idx, frame_idx].cpu().numpy()
            if human_mask.shape != mask.shape:
                import cv2

                human_mask = cv2.resize(
                    human_mask.astype(np.uint8),
                    (mask.shape[1], mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            mask &= ~human_mask

            kept_xyz = pts[mask]
            kept_rgb = np.clip(rgb[mask] * 255.0, 0, 255).astype(np.uint8)
            per_frame.append(
                {
                    "frame_idx": frame_idx,
                    "raw_points": int(np.prod(mask.shape)),
                    "kept_points": int(len(kept_xyz)),
                }
            )
            if len(kept_xyz):
                world_xyz.append(kept_xyz.astype(np.float32))
                world_rgb.append(kept_rgb)

    xyz = np.concatenate(world_xyz, axis=0)
    rgb = np.concatenate(world_rgb, axis=0)

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

    output_ply = Path(args.output_ply)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    write_ply(output_ply, xyz_out, rgb_out)

    summary = {
        "frames_dir": args.frames_dir,
        "conf_thres": args.conf_thres,
        "voxel_size": args.voxel_size,
        "nb_neighbors": args.nb_neighbors,
        "std_ratio": args.std_ratio,
        "raw_merged_points": int(len(xyz)),
        "final_points": int(len(xyz_out)),
        "per_frame": per_frame,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
