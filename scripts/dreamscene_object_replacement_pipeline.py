#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy import ndimage
from scipy.spatial.transform import Rotation


@dataclass
class PoseState:
    log_scale: float
    rotvec: np.ndarray
    translation: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align a high-quality object point cloud into a DreamScene360 panorama scene and export replacement assets."
    )
    parser.add_argument("--dreamscene_output_dir", required=True)
    parser.add_argument("--dreamscene_data_dir")
    parser.add_argument("--panorama_path")
    parser.add_argument("--mask_png")
    parser.add_argument("--bbox_xyxy", help="Fallback bbox x0,y0,x1,y1 if no mask is provided.")
    parser.add_argument("--object_ply", required=True)
    parser.add_argument("--object_kind", choices=["generic", "gs", "auto"], default="auto")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_limit", type=int, default=12000)
    parser.add_argument("--coarse_rotation_degrees", default="0,90,180,270")
    parser.add_argument("--search_iters", type=int, default=24)
    parser.add_argument("--search_candidates", type=int, default=40)
    parser.add_argument("--mask_dilate_px", type=int, default=2)
    parser.add_argument("--bbox_weight", type=float, default=0.15)
    parser.add_argument("--depth_weight", type=float, default=0.05)
    parser.add_argument("--prune_static", action="store_true")
    parser.add_argument("--prune_depth_margin_ratio", type=float, default=0.22)
    parser.add_argument("--make_preview", action="store_true")
    parser.add_argument("--preview_num_views", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_cfg_args(path: Path) -> dict:
    text = path.read_text().strip()
    safe_env = {"Namespace": lambda **kwargs: kwargs}
    cfg = eval(text, {"__builtins__": {}}, safe_env)  # noqa: S307
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Could not parse cfg_args: {path}")
    return cfg


def load_mask(mask_png: Path | None, bbox_xyxy: str | None, width: int, height: int) -> np.ndarray:
    if mask_png is not None:
        mask = np.array(Image.open(mask_png).convert("L")) > 127
        if mask.shape != (height, width):
            raise RuntimeError(
                f"Mask shape {mask.shape[::-1]} does not match panorama size {(width, height)}"
            )
        return mask
    if bbox_xyxy is None:
        raise RuntimeError("Either --mask_png or --bbox_xyxy is required.")
    vals = [float(x.strip()) for x in bbox_xyxy.split(",")]
    if len(vals) != 4:
        raise RuntimeError("--bbox_xyxy must be x0,y0,x1,y1")
    x0, y0, x1, y1 = vals
    mask = np.zeros((height, width), dtype=bool)
    x0i = max(int(math.floor(x0)), 0)
    y0i = max(int(math.floor(y0)), 0)
    x1i = min(int(math.ceil(x1)), width - 1)
    y1i = min(int(math.ceil(y1)), height - 1)
    if x1i <= x0i or y1i <= y0i:
        raise RuntimeError("bbox_xyxy is empty after clipping")
    mask[y0i : y1i + 1, x0i : x1i + 1] = True
    return mask


def pano_pixels_to_dirs(height: int, width: int) -> np.ndarray:
    i = (np.arange(height, dtype=np.float64) + 0.5) / float(height)
    j = (np.arange(width, dtype=np.float64) + 0.5) / float(width)
    ii, jj = np.meshgrid(i, j, indexing="ij")
    beta = -(ii - 0.5) * np.pi
    alpha = -(jj - 0.5) * 2.0 * np.pi
    dirs = np.stack(
        [
            np.cos(alpha) * np.cos(beta),
            np.sin(alpha) * np.cos(beta),
            np.sin(beta),
        ],
        axis=-1,
    )
    return dirs.astype(np.float64)


def points_to_pano_uvd(points: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(points, axis=1)
    valid = norms > 1e-8
    dirs = np.zeros_like(points)
    dirs[valid] = points[valid] / norms[valid, None]
    beta = np.arcsin(np.clip(dirs[:, 2], -1.0, 1.0))
    alpha = np.arctan2(dirs[:, 1], dirs[:, 0])
    row = (-beta / np.pi + 0.5) * height - 0.5
    col = (-alpha / (2.0 * np.pi) + 0.5) * width - 0.5
    col = np.mod(col, width)
    uvd = np.column_stack([col, row, norms])
    return uvd, valid


def build_scene_depth_from_ply(points: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    uvd, valid = points_to_pano_uvd(points, width, height)
    depth = np.full((height, width), np.inf, dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.int32)
    cols = np.clip(np.round(uvd[valid, 0]).astype(np.int32), 0, width - 1)
    rows = np.clip(np.round(uvd[valid, 1]).astype(np.int32), 0, height - 1)
    vals = uvd[valid, 2]
    for c, r, d in zip(cols, rows, vals):
        if d < depth[r, c]:
            depth[r, c] = d
        counts[r, c] += 1
    valid_mask = np.isfinite(depth)
    if not np.any(valid_mask):
        raise RuntimeError("Could not rasterize any depth from DreamScene points.")
    _, indices = ndimage.distance_transform_edt(~valid_mask, return_indices=True)
    filled = depth[tuple(indices)]
    return filled, valid_mask


def save_depth_preview(depth: np.ndarray, valid_mask: np.ndarray, out_png: Path) -> None:
    d = depth.copy()
    d[~np.isfinite(d)] = np.nan
    lo = np.nanpercentile(d, 5.0)
    hi = np.nanpercentile(d, 95.0)
    hi = max(hi, lo + 1e-6)
    d = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    d = np.nan_to_num(d, nan=0.0)
    color = plt.get_cmap("turbo")(d)[..., :3]
    color[~valid_mask] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(color * 255)).save(out_png)


def load_vertex(path: Path):
    ply = PlyData.read(str(path))
    return ply["vertex"].data


def detect_object_kind(vertex) -> str:
    names = set(vertex.dtype.names or [])
    if {"opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}.issubset(names):
        return "gs"
    return "generic"


def subsample_points(points: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if points.shape[0] <= limit:
        return points
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(points.shape[0], size=limit, replace=False))
    return points[idx]


def rotation_align_object_to_camera(center_dir: np.ndarray) -> Rotation:
    z_axis = -center_dir / max(np.linalg.norm(center_dir), 1e-8)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(z_axis, world_up)) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(world_up, z_axis)
    x_axis /= max(np.linalg.norm(x_axis), 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(np.linalg.norm(y_axis), 1e-8)
    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    return Rotation.from_matrix(rot)


def mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("Mask is empty.")
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def bbox_from_binary(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def bbox_iou(a: tuple[float, float, float, float] | None, b: tuple[float, float, float, float]) -> float:
    if a is None:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(ix1 - ix0 + 1.0, 0.0) * max(iy1 - iy0 + 1.0, 0.0)
    area_a = max(ax1 - ax0 + 1.0, 0.0) * max(ay1 - ay0 + 1.0, 0.0)
    area_b = max(bx1 - bx0 + 1.0, 0.0) * max(by1 - by0 + 1.0, 0.0)
    union = max(area_a + area_b - inter, 1e-6)
    return float(inter / union)


def apply_pose(points_centered: np.ndarray, pose: PoseState) -> np.ndarray:
    scale = math.exp(float(pose.log_scale))
    rot = Rotation.from_rotvec(pose.rotvec).as_matrix()
    return (scale * (rot @ points_centered.T)).T + pose.translation[None, :]


def rasterize_point_mask(
    points_world: np.ndarray,
    width: int,
    height: int,
    dilate_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    uvd, valid = points_to_pano_uvd(points_world, width, height)
    mask = np.zeros((height, width), dtype=bool)
    depth = np.full((height, width), np.inf, dtype=np.float64)
    if np.any(valid):
        cols = np.clip(np.round(uvd[valid, 0]).astype(np.int32), 0, width - 1)
        rows = np.clip(np.round(uvd[valid, 1]).astype(np.int32), 0, height - 1)
        vals = uvd[valid, 2]
        for c, r, d in zip(cols, rows, vals):
            mask[r, c] = True
            if d < depth[r, c]:
                depth[r, c] = d
    if dilate_px > 0:
        structure = ndimage.generate_binary_structure(2, 1)
        mask = ndimage.binary_dilation(mask, structure=structure, iterations=dilate_px)
    return mask, depth


def score_pose(
    points_centered: np.ndarray,
    pose: PoseState,
    target_mask: np.ndarray,
    scene_depth: np.ndarray,
    target_bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    dilate_px: int,
    bbox_weight: float,
    depth_weight: float,
) -> tuple[float, dict]:
    points_world = apply_pose(points_centered, pose)
    pred_mask, pred_depth = rasterize_point_mask(points_world, width, height, dilate_px)
    inter = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()
    mask_iou = float(inter / max(union, 1))
    pred_bbox = bbox_from_binary(pred_mask)
    pred_bbox_iou = bbox_iou(pred_bbox, target_bbox)
    depth_term = 0.0
    overlap = np.logical_and(pred_mask, target_mask) & np.isfinite(pred_depth)
    if np.any(overlap):
        rel = np.abs(pred_depth[overlap] - scene_depth[overlap]) / np.maximum(scene_depth[overlap], 1e-6)
        depth_term = float(np.exp(-np.median(rel)))
    score = mask_iou + bbox_weight * pred_bbox_iou + depth_weight * depth_term
    metrics = {
        "mask_iou": mask_iou,
        "bbox_iou": pred_bbox_iou,
        "depth_score": depth_term,
        "score": score,
    }
    return score, metrics


def estimate_initial_pose(
    object_points: np.ndarray,
    target_mask: np.ndarray,
    scene_depth: np.ndarray,
    pano_dirs: np.ndarray,
) -> tuple[PoseState, dict]:
    masked_dirs = pano_dirs[target_mask]
    masked_depth = scene_depth[target_mask]
    masked_points = masked_dirs * masked_depth[:, None]
    center = masked_points.mean(axis=0)
    center_depth = float(np.median(masked_depth))
    x0, y0, x1, y1 = mask_bbox(target_mask)
    height, width = target_mask.shape
    horiz_angle = 2.0 * np.pi * max((x1 - x0 + 1.0) / width, 1.0 / width)
    vert_angle = np.pi * max((y1 - y0 + 1.0) / height, 1.0 / height)
    target_span = max(
        2.0 * center_depth * math.tan(min(horiz_angle, np.pi - 1e-4) / 2.0),
        2.0 * center_depth * math.tan(min(vert_angle, np.pi - 1e-4) / 2.0),
    )
    extents = object_points.max(axis=0) - object_points.min(axis=0)
    base_extent = float(max(np.max(extents), 1e-6))
    scale = target_span / base_extent
    base_rot = rotation_align_object_to_camera(center / max(np.linalg.norm(center), 1e-8))
    pose = PoseState(
        log_scale=float(np.log(max(scale, 1e-6))),
        rotvec=base_rot.as_rotvec(),
        translation=center.astype(np.float64),
    )
    init_info = {
        "masked_depth_median": center_depth,
        "target_span_estimate": target_span,
        "initial_scale": scale,
        "initial_translation": center.tolist(),
        "initial_rotvec": pose.rotvec.tolist(),
    }
    return pose, init_info


def coarse_orientation_search(
    points_centered: np.ndarray,
    pose: PoseState,
    scene_depth: np.ndarray,
    pano_dirs: np.ndarray,
    target_mask: np.ndarray,
    coarse_rotation_degrees: str,
    dilate_px: int,
    bbox_weight: float,
    depth_weight: float,
) -> tuple[PoseState, dict]:
    target_bbox = mask_bbox(target_mask)
    best_pose = pose
    best_metrics = {"score": -1.0}
    height, width = target_mask.shape
    angle_grid = [float(x) for x in coarse_rotation_degrees.split(",") if x.strip()]
    base_rot = Rotation.from_rotvec(pose.rotvec)
    for rx in angle_grid:
        for ry in angle_grid:
            for rz in angle_grid:
                local = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True)
                candidate = PoseState(
                    log_scale=pose.log_scale,
                    rotvec=(base_rot * local).as_rotvec(),
                    translation=pose.translation.copy(),
                )
                score, metrics = score_pose(
                    points_centered,
                    candidate,
                    target_mask,
                    scene_depth,
                    target_bbox,
                    width,
                    height,
                    dilate_px,
                    bbox_weight,
                    depth_weight,
                )
                if score > best_metrics["score"]:
                    best_pose = candidate
                    best_metrics = metrics
                    best_metrics["coarse_euler_deg"] = [rx, ry, rz]
    return best_pose, best_metrics


def random_local_search(
    points_centered: np.ndarray,
    pose: PoseState,
    target_mask: np.ndarray,
    scene_depth: np.ndarray,
    search_iters: int,
    search_candidates: int,
    dilate_px: int,
    bbox_weight: float,
    depth_weight: float,
    seed: int,
) -> tuple[PoseState, dict, list[dict]]:
    rng = np.random.default_rng(seed)
    target_bbox = mask_bbox(target_mask)
    height, width = target_mask.shape
    extent_hint = np.linalg.norm(points_centered.max(axis=0) - points_centered.min(axis=0))
    trans_sigma = max(float(np.linalg.norm(pose.translation) * 0.08), extent_hint * math.exp(pose.log_scale) * 0.3, 0.05)
    rot_sigma = np.deg2rad(18.0)
    log_scale_sigma = 0.18
    best_pose = pose
    best_score, best_metrics = score_pose(
        points_centered,
        best_pose,
        target_mask,
        scene_depth,
        target_bbox,
        width,
        height,
        dilate_px,
        bbox_weight,
        depth_weight,
    )
    history = [{"iter": -1, **best_metrics}]
    for iter_idx in range(search_iters):
        iter_best_pose = best_pose
        iter_best_score = best_score
        iter_best_metrics = best_metrics
        for _ in range(search_candidates):
            candidate = PoseState(
                log_scale=float(best_pose.log_scale + rng.normal(0.0, log_scale_sigma)),
                rotvec=best_pose.rotvec + rng.normal(0.0, rot_sigma, size=3),
                translation=best_pose.translation + rng.normal(0.0, trans_sigma, size=3),
            )
            score, metrics = score_pose(
                points_centered,
                candidate,
                target_mask,
                scene_depth,
                target_bbox,
                width,
                height,
                dilate_px,
                bbox_weight,
                depth_weight,
            )
            if score > iter_best_score:
                iter_best_pose = candidate
                iter_best_score = score
                iter_best_metrics = metrics
        best_pose = iter_best_pose
        best_score = iter_best_score
        best_metrics = iter_best_metrics
        history.append({"iter": iter_idx, **best_metrics})
        trans_sigma *= 0.82
        rot_sigma *= 0.84
        log_scale_sigma *= 0.86
    return best_pose, best_metrics, history


def transform_vertex_positions(vertex, rot: Rotation, scale: float, translation: np.ndarray):
    out = np.array(vertex, copy=True)
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)
    transformed = (scale * rot.apply(xyz)) + translation[None, :]
    out["x"] = transformed[:, 0].astype(np.float32)
    out["y"] = transformed[:, 1].astype(np.float32)
    out["z"] = transformed[:, 2].astype(np.float32)
    names = set(vertex.dtype.names or [])
    if {"scale_0", "scale_1", "scale_2"}.issubset(names):
        for key in ("scale_0", "scale_1", "scale_2"):
            out[key] = (np.asarray(vertex[key], dtype=np.float64) + np.log(scale)).astype(np.float32)
    if {"rot_0", "rot_1", "rot_2", "rot_3"}.issubset(names):
        local_q = np.column_stack(
            [
                np.asarray(vertex["rot_0"], dtype=np.float64),
                np.asarray(vertex["rot_1"], dtype=np.float64),
                np.asarray(vertex["rot_2"], dtype=np.float64),
                np.asarray(vertex["rot_3"], dtype=np.float64),
            ]
        )
        local_q /= np.maximum(np.linalg.norm(local_q, axis=1, keepdims=True), 1e-12)
        global_xyzw = rot.as_quat()
        global_wxyz = np.array([global_xyzw[3], global_xyzw[0], global_xyzw[1], global_xyzw[2]], dtype=np.float64)
        gw, gx, gy, gz = global_wxyz
        lw, lx, ly, lz = local_q.T
        merged = np.empty_like(local_q)
        merged[:, 0] = gw * lw - gx * lx - gy * ly - gz * lz
        merged[:, 1] = gw * lx + gx * lw + gy * lz - gz * ly
        merged[:, 2] = gw * ly - gx * lz + gy * lw + gz * lx
        merged[:, 3] = gw * lz + gx * ly - gy * lx + gz * lw
        merged /= np.maximum(np.linalg.norm(merged, axis=1, keepdims=True), 1e-12)
        out["rot_0"] = merged[:, 0].astype(np.float32)
        out["rot_1"] = merged[:, 1].astype(np.float32)
        out["rot_2"] = merged[:, 2].astype(np.float32)
        out["rot_3"] = merged[:, 3].astype(np.float32)
    return out


def write_vertex(vertex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))


def filter_static_for_replacement(
    static_vertex,
    mask: np.ndarray,
    object_depth: np.ndarray,
    margin_ratio: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, int]:
    xyz = np.column_stack([static_vertex["x"], static_vertex["y"], static_vertex["z"]]).astype(np.float64)
    uvd, valid = points_to_pano_uvd(xyz, width, height)
    keep = np.ones(len(static_vertex), dtype=bool)
    for idx in np.where(valid)[0]:
        col = int(np.clip(np.round(uvd[idx, 0]), 0, width - 1))
        row = int(np.clip(np.round(uvd[idx, 1]), 0, height - 1))
        if not mask[row, col]:
            continue
        obj_depth = object_depth[row, col]
        if not np.isfinite(obj_depth):
            continue
        margin = max(obj_depth * margin_ratio, 1e-4)
        if abs(uvd[idx, 2] - obj_depth) <= margin:
            keep[idx] = False
    return np.array(static_vertex[keep], copy=True), int((~keep).sum())


def save_overlay(
    panorama_path: Path,
    target_mask: np.ndarray,
    pred_mask: np.ndarray,
    out_path: Path,
) -> None:
    pano = np.array(Image.open(panorama_path).convert("RGB")).astype(np.float32) / 255.0
    if pano.shape[:2] != target_mask.shape:
        pano = np.array(Image.fromarray(np.uint8(pano * 255)).resize((target_mask.shape[1], target_mask.shape[0]), Image.BILINEAR)).astype(np.float32) / 255.0
    overlay = np.clip(pano.copy(), 0.0, 1.0)
    overlay[target_mask] = 0.6 * overlay[target_mask] + 0.4 * np.array([0.0, 1.0, 0.0], dtype=np.float32)
    overlay[pred_mask] = 0.6 * overlay[pred_mask] + 0.4 * np.array([1.0, 0.0, 0.0], dtype=np.float32)
    overlap = target_mask & pred_mask
    overlay[overlap] = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(overlay, 0.0, 1.0) * 255)).save(out_path)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dreamscene_output_dir = Path(args.dreamscene_output_dir)
    cfg = load_cfg_args(dreamscene_output_dir / "cfg_args")
    dreamscene_data_dir = Path(args.dreamscene_data_dir or cfg["source_path"])
    panorama_path = Path(args.panorama_path or (dreamscene_data_dir / "frame_000001.png"))
    if not panorama_path.exists():
        pngs = sorted(dreamscene_data_dir.glob("*.png"))
        if not pngs:
            raise RuntimeError(f"No panorama png found in {dreamscene_data_dir}")
        panorama_path = pngs[0]

    pano_width = int(cfg.get("pano_width", Image.open(panorama_path).size[0]))
    pano_height = int(cfg.get("pano_height", Image.open(panorama_path).size[1]))
    mask = load_mask(Path(args.mask_png) if args.mask_png else None, args.bbox_xyxy, pano_width, pano_height)
    Image.fromarray(np.uint8(mask) * 255).save(out_dir / "mask_used.png")

    pano_dirs = pano_pixels_to_dirs(pano_height, pano_width)
    preproc_points_path = dreamscene_data_dir / "sparse" / "0" / "points3D.ply"
    if not preproc_points_path.exists():
        raise RuntimeError(f"Missing DreamScene sparse points: {preproc_points_path}")
    preproc_vertex = load_vertex(preproc_points_path)
    preproc_xyz = np.column_stack([preproc_vertex["x"], preproc_vertex["y"], preproc_vertex["z"]]).astype(np.float64)
    scene_depth, depth_valid = build_scene_depth_from_ply(preproc_xyz, pano_width, pano_height)
    np.savez_compressed(out_dir / "scene_depth_map.npz", depth=scene_depth, valid=depth_valid)
    save_depth_preview(scene_depth, depth_valid, out_dir / "scene_depth_preview.png")

    object_vertex = load_vertex(Path(args.object_ply))
    object_kind = detect_object_kind(object_vertex) if args.object_kind == "auto" else args.object_kind
    object_xyz = np.column_stack([object_vertex["x"], object_vertex["y"], object_vertex["z"]]).astype(np.float64)
    object_center = object_xyz.mean(axis=0)
    object_points_centered_full = object_xyz - object_center[None, :]
    object_points_centered = subsample_points(object_points_centered_full, args.sample_limit, args.seed)

    init_pose, init_info = estimate_initial_pose(object_points_centered, mask, scene_depth, pano_dirs)
    coarse_pose, coarse_metrics = coarse_orientation_search(
        object_points_centered,
        init_pose,
        scene_depth,
        pano_dirs,
        mask,
        args.coarse_rotation_degrees,
        args.mask_dilate_px,
        args.bbox_weight,
        args.depth_weight,
    )
    refined_pose, refined_metrics, history = random_local_search(
        object_points_centered,
        coarse_pose,
        mask,
        scene_depth,
        args.search_iters,
        args.search_candidates,
        args.mask_dilate_px,
        args.bbox_weight,
        args.depth_weight,
        args.seed,
    )

    final_rot = Rotation.from_rotvec(refined_pose.rotvec)
    final_scale = math.exp(refined_pose.log_scale)
    aligned_world_xyz = apply_pose(object_points_centered_full, refined_pose)
    aligned_vertex = transform_vertex_positions(
        object_vertex,
        final_rot,
        final_scale,
        refined_pose.translation - final_scale * final_rot.apply(object_center),
    )
    aligned_object_path = out_dir / "aligned_object.ply"
    write_vertex(aligned_vertex, aligned_object_path)

    pred_mask, object_depth = rasterize_point_mask(aligned_world_xyz, pano_width, pano_height, args.mask_dilate_px)
    save_overlay(panorama_path, mask, pred_mask, out_dir / "panorama_replacement_overlay.png")
    np.savez_compressed(out_dir / "aligned_object_depth_map.npz", depth=object_depth)

    scene_static_ply = dreamscene_output_dir / "point_cloud" / "iteration_10000" / "point_cloud.ply"
    static_vertex = load_vertex(scene_static_ply)
    replacement_info: dict[str, object] = {
        "dreamscene_output_dir": str(dreamscene_output_dir),
        "dreamscene_data_dir": str(dreamscene_data_dir),
        "panorama_path": str(panorama_path),
        "mask_png": str(args.mask_png) if args.mask_png else None,
        "bbox_xyxy": args.bbox_xyxy,
        "object_ply": str(Path(args.object_ply)),
        "object_kind": object_kind,
        "pano_width": pano_width,
        "pano_height": pano_height,
        "init_info": init_info,
        "coarse_metrics": coarse_metrics,
        "refined_metrics": refined_metrics,
        "search_history": history,
        "final_scale": final_scale,
        "final_rotvec": refined_pose.rotvec.tolist(),
        "final_translation": refined_pose.translation.tolist(),
        "object_source_center": object_center.tolist(),
    }

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = final_scale * final_rot.as_matrix()
    matrix[:3, 3] = refined_pose.translation - final_scale * final_rot.apply(object_center)
    replacement_info["object_to_world_matrix"] = matrix.tolist()

    if args.prune_static:
        pruned_static, removed = filter_static_for_replacement(
            static_vertex,
            ndimage.binary_dilation(mask, iterations=args.mask_dilate_px),
            object_depth,
            args.prune_depth_margin_ratio,
            pano_width,
            pano_height,
        )
        pruned_path = out_dir / "pruned_static_scene.ply"
        write_vertex(pruned_static, pruned_path)
        replacement_info["pruned_static_scene"] = str(pruned_path)
        replacement_info["pruned_static_removed_points"] = removed
    else:
        pruned_path = scene_static_ply

    merge_json = out_dir / "aligned_object_transform_for_merge.json"
    rotvec = final_rot.as_rotvec()
    merge_json.write_text(
        json.dumps(
            {
                "optimized_scale": final_scale,
                "optimized_rotvec": rotvec.tolist(),
                "optimized_translation": (refined_pose.translation - final_scale * final_rot.apply(object_center)).tolist(),
            },
            indent=2,
        )
    )
    replacement_info["merge_transform_json"] = str(merge_json)

    if object_kind == "gs":
        merged_path = out_dir / "merged_scene_with_object.ply"
        subprocess.run(
            [
                sys.executable,
                "/mnt/d/develop/master_thesis/DynamicPoint/scripts/merge_registered_gaussians.py",
                "--static_ply",
                str(pruned_path),
                "--dynamic_ply",
                str(Path(args.object_ply)),
                "--transform_json",
                str(merge_json),
                "--output_ply",
                str(merged_path),
            ],
            check=True,
        )
        replacement_info["merged_scene_ply"] = str(merged_path)

    if args.make_preview:
        preview_dir = out_dir / "preview"
        subprocess.run(
            [
                sys.executable,
                "/mnt/d/develop/master_thesis/DynamicPoint/scripts/render_registered_novel_views.py",
                "--static_ply",
                str(pruned_path),
                "--dynamic_ply",
                str(aligned_object_path),
                "--output_dir",
                str(preview_dir),
                "--num_views",
                str(args.preview_num_views),
            ],
            check=True,
        )
        replacement_info["preview_dir"] = str(preview_dir)

    (out_dir / "replacement_manifest.json").write_text(json.dumps(replacement_info, indent=2))
    print(out_dir / "replacement_manifest.json")


if __name__ == "__main__":
    main()
