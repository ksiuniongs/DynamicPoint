#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from plyfile import PlyData, PlyElement
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass
class CameraFrame:
    width: int
    height: int
    K: np.ndarray
    w2c: np.ndarray
    native_width: int
    native_height: int


@dataclass
class MaskBox:
    x0: float
    y0: float
    x1: float
    y1: float
    width: int
    height: int

    @property
    def cx(self) -> float:
        return 0.5 * (self.x0 + self.x1)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y0 + self.y1)

    @property
    def w(self) -> float:
        return self.x1 - self.x0 + 1.0

    @property
    def h(self) -> float:
        return self.y1 - self.y0 + 1.0

    @property
    def obj_cx(self) -> float:
        return (self.cx / self.width) * 2.0 - 1.0

    @property
    def obj_cy(self) -> float:
        return (self.cy / self.height) * 2.0 - 1.0

    @property
    def input_scale(self) -> float:
        width_n = (self.x1 - self.x0) / self.width
        height_n = (self.y1 - self.y0) / self.height
        return max(width_n / 0.65, height_n / 0.65)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mask-constrained coarse Sim(3) alignment for a ply/GS model.")
    parser.add_argument("--src_ply", required=True)
    parser.add_argument("--src_kind", choices=["gs", "generic"], default="gs")
    parser.add_argument("--mask_png", required=True)
    parser.add_argument("--camera_npz", required=True)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--ref_ply", help="Optional reference point cloud to provide a depth anchor.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--point_limit", type=int, default=30000)
    parser.add_argument("--crop_percentile", type=float, default=5.0)
    parser.add_argument("--opacity_threshold", type=float, default=0.7)
    parser.add_argument("--coarse_rotation_degrees", default="0,90,180,270")
    parser.add_argument("--freeze_rotation", action="store_true")
    parser.add_argument("--max_nfev", type=int, default=150)
    parser.add_argument("--mask_refine_iters", type=int, default=0)
    parser.add_argument("--target_mask_iou", type=float, default=0.8)
    return parser.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sample_points(points: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or points.shape[0] <= limit:
        return np.arange(points.shape[0])
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(points.shape[0], size=limit, replace=False))


def load_source_points(path: Path, kind: str, opacity_threshold: float, crop_percentile: float, point_limit: int) -> np.ndarray:
    ply = PlyData.read(path)
    v = ply.elements[0].data
    points = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    if kind == "gs" and "opacity" in v.dtype.names:
        opacity = sigmoid(np.asarray(v["opacity"]).astype(np.float64))
        mask = opacity > opacity_threshold
        lo = crop_percentile
        hi = 100.0 - crop_percentile
        for axis in range(3):
            low = np.percentile(points[:, axis], lo)
            high = np.percentile(points[:, axis], hi)
            mask &= points[:, axis] >= low
            mask &= points[:, axis] <= high
        points = points[mask]
    keep = sample_points(points, point_limit)
    return points[keep]


def infer_native_image_size(intrinsics: np.ndarray) -> tuple[int, int]:
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    native_width = max(int(round(cx * 2.0)), 1)
    native_height = max(int(round(cy * 2.0)), 1)
    return native_width, native_height


def rescale_intrinsics(intrinsics: np.ndarray, src_width: int, src_height: int, dst_width: int, dst_height: int) -> np.ndarray:
    K = intrinsics.copy().astype(np.float64)
    sx = float(dst_width) / float(max(src_width, 1))
    sy = float(dst_height) / float(max(src_height, 1))
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K


def load_camera_frame(camera_npz: Path, frame_idx: int, width: int, height: int) -> CameraFrame:
    obj = np.load(camera_npz)
    intrinsics = obj["intrinsics"][frame_idx].astype(np.float64)
    extrinsics = obj["extrinsics"][frame_idx].astype(np.float64)
    native_width, native_height = infer_native_image_size(intrinsics)
    intrinsics_scaled = rescale_intrinsics(intrinsics, native_width, native_height, width, height)
    return CameraFrame(
        width=width,
        height=height,
        K=intrinsics_scaled,
        w2c=extrinsics,
        native_width=native_width,
        native_height=native_height,
    )


def load_mask_box(mask_png: Path) -> MaskBox:
    mask = np.array(Image.open(mask_png).convert("L"))
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        raise RuntimeError(f"Mask is empty: {mask_png}")
    return MaskBox(
        x0=float(xs.min()),
        y0=float(ys.min()),
        x1=float(xs.max()),
        y1=float(ys.max()),
        width=mask.shape[1],
        height=mask.shape[0],
    )


def load_mask_binary(mask_png: Path) -> np.ndarray:
    return np.array(Image.open(mask_png).convert("L")) > 127


def transform_points(points: np.ndarray, log_scale: float, quat_wxyz: np.ndarray, translation: np.ndarray, rotation_center: np.ndarray) -> np.ndarray:
    scale = float(np.exp(log_scale))
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    rotation = Rotation.from_quat(quat_xyzw).as_matrix()
    centered = points - rotation_center[None, :]
    transformed = (scale * (rotation @ centered.T)).T + rotation_center[None, :] + translation[None, :]
    return transformed


def project_points(world_xyz: np.ndarray, cam: CameraFrame) -> tuple[np.ndarray, np.ndarray]:
    xyz_cam = (cam.w2c[:3, :3] @ world_xyz.T).T + cam.w2c[:3, 3][None, :]
    valid = xyz_cam[:, 2] > 1e-5
    uv = np.empty((world_xyz.shape[0], 2), dtype=np.float64)
    uv[:] = np.nan
    fx, fy = cam.K[0, 0], cam.K[1, 1]
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    uv[valid, 0] = fx * (xyz_cam[valid, 0] / xyz_cam[valid, 2]) + cx
    uv[valid, 1] = fy * (xyz_cam[valid, 1] / xyz_cam[valid, 2]) + cy
    return uv, valid


def robust_bbox(uv: np.ndarray, valid: np.ndarray, cam: CameraFrame) -> tuple[float, float, float, float] | None:
    pts = uv[valid]
    if pts.shape[0] < 16:
        return None
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 16:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    qx0, qx1 = np.percentile(x, [3.0, 97.0])
    qy0, qy1 = np.percentile(y, [3.0, 97.0])
    qx0 = float(np.clip(qx0, 0, cam.width - 1))
    qx1 = float(np.clip(qx1, 0, cam.width - 1))
    qy0 = float(np.clip(qy0, 0, cam.height - 1))
    qy1 = float(np.clip(qy1, 0, cam.height - 1))
    if qx1 <= qx0 or qy1 <= qy0:
        return None
    return qx0, qy0, qx1, qy1


def bbox_iou(a: tuple[float, float, float, float] | None, b: MaskBox) -> float:
    if a is None:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b.x0, b.y0, b.x1, b.y1
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    area_a = max(ax1 - ax0, 0.0) * max(ay1 - ay0, 0.0)
    area_b = max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0)
    union = max(area_a + area_b - inter, 1e-6)
    return inter / union


def estimate_splat_radius_px(uv: np.ndarray, valid: np.ndarray) -> int:
    pts = uv[valid]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 8:
        return 2
    if pts.shape[0] > 2000:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(pts.shape[0], size=2000, replace=False)]
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    nn = dists[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if nn.size == 0:
        return 2
    radius = int(np.clip(np.ceil(np.median(nn) * 0.6), 1, 6))
    return radius


def disk_structure(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def projected_mask_from_points(uv: np.ndarray, valid: np.ndarray, cam: CameraFrame) -> np.ndarray:
    pts = uv[valid]
    pts = pts[np.isfinite(pts).all(axis=1)]
    raster = np.zeros((cam.height, cam.width), dtype=bool)
    if pts.shape[0] == 0:
        return raster
    xi = np.rint(pts[:, 0]).astype(np.int32)
    yi = np.rint(pts[:, 1]).astype(np.int32)
    inside = (xi >= 0) & (xi < cam.width) & (yi >= 0) & (yi < cam.height)
    xi = xi[inside]
    yi = yi[inside]
    raster[yi, xi] = True
    radius = estimate_splat_radius_px(uv, valid)
    structure = disk_structure(radius)
    raster = ndimage.binary_dilation(raster, structure=structure)
    raster = ndimage.binary_closing(raster, structure=disk_structure(max(radius, 1)))
    raster = ndimage.binary_fill_holes(raster)
    return raster


def mask_iou(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    target = target_mask.astype(bool)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def evaluate_mask_iou(
    points: np.ndarray,
    cam: CameraFrame,
    target_mask: np.ndarray,
    rotation_center: np.ndarray,
    log_scale: float,
    quat: np.ndarray,
    translation: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, tuple[float, float, float, float] | None]:
    transformed = transform_points(points, log_scale, quat, translation, rotation_center)
    uv, valid = project_points(transformed, cam)
    pred_mask = projected_mask_from_points(uv, valid, cam)
    score = mask_iou(pred_mask, target_mask)
    pred_bbox = robust_bbox(uv, valid, cam)
    return score, transformed, pred_mask, pred_bbox


def refine_with_mask_iou(
    points: np.ndarray,
    cam: CameraFrame,
    target_mask: np.ndarray,
    rotation_center: np.ndarray,
    init_log_scale: float,
    quat: np.ndarray,
    init_translation: np.ndarray,
    iters: int,
    target_score: float,
) -> tuple[float, np.ndarray, float, int]:
    rng = np.random.default_rng(0)
    best_log_scale = float(init_log_scale)
    best_translation = np.asarray(init_translation, dtype=np.float64).copy()
    best_score, _, _, _ = evaluate_mask_iou(
        points, cam, target_mask, rotation_center, best_log_scale, quat, best_translation
    )
    if best_score >= target_score or iters <= 0:
        return best_log_scale, best_translation, best_score, 0

    scale_step = 0.12
    trans_step = 0.18
    accepted = 0
    for idx in range(iters):
        frac = 1.0 - (idx / max(iters, 1))
        proposal_log_scale = best_log_scale + rng.normal(0.0, scale_step * frac)
        proposal_translation = best_translation + rng.normal(0.0, trans_step * frac, size=3)
        score, _, _, _ = evaluate_mask_iou(
            points, cam, target_mask, rotation_center, proposal_log_scale, quat, proposal_translation
        )
        if score > best_score:
            best_score = score
            best_log_scale = proposal_log_scale
            best_translation = proposal_translation
            accepted += 1
            if best_score >= target_score:
                return best_log_scale, best_translation, best_score, idx + 1
    return best_log_scale, best_translation, best_score, iters


def quat_wxyz_from_euler(xyz_deg: list[float]) -> np.ndarray:
    q_xyzw = Rotation.from_euler("xyz", xyz_deg, degrees=True).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def normalized_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def residual(
    params: np.ndarray,
    points: np.ndarray,
    cam: CameraFrame,
    obs: MaskBox,
    rotation_center: np.ndarray,
    anchor_center: np.ndarray,
) -> np.ndarray:
    log_scale = float(params[0])
    quat = normalized_quat_wxyz(params[1:5])
    translation = params[5:8]
    pts = transform_points(points, log_scale, quat, translation, rotation_center)
    uv, valid = project_points(pts, cam)
    pred = robust_bbox(uv, valid, cam)
    if pred is None:
        return np.array([3.0, 3.0, 1.0, 1.0, 3.0, 3.0, 3.0], dtype=np.float64)
    px0, py0, px1, py1 = pred
    pcx = 0.5 * (px0 + px1)
    pcy = 0.5 * (py0 + py1)
    pw = max(px1 - px0 + 1.0, 1.0)
    ph = max(py1 - py0 + 1.0, 1.0)
    center_pred = pts.mean(axis=0)
    return np.array(
        [
            (pcx - obs.cx) / cam.width,
            (pcy - obs.cy) / cam.height,
            math.log(pw / obs.w),
            math.log(ph / obs.h),
            (center_pred[0] - anchor_center[0]) / 5.0,
            (center_pred[1] - anchor_center[1]) / 5.0,
            (center_pred[2] - anchor_center[2]) / 5.0,
        ],
        dtype=np.float64,
    )


def residual_fixed_rotation(
    params: np.ndarray,
    points: np.ndarray,
    cam: CameraFrame,
    obs: MaskBox,
    rotation_center: np.ndarray,
    anchor_center: np.ndarray,
    fixed_quat: np.ndarray,
) -> np.ndarray:
    log_scale = float(params[0])
    translation = params[1:4]
    pts = transform_points(points, log_scale, fixed_quat, translation, rotation_center)
    uv, valid = project_points(pts, cam)
    pred = robust_bbox(uv, valid, cam)
    if pred is None:
        return np.array([3.0, 3.0, 1.0, 1.0, 3.0, 3.0, 3.0], dtype=np.float64)
    px0, py0, px1, py1 = pred
    pcx = 0.5 * (px0 + px1)
    pcy = 0.5 * (py0 + py1)
    pw = max(px1 - px0 + 1.0, 1.0)
    ph = max(py1 - py0 + 1.0, 1.0)
    center_pred = pts.mean(axis=0)
    return np.array(
        [
            (pcx - obs.cx) / cam.width,
            (pcy - obs.cy) / cam.height,
            math.log(pw / obs.w),
            math.log(ph / obs.h),
            (center_pred[0] - anchor_center[0]) / 5.0,
            (center_pred[1] - anchor_center[1]) / 5.0,
            (center_pred[2] - anchor_center[2]) / 5.0,
        ],
        dtype=np.float64,
    )


def save_overlay(image_size: tuple[int, int], obs: MaskBox, pred: tuple[float, float, float, float] | None, out_path: Path, label: str) -> None:
    image = Image.new("RGB", image_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([obs.x0, obs.y0, obs.x1, obs.y1], outline=(0, 255, 0), width=3)
    if pred is not None:
        draw.rectangle(list(pred), outline=(255, 0, 0), width=3)
    draw.text((12, 12), label, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def save_mask_overlay(target_mask: np.ndarray, pred_mask: np.ndarray, out_path: Path, label: str) -> None:
    image = np.full((target_mask.shape[0], target_mask.shape[1], 3), 255, dtype=np.uint8)
    target = target_mask.astype(bool)
    pred = pred_mask.astype(bool)
    image[target] = np.array([180, 255, 180], dtype=np.uint8)
    image[pred] = np.array([255, 180, 180], dtype=np.uint8)
    overlap = target & pred
    image[overlap] = np.array([255, 230, 0], dtype=np.uint8)
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    draw.text((12, 12), label, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def export_points(points: np.ndarray, out_path: Path) -> None:
    verts = np.empty(points.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    verts["x"] = points[:, 0].astype(np.float32)
    verts["y"] = points[:, 1].astype(np.float32)
    verts["z"] = points[:, 2].astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(out_path)


def save_transformed_gs(src_path: Path, transform: np.ndarray, out_path: Path) -> None:
    ply = PlyData.read(src_path)
    vertex = np.array(ply.elements[0].data, copy=True)
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    linear = transform[:3, :3]
    translation = transform[:3, 3]
    scale = float(np.cbrt(abs(np.linalg.det(linear))))
    rotation_matrix = linear / max(scale, 1e-8)
    xyz_transformed = (xyz @ rotation_matrix.T) * scale + translation
    vertex["x"] = xyz_transformed[:, 0].astype(vertex["x"].dtype)
    vertex["y"] = xyz_transformed[:, 1].astype(vertex["y"].dtype)
    vertex["z"] = xyz_transformed[:, 2].astype(vertex["z"].dtype)
    rot_fields = [f"rot_{i}" for i in range(4)]
    if all(field in vertex.dtype.names for field in rot_fields):
        rots = np.stack([vertex[field] for field in rot_fields], axis=1).astype(np.float64)
        local_rot = Rotation.from_quat(np.stack([rots[:, 1], rots[:, 2], rots[:, 3], rots[:, 0]], axis=1))
        global_rot = Rotation.from_matrix(rotation_matrix)
        composed = global_rot * local_rot
        q = composed.as_quat()
        vertex["rot_0"] = q[:, 3].astype(vertex["rot_0"].dtype)
        vertex["rot_1"] = q[:, 0].astype(vertex["rot_1"].dtype)
        vertex["rot_2"] = q[:, 1].astype(vertex["rot_2"].dtype)
        vertex["rot_3"] = q[:, 2].astype(vertex["rot_3"].dtype)
    scale_fields = [f"scale_{i}" for i in range(3)]
    if all(field in vertex.dtype.names for field in scale_fields) and scale > 0:
        log_scale = np.log(scale)
        for field in scale_fields:
            vertex[field] = (vertex[field].astype(np.float64) + log_scale).astype(vertex[field].dtype)
    elements = [PlyElement.describe(vertex, ply.elements[0].name)]
    for element in ply.elements[1:]:
        elements.append(PlyElement.describe(np.array(element.data, copy=True), element.name))
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(out_path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    box = load_mask_box(Path(args.mask_png))
    target_mask = load_mask_binary(Path(args.mask_png))
    cam = load_camera_frame(Path(args.camera_npz), args.frame_idx, box.width, box.height)
    points = load_source_points(Path(args.src_ply), args.src_kind, args.opacity_threshold, args.crop_percentile, args.point_limit)
    rotation_center = points.mean(axis=0)
    source_height = float(np.max(np.ptp(points, axis=0)))

    if args.ref_ply:
        ref_pcd = o3d.io.read_point_cloud(args.ref_ply)
        ref_points = np.asarray(ref_pcd.points)
        ref_center_world = ref_points.mean(axis=0)
        ref_cam = (cam.w2c[:3, :3] @ ref_points.T).T + cam.w2c[:3, 3][None, :]
        positive = ref_cam[:, 2] > 1e-5
        depth_hint = float(np.median(ref_cam[positive, 2])) if np.any(positive) else 1.0
    else:
        ref_center_world = rotation_center.copy()
        cam_center = -cam.w2c[:3, :3].T @ cam.w2c[:3, 3]
        depth_hint = float(np.linalg.norm(rotation_center - cam_center))

    fx = cam.K[0, 0]
    fy = cam.K[1, 1]
    obj_height_world = box.h * depth_hint / max(fy, 1e-6)
    init_scale = max(obj_height_world / max(source_height, 1e-6), 1e-4)

    x_cam = (box.cx - cam.K[0, 2]) * depth_hint / max(fx, 1e-6)
    y_cam = (box.cy - cam.K[1, 2]) * depth_hint / max(fy, 1e-6)
    cam_xyz = np.array([x_cam, y_cam, depth_hint], dtype=np.float64)
    cam_to_world = np.linalg.inv(cam.w2c)
    anchor_center = cam_to_world[:3, :3] @ cam_xyz + cam_to_world[:3, 3]
    init_translation = anchor_center - init_scale * rotation_center

    if args.freeze_rotation:
        best_iou = bbox_iou(
            robust_bbox(*project_points(transform_points(points, math.log(init_scale), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), init_translation, rotation_center), cam), cam),
            box,
        )
        best_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        x0 = np.concatenate([[math.log(init_scale)], init_translation])
        result = least_squares(
            residual_fixed_rotation,
            x0,
            loss="soft_l1",
            f_scale=0.25,
            max_nfev=args.max_nfev,
            args=(points, cam, box, rotation_center, anchor_center, best_quat),
        )
        log_scale = float(result.x[0])
        quat = best_quat
        translation = result.x[1:4]
    else:
        angle_grid = [float(x) for x in args.coarse_rotation_degrees.split(",") if x.strip()]
        best_iou = -1.0
        best_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        for rx in angle_grid:
            for ry in angle_grid:
                for rz in angle_grid:
                    quat = quat_wxyz_from_euler([rx, ry, rz])
                    pts = transform_points(points, math.log(init_scale), quat, init_translation, rotation_center)
                    uv, valid = project_points(pts, cam)
                    pred = robust_bbox(uv, valid, cam)
                    iou = bbox_iou(pred, box)
                    if iou > best_iou:
                        best_iou = iou
                        best_quat = quat

        x0 = np.concatenate([[math.log(init_scale)], best_quat, init_translation])
        result = least_squares(
            residual,
            x0,
            loss="soft_l1",
            f_scale=0.25,
            max_nfev=args.max_nfev,
            args=(points, cam, box, rotation_center, anchor_center),
        )
        log_scale = float(result.x[0])
        quat = normalized_quat_wxyz(result.x[1:5])
        translation = result.x[5:8]

    pre_refine_log_scale = float(log_scale)
    pre_refine_translation = np.asarray(translation, dtype=np.float64).copy()
    pre_refine_mask_iou, _, _, _ = evaluate_mask_iou(
        points, cam, target_mask, rotation_center, pre_refine_log_scale, quat, pre_refine_translation
    )
    mask_refine_steps = 0
    mask_refine_reached_target = False
    if args.mask_refine_iters > 0:
        log_scale, translation, refined_mask_iou, mask_refine_steps = refine_with_mask_iou(
            points,
            cam,
            target_mask,
            rotation_center,
            pre_refine_log_scale,
            quat,
            pre_refine_translation,
            args.mask_refine_iters,
            args.target_mask_iou,
        )
        mask_refine_reached_target = refined_mask_iou >= args.target_mask_iou

    scale = float(np.exp(log_scale))
    q_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)
    rotation_matrix = Rotation.from_quat(q_xyzw).as_matrix()
    transformed = transform_points(points, log_scale, quat, translation, rotation_center)
    uv, valid = project_points(transformed, cam)
    pred = robust_bbox(uv, valid, cam)
    final_iou = bbox_iou(pred, box)
    pred_mask = projected_mask_from_points(uv, valid, cam)
    final_mask_iou = mask_iou(pred_mask, target_mask)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation_matrix
    transform[:3, 3] = translation + rotation_center - scale * (rotation_matrix @ rotation_center)

    summary = {
        "frame_idx": args.frame_idx,
        "camera_native_size": {"width": cam.native_width, "height": cam.native_height},
        "camera_target_size": {"width": cam.width, "height": cam.height},
        "camera_scale_factors": {
            "sx": float(cam.width) / float(max(cam.native_width, 1)),
            "sy": float(cam.height) / float(max(cam.native_height, 1)),
        },
        "mask_box": {"x0": box.x0, "y0": box.y0, "x1": box.x1, "y1": box.y1},
        "obj_cx": box.obj_cx,
        "obj_cy": box.obj_cy,
        "input_scale": box.input_scale,
        "depth_hint": depth_hint,
        "initial_scale": init_scale,
        "initial_translation": init_translation.tolist(),
        "coarse_rotation_quat_wxyz": best_quat.tolist(),
        "coarse_rotation_iou": best_iou,
        "freeze_rotation": bool(args.freeze_rotation),
        "mask_refine_iters": int(args.mask_refine_iters),
        "target_mask_iou": float(args.target_mask_iou),
        "optimized_scale": scale,
        "optimized_translation": translation.tolist(),
        "optimized_rotation_quat_wxyz": quat.tolist(),
        "optimized_rotation_center": rotation_center.tolist(),
        "pre_refine_mask_iou": pre_refine_mask_iou,
        "mask_refine_steps": int(mask_refine_steps),
        "mask_refine_reached_target": bool(mask_refine_reached_target),
        "final_bbox_iou": final_iou,
        "final_mask_iou": final_mask_iou,
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": result.message,
    }

    (output_dir / "transform.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(output_dir / "estimated_transform.npz", estimated_transform=transform)
    export_points(transformed, output_dir / "aligned_points.ply")
    save_overlay((box.width, box.height), box, pred, output_dir / "overlay_bbox.png", f"IoU={final_iou:.4f}")
    save_mask_overlay(target_mask, pred_mask, output_dir / "overlay_mask.png", f"mask IoU={final_mask_iou:.4f}")

    pose_dict = {
        "translation": translation,
        "scale": np.array([scale], dtype=np.float64),
        "rotation": quat,
        "rotation_center": rotation_center,
    }
    with (output_dir / "calibrated_pose.pkl").open("wb") as f:
        pickle.dump(pose_dict, f)

    if args.src_kind == "gs":
        save_transformed_gs(Path(args.src_ply), transform, output_dir / "aligned_source_gs.ply")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
