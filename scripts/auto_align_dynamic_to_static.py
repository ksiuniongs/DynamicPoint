#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData, PlyElement
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
GS_UTILS = REPO_ROOT / "submodules" / "gaussian-splatting" / "utils"
if str(GS_UTILS) not in sys.path:
    sys.path.append(str(GS_UTILS))

from read_write_model import read_model  # type: ignore


@dataclass
class FrameCamera:
    name: str
    frame_idx: int
    view: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R_wc: np.ndarray
    t_wc: np.ndarray
    point2d: np.ndarray
    point3d_ids: np.ndarray

    @property
    def camera_center(self) -> np.ndarray:
        return -self.R_wc.T @ self.t_wc


@dataclass
class MaskBox:
    frame_idx: int
    view: str
    x0: float
    y0: float
    x1: float
    y1: float

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


def load_dynamic_model(dynamic_pkl: Path, dynamic_motion_pkl: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    with dynamic_pkl.open("rb") as f:
        state = pickle.load(f)
    with dynamic_motion_pkl.open("rb") as f:
        motion = pickle.load(f)

    xyz = state["xyz"].detach().cpu().numpy().astype(np.float64)
    translation = motion["translation"].detach().cpu().numpy().astype(np.float64)
    scale = motion["scale"].detach().cpu().numpy().astype(np.float64).reshape(-1, 1, 1)
    frames_xyz = xyz[None, :, :] * scale + translation[:, None, :]
    return xyz, frames_xyz, translation


def parse_image_name(name: str) -> tuple[int, str]:
    stem = Path(name).stem
    frame_str, view = stem.split("_", 1)
    return int(frame_str), view


def load_colmap_cameras(model_dir: Path, views: set[str]) -> tuple[dict[tuple[str, int], FrameCamera], dict[int, np.ndarray]]:
    cameras, images, points3d = read_model(str(model_dir), ext=".bin")
    frame_cameras: dict[tuple[str, int], FrameCamera] = {}

    for _, image in images.items():
        frame_idx, view = parse_image_name(image.name)
        if view not in views:
            continue
        cam = cameras[image.camera_id]
        params = np.asarray(cam.params, dtype=np.float64)
        if cam.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            fx = fy = float(params[0])
            cx = float(params[1])
            cy = float(params[2])
        elif cam.model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
            fx = float(params[0])
            fy = float(params[1])
            cx = float(params[2])
            cy = float(params[3])
        else:
            raise ValueError(f"Unsupported camera model: {cam.model}")

        frame_cameras[(view, frame_idx)] = FrameCamera(
            name=image.name,
            frame_idx=frame_idx,
            view=view,
            width=int(cam.width),
            height=int(cam.height),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            R_wc=image.qvec2rotmat().astype(np.float64),
            t_wc=np.asarray(image.tvec, dtype=np.float64),
            point2d=np.asarray(image.xys, dtype=np.float64),
            point3d_ids=np.asarray(image.point3D_ids, dtype=np.int64),
        )

    xyz_by_id = {int(pid): np.asarray(point.xyz, dtype=np.float64) for pid, point in points3d.items()}
    return frame_cameras, xyz_by_id


def load_mask_boxes(mask_dir: Path, view: str) -> dict[tuple[str, int], MaskBox]:
    boxes: dict[tuple[str, int], MaskBox] = {}
    for mask_path in sorted(mask_dir.glob("*.png")):
        frame_idx = int(mask_path.stem.split("_")[0])
        mask = np.array(Image.open(mask_path).convert("L"))
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        boxes[(view, frame_idx)] = MaskBox(
            frame_idx=frame_idx,
            view=view,
            x0=float(xs.min()),
            y0=float(ys.min()),
            x1=float(xs.max()),
            y1=float(ys.max()),
        )
    return boxes


def get_depth_hint(cam: FrameCamera, box: MaskBox, xyz_by_id: dict[int, np.ndarray], expand: float = 1.2) -> float:
    ids = cam.point3d_ids
    valid = ids >= 0
    if not np.any(valid):
        raise RuntimeError(f"No valid sparse points for image {cam.name}")
    xys = cam.point2d[valid]
    ids = ids[valid]

    bw = box.w
    bh = box.h
    ex0 = box.cx - 0.5 * bw * expand
    ex1 = box.cx + 0.5 * bw * expand
    ey0 = box.cy - 0.5 * bh * expand
    ey1 = box.cy + 0.5 * bh * expand

    selected_depths: list[float] = []
    fallback_depths: list[float] = []
    for xy, pid in zip(xys, ids):
        xyz = xyz_by_id.get(int(pid))
        if xyz is None:
            continue
        cam_xyz = cam.R_wc @ xyz + cam.t_wc
        if cam_xyz[2] <= 1e-4:
            continue
        depth = float(cam_xyz[2])
        fallback_depths.append(depth)
        if ex0 <= xy[0] <= ex1 and ey0 <= xy[1] <= ey1:
            selected_depths.append(depth)

    if selected_depths:
        return float(np.median(selected_depths))
    return float(np.median(fallback_depths))


def estimate_world_up(cams: dict[tuple[str, int], FrameCamera], frame_keys: list[tuple[str, int]]) -> np.ndarray:
    up_vectors = []
    cam_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    for key in frame_keys:
        cam = cams[key]
        up_w = cam.R_wc.T @ cam_up
        norm = np.linalg.norm(up_w)
        if norm > 1e-8:
            up_vectors.append(up_w / norm)
    if not up_vectors:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    up = np.mean(np.stack(up_vectors, axis=0), axis=0)
    norm = np.linalg.norm(up)
    if norm < 1e-8:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return up / norm


def backproject_to_world(cam: FrameCamera, px: float, py: float, depth: float) -> np.ndarray:
    x = (px - cam.cx) / cam.fx
    y = (py - cam.cy) / cam.fy
    cam_xyz = np.array([x * depth, y * depth, depth], dtype=np.float64)
    return cam.R_wc.T @ (cam_xyz - cam.t_wc)


def project_points(world_xyz: np.ndarray, cam: FrameCamera) -> tuple[np.ndarray, np.ndarray]:
    cam_xyz = (cam.R_wc @ world_xyz.T).T + cam.t_wc[None, :]
    valid = cam_xyz[:, 2] > 1e-5
    uv = np.empty((world_xyz.shape[0], 2), dtype=np.float64)
    uv[:] = np.nan
    uv[valid, 0] = cam.fx * (cam_xyz[valid, 0] / cam_xyz[valid, 2]) + cam.cx
    uv[valid, 1] = cam.fy * (cam_xyz[valid, 1] / cam_xyz[valid, 2]) + cam.cy
    return uv, valid


def robust_bbox(uv: np.ndarray, valid: np.ndarray, cam: FrameCamera) -> tuple[float, float, float, float] | None:
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


def transform_points(points: np.ndarray, log_scale: float, rotvec: np.ndarray, trans: np.ndarray) -> np.ndarray:
    scale = float(np.exp(log_scale))
    R = Rotation.from_rotvec(rotvec).as_matrix()
    return (scale * (R @ points.T)).T + trans[None, :]


def lowest_point_along_up(points: np.ndarray, up_world: np.ndarray) -> np.ndarray:
    proj = points @ up_world
    return points[int(np.argmin(proj))]


def build_residual(
    params: np.ndarray,
    frame_keys: list[tuple[str, int]],
    frame_points: np.ndarray,
    cams: dict[tuple[str, int], FrameCamera],
    boxes: dict[tuple[str, int], MaskBox],
    first_anchor: np.ndarray,
    first_foot_anchor: np.ndarray,
    up_world: np.ndarray,
    foot_weight: float,
    foot_sigma: float,
) -> np.ndarray:
    log_scale = float(params[0])
    rotvec = params[1:4]
    trans = params[4:7]
    residuals: list[float] = []

    for frame_key in frame_keys:
        _, frame_idx = frame_key
        pts = transform_points(frame_points[frame_idx], log_scale, rotvec, trans)
        uv, valid = project_points(pts, cams[frame_key])
        pred = robust_bbox(uv, valid, cams[frame_key])
        obs = boxes[frame_key]
        if pred is None:
            residuals.extend([3.0, 3.0, 1.0, 1.0])
            continue
        px0, py0, px1, py1 = pred
        pcx = 0.5 * (px0 + px1)
        pcy = 0.5 * (py0 + py1)
        pw = max(px1 - px0 + 1.0, 1.0)
        ph = max(py1 - py0 + 1.0, 1.0)
        residuals.append((pcx - obs.cx) / cams[frame_key].width)
        residuals.append((pcy - obs.cy) / cams[frame_key].height)
        residuals.append(math.log(pw / obs.w))
        residuals.append(math.log(ph / obs.h))

    first_pts = transform_points(frame_points[frame_keys[0][1]], log_scale, rotvec, trans)
    anchor_pred = first_pts.mean(axis=0)
    residuals.extend(((anchor_pred - first_anchor) / 5.0).tolist())
    if foot_weight > 0.0:
        foot_pred = lowest_point_along_up(first_pts, up_world)
        residuals.extend((foot_weight * (foot_pred - first_foot_anchor) / max(foot_sigma, 1e-6)).tolist())
    return np.asarray(residuals, dtype=np.float64)


def export_aligned_ply(out_path: Path, points: np.ndarray) -> None:
    verts = np.empty(points.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    verts["x"] = points[:, 0].astype(np.float32)
    verts["y"] = points[:, 1].astype(np.float32)
    verts["z"] = points[:, 2].astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(out_path)


def make_overlay(
    image_path: Path,
    obs: MaskBox,
    pred: tuple[float, float, float, float] | None,
    out_path: Path,
    label: str,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle([obs.x0, obs.y0, obs.x1, obs.y1], outline=(0, 255, 0), width=3)
    if pred is not None:
        draw.rectangle(list(pred), outline=(255, 0, 0), width=3)
    draw.text((12, 12), label, fill=(255, 255, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def make_contact_sheet(image_paths: list[Path], out_path: Path, cols: int = 3) -> None:
    images = [Image.open(p).convert("RGB") for p in image_paths]
    if not images:
        return
    w, h = images[0].size
    rows = math.ceil(len(images) / cols)
    canvas = Image.new("RGB", (cols * w, rows * h), color=(0, 0, 0))
    for idx, img in enumerate(images):
        x = (idx % cols) * w
        y = (idx // cols) * h
        canvas.paste(img, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-align DreamScene4D dynamic object to static COLMAP scene.")
    parser.add_argument("--colmap_model", required=True)
    parser.add_argument("--view_image_dir", action="append", default=[], help="view=/path/to/images")
    parser.add_argument("--view_mask_dir", action="append", default=[], help="view=/path/to/masks")
    parser.add_argument("--dynamic_pkl", required=True)
    parser.add_argument("--dynamic_motion_pkl", required=True)
    parser.add_argument("--dynamic_ply", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frame_ids", default="0,5,10,15,19")
    parser.add_argument("--foot_weight", type=float, default=1.0)
    parser.add_argument("--foot_sigma", type=float, default=0.35)
    return parser.parse_args()


def parse_view_mapping(items: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected view mapping as view=/path, got: {item}")
        view, raw_path = item.split("=", 1)
        mapping[view.strip()] = Path(raw_path.strip())
    if not mapping:
        raise ValueError("At least one view mapping is required.")
    return mapping


def main() -> None:
    args = parse_args()
    model_dir = Path(args.colmap_model)
    image_dirs = parse_view_mapping(args.view_image_dir)
    mask_dirs = parse_view_mapping(args.view_mask_dir)
    dynamic_pkl = Path(args.dynamic_pkl)
    dynamic_motion_pkl = Path(args.dynamic_motion_pkl)
    dynamic_ply = Path(args.dynamic_ply)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    views = set(image_dirs.keys()) & set(mask_dirs.keys())
    if not views:
        raise RuntimeError("No overlapping views between image dirs and mask dirs.")
    cams, xyz_by_id = load_colmap_cameras(model_dir, views)
    boxes: dict[tuple[str, int], MaskBox] = {}
    for view in sorted(views):
        boxes.update(load_mask_boxes(mask_dirs[view], view))
    _, frame_points, _ = load_dynamic_model(dynamic_pkl, dynamic_motion_pkl)
    requested_frames = [int(x.strip()) for x in args.frame_ids.split(",") if x.strip()]
    frame_keys: list[tuple[str, int]] = []
    for frame_idx in requested_frames:
        if frame_idx >= frame_points.shape[0]:
            continue
        for view in sorted(views):
            key = (view, frame_idx)
            if key in cams and key in boxes:
                frame_keys.append(key)
    if not frame_keys:
        raise RuntimeError("No overlapping frame ids between COLMAP, masks, and dynamic motion.")

    first_cam = cams[frame_keys[0]]
    first_box = boxes[frame_keys[0]]
    depth_hint = get_depth_hint(first_cam, first_box, xyz_by_id)
    first_anchor = backproject_to_world(first_cam, first_box.cx, first_box.cy, depth_hint)
    first_foot_anchor = backproject_to_world(first_cam, first_box.cx, first_box.y1, depth_hint)
    up_world = estimate_world_up(cams, frame_keys)

    base_pts0 = frame_points[frame_keys[0][1]]
    base_center = base_pts0.mean(axis=0)
    base_extent = np.ptp(base_pts0, axis=0)
    object_height = float(np.max(base_extent))
    init_scale = max((first_box.h * depth_hint) / max(first_cam.fy * object_height, 1e-6), 1e-4)
    init_rotvec = np.zeros(3, dtype=np.float64)
    init_trans = first_anchor - init_scale * base_center
    x0 = np.concatenate(([math.log(init_scale)], init_rotvec, init_trans))

    result = least_squares(
        build_residual,
        x0,
        loss="soft_l1",
        f_scale=0.25,
        args=(frame_keys, frame_points, cams, boxes, first_anchor, first_foot_anchor, up_world, args.foot_weight, args.foot_sigma),
        max_nfev=300,
        verbose=0,
    )

    log_scale = float(result.x[0])
    rotvec = result.x[1:4]
    trans = result.x[4:7]
    scale = float(np.exp(log_scale))
    rot = Rotation.from_rotvec(rotvec)

    summary = {
        "selected_frames": frame_keys,
        "selected_views": sorted(views),
        "initial_depth_hint": depth_hint,
        "first_foot_anchor": first_foot_anchor.tolist(),
        "estimated_world_up": up_world.tolist(),
        "foot_weight": args.foot_weight,
        "foot_sigma": args.foot_sigma,
        "initial_scale": init_scale,
        "optimized_scale": scale,
        "optimized_rotvec": rotvec.tolist(),
        "optimized_quat_xyzw": rot.as_quat().tolist(),
        "optimized_translation": trans.tolist(),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": result.message,
    }
    (output_dir / "transform.json").write_text(json.dumps(summary, indent=2))

    centers = []
    overlay_paths: list[Path] = []
    for frame_key in frame_keys:
        view, frame_idx = frame_key
        aligned = transform_points(frame_points[frame_idx], log_scale, rotvec, trans)
        centers.append(aligned.mean(axis=0).tolist())
        uv, valid = project_points(aligned, cams[frame_key])
        pred = robust_bbox(uv, valid, cams[frame_key])
        image_path = image_dirs[view] / f"frame_{frame_idx:06d}.jpg"
        overlay_path = output_dir / "overlays" / f"{view}_frame_{frame_idx:06d}.png"
        label = f"v={view} f={frame_idx} s={scale:.3f}"
        make_overlay(image_path, boxes[frame_key], pred, overlay_path, label)
        overlay_paths.append(overlay_path)
        if frame_key == frame_keys[0]:
            export_aligned_ply(output_dir / "aligned_dynamic_frame0.ply", aligned)

    (output_dir / "centers_world.json").write_text(json.dumps(centers, indent=2))
    if overlay_paths:
        make_contact_sheet(overlay_paths, output_dir / "overlay_contact_sheet.png")

    source_ply = PlyData.read(str(dynamic_ply))
    vertex = source_ply["vertex"].data
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)
    aligned_static = transform_points(xyz, log_scale, rotvec, trans)
    export_aligned_ply(output_dir / "aligned_dynamic_snapshot_from_ply.ply", aligned_static)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
