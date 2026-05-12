#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from PIL import Image
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

ROOT = Path("/mnt/d/develop/4D/submodules/dreamscene4d")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cameras import MiniCam, orbit_camera  # noqa: E402
from gs_renderer import Renderer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GS-only mask alignment under real cameras from camera_parameters.npz")
    parser.add_argument("--config", default=str(ROOT / "configs" / "image.yaml"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--input_mask", required=True)
    parser.add_argument("--camera_npz", required=True)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--external_ply", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--visdir", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--stop_iou_threshold", type=float, default=0.0)
    parser.add_argument("--coarse_rotation_degrees", default="0,90,180,270")
    parser.add_argument("--mask_only", action="store_true")
    parser.add_argument("--init_pose_pkl")
    parser.add_argument("--init_old_radius", type=float, default=2.0)
    parser.add_argument("--init_old_fovy", type=float, default=49.1)
    parser.add_argument("--init_old_elevation", type=float, default=0.0)
    parser.add_argument("--init_old_azimuth", type=float, default=0.0)
    parser.add_argument("--save_every_step", action="store_true")
    return parser.parse_args()


def infer_native_image_size(intrinsics: np.ndarray) -> tuple[int, int]:
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    return max(int(round(cx * 2.0)), 1), max(int(round(cy * 2.0)), 1)


def rescale_intrinsics(intrinsics: np.ndarray, src_width: int, src_height: int, dst_width: int, dst_height: int) -> np.ndarray:
    K = intrinsics.copy().astype(np.float64)
    sx = float(dst_width) / float(max(src_width, 1))
    sy = float(dst_height) / float(max(src_height, 1))
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K


class RealMiniCam:
    def __init__(self, w2c: np.ndarray, width: int, height: int, K: np.ndarray, znear: float = 0.01, zfar: float = 100.0):
        self.image_width = width
        self.image_height = height
        self.znear = znear
        self.zfar = zfar

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        self.FoVx = float(2.0 * math.atan(width / (2.0 * max(fx, 1e-8))))
        self.FoVy = float(2.0 * math.atan(height / (2.0 * max(fy, 1e-8))))

        c2w = np.linalg.inv(w2c)
        self._c2w = c2w
        self.world_view_transform = torch.tensor(w2c, dtype=torch.float32).transpose(0, 1).cuda()

        tan_half_x = width / (2.0 * max(fx, 1e-8))
        tan_half_y = height / (2.0 * max(fy, 1e-8))
        P = torch.zeros((4, 4), dtype=torch.float32, device="cuda")
        P[0, 0] = 1.0 / max(tan_half_x, 1e-8)
        P[1, 1] = 1.0 / max(tan_half_y, 1e-8)
        P[2, 2] = zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        P[3, 2] = 1.0
        self.projection_matrix = P.transpose(0, 1)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = torch.tensor(c2w[:3, 3], dtype=torch.float32, device="cuda")


class RealCameraGSAligner:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda")
        self.renderer = Renderer(sh_degree=int(opt.sh_degree))
        self.external_ply = opt.external_ply
        self.train_steps = 1
        self.step = 0

        self.input_img = None
        self.input_mask = None
        self.input_scale = None

        self.load_input()
        self.prepare_inputs()
        self.cam = self.load_real_camera()
        self.renderer.initialize(self.external_ply)
        self.rotation_center = self.renderer.gaussians.get_xyz.detach().mean(dim=0).cpu().tolist()
        self.step_dir = Path(self.opt.visdir) / "steps"
        if self.opt.save_every_step:
            self.step_dir.mkdir(parents=True, exist_ok=True)

    def load_input(self) -> None:
        img = np.array(Image.open(self.opt.input).convert("RGB")).astype(np.float32) / 255.0
        mask = np.array(Image.open(self.opt.input_mask).convert("L")).astype(np.float32) / 255.0
        mask = mask[..., None]
        self.input_mask = mask
        self.input_img = img * mask + (1.0 - mask)

    def prepare_inputs(self) -> None:
        self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)

        _, _, H, W = self.input_mask_torch.shape
        mask = self.input_mask_torch > 0.5
        nonzero = torch.nonzero(mask[0, 0])
        if len(nonzero) == 0:
            raise RuntimeError("Input mask is empty.")
        min_x = nonzero[:, 1].min()
        max_x = nonzero[:, 1].max()
        min_y = nonzero[:, 0].min()
        max_y = nonzero[:, 0].max()
        width = (max_x - min_x) / W
        height = (max_y - min_y) / H
        self.obj_cx = ((max_x + min_x) / 2 / W) * 2 - 1
        self.obj_cy = ((max_y + min_y) / 2 / H) * 2 - 1
        self.input_scale = max(width / 0.65, height / 0.65)

    def load_real_camera(self) -> RealMiniCam:
        obj = np.load(self.opt.camera_npz)
        K_native = obj["intrinsics"][self.opt.frame_idx].astype(np.float64)
        w2c = obj["extrinsics"][self.opt.frame_idx].astype(np.float64)
        H, W = self.input_img.shape[:2]
        native_w, native_h = infer_native_image_size(K_native)
        K = rescale_intrinsics(K_native, native_w, native_h, W, H)
        self.camera_info = {
            "native_width": native_w,
            "native_height": native_h,
            "target_width": W,
            "target_height": H,
            "K_native": K_native.tolist(),
            "K_scaled": K.tolist(),
        }
        return RealMiniCam(w2c=w2c, width=W, height=H, K=K)

    def balanced_mask_loss(self, pred, target, mask):
        masked = (F.mse_loss(pred, target, reduction="none") * mask).sum() / max(mask.sum(), 1)
        masked_empty = (F.mse_loss(pred, target, reduction="none") * (1 - mask)).sum() / max((1 - mask).sum(), 1)
        return masked + masked_empty

    def _save_step_overlay(self, step_idx: int, iou_value: float) -> None:
        out = self.renderer.render(self.cam, account_for_global_motion=True)
        target_mask = (self.input_mask_torch.squeeze(0).squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)
        pred_mask = np.squeeze((out["alpha"].detach().cpu().numpy() > 0.5).astype(np.uint8))
        overlay = np.clip(self.input_img.copy(), 0.0, 1.0)
        t_edge = target_mask.astype(bool)
        p_edge = pred_mask.astype(bool)
        overlay[t_edge] = 0.65 * overlay[t_edge] + 0.35 * np.array([0.0, 1.0, 0.0], dtype=np.float32)
        overlay[p_edge] = 0.65 * overlay[p_edge] + 0.35 * np.array([1.0, 0.0, 0.0], dtype=np.float32)
        overlay[t_edge & p_edge] = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        Image.fromarray(np.uint8(overlay * 255)).save(self.step_dir / f"step_{step_idx:04d}_iou_{iou_value:.4f}.png")

    def _quat_wxyz_from_euler(self, xyz_deg):
        q_xyzw = R.from_euler("xyz", xyz_deg, degrees=True).as_quat()
        return [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]

    @torch.no_grad()
    def coarse_search_rotation(self, translation, scale):
        target_mask = (self.input_mask_torch > 0.5).float()
        angle_grid = [float(x) for x in self.opt.coarse_rotation_degrees.split(",") if x.strip()]
        best_score = -1.0
        best_rotation = [1.0, 0.0, 0.0, 0.0]
        for rx in angle_grid:
            for ry in angle_grid:
                for rz in angle_grid:
                    rotation = self._quat_wxyz_from_euler([rx, ry, rz])
                    self.renderer.initialize_global_motion(
                        self.opt,
                        translation=translation,
                        scale=scale,
                        rotation=rotation,
                        rotation_center=self.rotation_center,
                        optimize_translation=False,
                        optimize_scale=False,
                        optimize_rotation=False,
                    )
                    out = self.renderer.render(self.cam, account_for_global_motion=True)
                    pred_mask = (out["alpha"].unsqueeze(0) > 0.5).float()
                    inter = (pred_mask * target_mask).sum()
                    union = ((pred_mask + target_mask) > 0).float().sum().clamp(min=1.0)
                    iou = (inter / union).item()
                    if iou > best_score:
                        best_score = iou
                        best_rotation = rotation
        return best_rotation, best_score

    def optimize_global_motion(self):
        for _ in range(self.train_steps):
            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)
            self.renderer.update_learning_rate(self.step)
            out = self.renderer.render(self.cam, account_for_global_motion=True)
            target_mask = (self.input_mask_torch > 0.5).float()
            mask = out["alpha"].unsqueeze(0)
            loss = 100 * step_ratio * self.balanced_mask_loss(mask, target_mask, target_mask)
            if not self.opt.mask_only:
                target_img = self.input_img_torch
                image = out["image"].unsqueeze(0)
                loss = loss + 1000 * step_ratio * self.balanced_mask_loss(image, target_img, target_mask)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def current_mask_iou(self) -> float:
        out = self.renderer.render(self.cam, account_for_global_motion=True)
        pred_mask = (out["alpha"].unsqueeze(0) > 0.5).float()
        target_mask = (self.input_mask_torch > 0.5).float()
        inter = (pred_mask * target_mask).sum()
        union = ((pred_mask + target_mask) > 0).float().sum().clamp(min=1.0)
        return float((inter / union).item())

    @torch.no_grad()
    def save_debug(self, prefix: Path):
        out = self.renderer.render(self.cam, account_for_global_motion=True)
        target_mask = (self.input_mask_torch.squeeze(0).squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)
        pred_mask = np.squeeze((out["alpha"].detach().cpu().numpy() > 0.5).astype(np.uint8))
        compare = np.concatenate([(target_mask * 255).astype(np.uint8), (pred_mask * 255).astype(np.uint8)], axis=1)
        Image.fromarray(compare).save(str(prefix) + "_mask_compare.png")
        overlay = np.clip(self.input_img.copy(), 0.0, 1.0)
        t_edge = target_mask.astype(bool)
        p_edge = pred_mask.astype(bool)
        overlay[t_edge] = 0.65 * overlay[t_edge] + 0.35 * np.array([0.0, 1.0, 0.0], dtype=np.float32)
        overlay[p_edge] = 0.65 * overlay[p_edge] + 0.35 * np.array([1.0, 0.0, 0.0], dtype=np.float32)
        overlay[t_edge & p_edge] = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        Image.fromarray(np.uint8(overlay * 255)).save(str(prefix) + "_mask_overlay.png")

    def save_pose(self, best_iou: float) -> None:
        gaussians_dir = Path(self.opt.outdir) / "gaussians"
        gaussians_dir.mkdir(parents=True, exist_ok=True)
        pose_dict = {
            "translation": self.renderer.gaussian_translation.detach().cpu(),
            "scale": self.renderer.gaussian_scale.detach().cpu(),
            "rotation": torch.nn.functional.normalize(self.renderer.gaussian_rotation.detach(), dim=-1).cpu(),
            "rotation_center": self.renderer.gaussian_rotation_center.detach().cpu(),
        }
        with (gaussians_dir / f"{self.opt.save_path}_calibrated_pose.pkl").open("wb") as f:
            pickle.dump(pose_dict, f)
        with (gaussians_dir / f"{self.opt.save_path}_global_motion.pkl").open("wb") as f:
            pickle.dump(pose_dict, f)
        summary = {
            "frame_idx": self.opt.frame_idx,
            "mask_only": bool(self.opt.mask_only),
            "camera_info": self.camera_info,
            "best_mask_iou": best_iou,
            "translation": pose_dict["translation"].tolist(),
            "scale": pose_dict["scale"].tolist(),
            "rotation_wxyz": pose_dict["rotation"].tolist(),
            "rotation_center": pose_dict["rotation_center"].tolist(),
        }
        (Path(self.opt.outdir) / "transform.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _rectified_old_w2c(self) -> np.ndarray:
        c2w_old = orbit_camera(
            self.opt.init_old_elevation,
            self.opt.init_old_azimuth,
            radius=self.opt.init_old_radius,
        )
        w2c_old = np.linalg.inv(c2w_old)
        w2c_old[1:3, :3] *= -1.0
        w2c_old[:3, 3] *= -1.0
        return w2c_old

    def _pose_dict_to_matrix(self, pose_dict: dict) -> np.ndarray:
        def _to_numpy(value) -> np.ndarray:
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=np.float64)

        scale = float(_to_numpy(pose_dict["scale"]).reshape(-1)[0])
        quat = _to_numpy(pose_dict["rotation"]).reshape(-1)
        center = _to_numpy(pose_dict["rotation_center"]).reshape(3)
        translation = _to_numpy(pose_dict["translation"]).reshape(3)
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        A = scale * rot
        b = center + translation - A @ center
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = A
        T[:3, 3] = b
        return T

    def _matrix_to_init_params(self, T: np.ndarray) -> tuple[list[float], float, list[float]]:
        linear = T[:3, :3]
        b = T[:3, 3]
        scale = float(np.cbrt(max(np.linalg.det(linear), 1e-12)))
        rot = linear / max(scale, 1e-12)
        quat_xyzw = R.from_matrix(rot).as_quat()
        quat_wxyz = [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]
        center = np.asarray(self.rotation_center, dtype=np.float64)
        translation = b - center + linear @ center
        return translation.tolist(), scale, quat_wxyz

    def init_from_old_pose(self) -> tuple[list[float], float, list[float]] | None:
        if not self.opt.init_pose_pkl:
            return None
        with open(self.opt.init_pose_pkl, "rb") as f:
            pose_dict = pickle.load(f)
        T_old = self._pose_dict_to_matrix(pose_dict)
        W_old = self._rectified_old_w2c()
        W_real = np.linalg.inv(self.cam._c2w)
        T_real = np.linalg.inv(W_real) @ W_old @ T_old
        return self._matrix_to_init_params(T_real)

    def train(self):
        Path(self.opt.outdir).mkdir(parents=True, exist_ok=True)
        Path(self.opt.visdir).mkdir(parents=True, exist_ok=True)
        target_model_path = Path(self.opt.outdir) / f"{self.opt.save_path}_model.ply"
        shutil.copy2(self.external_ply, target_model_path)

        self.renderer.freeze_gaussians()
        init_params = self.init_from_old_pose()
        if init_params is not None:
            translation, init_scale, rotation = init_params
            coarse_iou = self.current_mask_iou_from_params(translation, init_scale, rotation)
        else:
            xyz = self.renderer.gaussians.get_xyz.detach()
            xyz_cam = (torch.tensor(self.cam._c2w, dtype=torch.float32, device="cuda").inverse()[:3, :3] @ xyz.T).T
            xyz_cam = xyz_cam + torch.tensor(np.linalg.inv(self.cam._c2w)[:3, 3], dtype=torch.float32, device="cuda")
            median_z = torch.median(xyz_cam[:, 2]).detach()
            fx = float(self.camera_info["K_scaled"][0][0])
            fy = float(self.camera_info["K_scaled"][1][1])
            cx = float(self.camera_info["K_scaled"][0][2])
            cy = float(self.camera_info["K_scaled"][1][2])
            render_h, render_w = self.input_img.shape[:2]
            x_scale = float((median_z + self.cam.znear) / max(fx, 1e-8))
            y_scale = float((median_z + self.cam.znear) / max(fy, 1e-8))
            translation = [float((cx - render_w / 2.0) * x_scale), float((render_h / 2.0 - cy) * y_scale), 0.0]
            init_scale = float(self.input_scale)
            rotation, coarse_iou = self.coarse_search_rotation(translation, init_scale)

        self.renderer.initialize_global_motion(
            self.opt,
            translation=translation,
            scale=float(init_scale),
            rotation=rotation,
            rotation_center=self.rotation_center,
            optimize_translation=True,
            optimize_scale=True,
            optimize_rotation=True,
        )
        self.optimizer = self.renderer.global_motion_optimizer
        best_iou = -1.0
        best_state = None
        progress = tqdm.trange(self.opt.iters)
        for _ in progress:
            self.optimize_global_motion()
            current_iou = self.current_mask_iou()
            progress.set_postfix({"iou": f"{current_iou:.4f}"})
            if self.opt.save_every_step:
                self._save_step_overlay(self.step, current_iou)
            if current_iou > best_iou:
                best_iou = current_iou
                best_state = {
                    "translation": self.renderer.gaussian_translation.detach().clone(),
                    "scale": self.renderer.gaussian_scale.detach().clone(),
                    "rotation": self.renderer.gaussian_rotation.detach().clone(),
                }
            if self.opt.stop_iou_threshold > 0 and current_iou >= self.opt.stop_iou_threshold:
                break

        if best_state is not None:
            self.renderer.gaussian_translation = torch.nn.Parameter(best_state["translation"].requires_grad_(True))
            self.renderer.gaussian_scale = torch.nn.Parameter(best_state["scale"].requires_grad_(True))
            self.renderer.gaussian_rotation = torch.nn.Parameter(best_state["rotation"].requires_grad_(True))

        self.save_pose(best_iou)
        self.save_debug(Path(self.opt.visdir) / self.opt.save_path)
        print(json.dumps({"coarse_rotation_iou": coarse_iou, "best_mask_iou": best_iou}, indent=2))

    @torch.no_grad()
    def current_mask_iou_from_params(self, translation, scale, rotation) -> float:
        self.renderer.initialize_global_motion(
            self.opt,
            translation=translation,
            scale=scale,
            rotation=rotation,
            rotation_center=self.rotation_center,
            optimize_translation=False,
            optimize_scale=False,
            optimize_rotation=False,
        )
        return self.current_mask_iou()


def main() -> None:
    args = parse_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.create(vars(args)))
    aligner = RealCameraGSAligner(opt)
    aligner.train()


if __name__ == "__main__":
    main()
