import argparse
import os
import pickle
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf


DREAMSCENE_ROOT = Path("/mnt/d/develop/4D/submodules/dreamscene4d")
if str(DREAMSCENE_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMSCENE_ROOT))

from cameras import orbit_camera, OrbitCamera, MiniCam  # noqa: E402
from gs_renderer_4d import Renderer  # noqa: E402


def load_rgb_frame(path: Path, max_side: int = 720) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda()
    h, w = tensor.shape[-2:]
    resize_factor = max_side / max(h, w) if max(h, w) > max_side else 1.0
    h2 = int(h * resize_factor)
    w2 = int(w * resize_factor)
    return F.interpolate(tensor, (h2, w2), mode="bilinear", align_corners=False)


def alpha_blend(base: np.ndarray, fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha3 = np.repeat(alpha[..., None], 3, axis=-1)
    return np.clip(base * (1.0 - alpha3) + fg * alpha3, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--stage2-outdir", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-side", type=int, default=720)
    parser.add_argument("--overlay-alpha", type=float, default=0.6)
    args = parser.parse_args()

    opt = OmegaConf.load(args.config)

    input_dir = Path(args.input_dir)
    stage2_outdir = Path(args.stage2_outdir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {input_dir}")

    model_ply = stage2_outdir / f"{args.save_path}_4d_model.ply"
    motion_pkl = stage2_outdir / "gaussians" / f"{args.save_path}_4d_global_motion.pkl"
    if not model_ply.exists():
        raise FileNotFoundError(model_ply)
    if not motion_pkl.exists():
        raise FileNotFoundError(motion_pkl)

    with open(motion_pkl, "rb") as f:
        motion = pickle.load(f)

    renderer = Renderer(T=len(frame_paths), sh_degree=opt.sh_degree)
    renderer.initialize(str(model_ply))
    renderer.gaussians.load_model(str(stage2_outdir), args.save_path)
    renderer.initialize_global_motion(
        opt,
        translation=motion["translation"],
        scale=motion["scale"],
        base_translation=motion.get("base_translation", torch.zeros(3)),
        base_scale=motion.get("base_scale", torch.ones(1)),
        base_rotation=motion.get("base_rotation", torch.tensor([1.0, 0.0, 0.0, 0.0])),
        base_rotation_center=motion.get("base_rotation_center", torch.zeros(3)),
    )

    overlays = []
    renders = []
    render_cam_world = None

    for t, frame_path in enumerate(frame_paths):
        frame = load_rgb_frame(frame_path, max_side=args.max_side)
        h, w = frame.shape[-2:]
        if render_cam_world is None:
            render_cam_world = OrbitCamera(w, h, r=opt.radius, fovy=opt.fovy)

        pose = orbit_camera(opt.elevation, 0, opt.radius)
        cam = MiniCam(
            pose,
            w,
            h,
            render_cam_world.fovy,
            render_cam_world.fovx,
            render_cam_world.near,
            render_cam_world.far,
            time=t,
        )

        with torch.no_grad():
            out = renderer.render(
                cam,
                direct_render=True,
                account_for_global_motion=True,
            )

        frame_np = frame.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
        render_np = out["image"].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        alpha_np = out["alpha"].detach().cpu().numpy().astype(np.float32).squeeze()
        overlay_np = alpha_blend(frame_np, render_np, alpha_np * args.overlay_alpha)

        render_img = (np.clip(render_np, 0.0, 1.0) * 255).astype(np.uint8)
        overlay_img = (np.clip(overlay_np, 0.0, 1.0) * 255).astype(np.uint8)

        imageio.imwrite(output_dir / f"{t:04d}_render.png", render_img)
        imageio.imwrite(output_dir / f"{t:04d}_overlay.png", overlay_img)
        renders.append(render_img)
        overlays.append(overlay_img)

    imageio.mimsave(output_dir / "render.mp4", renders, fps=6)
    imageio.mimsave(output_dir / "overlay.mp4", overlays, fps=6)
    imageio.mimsave(output_dir / "render.gif", renders, duration=1 / 6, loop=0)
    imageio.mimsave(output_dir / "overlay.gif", overlays, duration=1 / 6, loop=0)

    print(f"[INFO] Saved image-frame render/overlay to {output_dir}")


if __name__ == "__main__":
    main()
