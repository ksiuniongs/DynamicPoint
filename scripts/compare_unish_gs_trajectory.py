#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = (dst_centered.T @ src_centered) / src.shape[0]
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1.0
    rot = u @ s @ vt
    src_var = np.mean(np.sum(src_centered**2, axis=1))
    scale = np.trace(np.diag(d) @ s) / max(src_var, 1e-12)
    trans = dst_mean - scale * (rot @ src_mean)
    return float(scale), rot, trans


def load_unish_centers(camera_npz: Path) -> Dict[str, np.ndarray]:
    d = np.load(camera_npz)
    extrinsics = d["extrinsics"].astype(np.float64)
    centers = {}
    for idx, w2c in enumerate(extrinsics):
        r = w2c[:3, :3]
        t = w2c[:3, 3]
        center = -r.T @ t
        centers[f"frame_{idx:06d}"] = center
    return centers


def load_gs_centers(cameras_json: Path) -> Dict[str, np.ndarray]:
    with open(cameras_json, "r", encoding="utf-8") as f:
        items = json.load(f)
    centers = {}
    for item in items:
        name = Path(item["img_name"]).stem
        centers[name] = np.asarray(item["position"], dtype=np.float64)
    return centers


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare GS camera trajectory against UniSH cameras.")
    ap.add_argument("--camera_npz", required=True)
    ap.add_argument("--cameras_json", required=True)
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()

    unish = load_unish_centers(Path(args.camera_npz))
    gs = load_gs_centers(Path(args.cameras_json))
    common = sorted(set(unish) & set(gs))
    if not common:
        raise ValueError("No shared camera/image names between UniSH and GS cameras.")

    unish_pts = np.stack([unish[k] for k in common], axis=0)
    gs_pts = np.stack([gs[k] for k in common], axis=0)
    scale, rot, trans = umeyama(gs_pts, unish_pts)
    aligned = (scale * (rot @ gs_pts.T)).T + trans[None, :]
    errors = np.linalg.norm(aligned - unish_pts, axis=1)

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rot
    matrix[:3, 3] = trans
    result = {
        "num_common_cameras": len(common),
        "scale": scale,
        "rotation": rot.tolist(),
        "translation": trans.tolist(),
        "sim3_matrix": matrix.tolist(),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "max_error": float(errors.max()),
        "mean_error": float(errors.mean()),
    }

    print(json.dumps(result, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
