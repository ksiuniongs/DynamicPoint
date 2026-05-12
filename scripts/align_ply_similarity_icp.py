#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from itertools import permutations, product
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align a source PLY to a target PLY with similarity ICP.")
    parser.add_argument("--source_ply", required=True)
    parser.add_argument("--target_ply", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_limit", type=int, default=5000)
    parser.add_argument("--coarse_candidates", type=int, default=48)
    parser.add_argument("--icp_iters", type=int, default=40)
    parser.add_argument("--trim_ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_vertex(path: Path):
    return PlyData.read(str(path))["vertex"].data


def xyz_from_vertex(vertex) -> np.ndarray:
    return np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)


def write_transformed_vertex(src_path: Path, dst_path: Path, xyz_new: np.ndarray) -> None:
    ply = PlyData.read(str(src_path))
    vertex = ply["vertex"].data.copy()
    vertex["x"] = xyz_new[:, 0].astype(vertex["x"].dtype)
    vertex["y"] = xyz_new[:, 1].astype(vertex["y"].dtype)
    vertex["z"] = xyz_new[:, 2].astype(vertex["z"].dtype)
    PlyData([ply["vertex"].__class__.describe(vertex, "vertex")], text=False).write(str(dst_path))


def subsample(points: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(points) <= limit:
        return points.copy()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(points), size=limit, replace=False))
    return points[idx]


def make_right_handed(mat: np.ndarray) -> np.ndarray:
    out = mat.copy()
    if np.linalg.det(out) < 0:
        out[:, -1] *= -1.0
    return out


def pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    centered = points - center[None, :]
    cov = centered.T @ centered / max(len(points) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    return center, make_right_handed(evecs)


def radius_scale(points: np.ndarray) -> float:
    centered = points - points.mean(axis=0, keepdims=True)
    radii = np.linalg.norm(centered, axis=1)
    return float(np.percentile(radii, 90))


def apply_similarity(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return scale * (points @ rot.T) + trans[None, :]


def umeyama_similarity(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = (dst_c.T @ src_c) / max(len(src), 1)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var = np.mean(np.sum(src_c * src_c, axis=1))
        scale = float(np.trace(np.diag(D) @ S) / max(var, 1e-12))
    else:
        scale = 1.0
    t = dst_mean - scale * (R @ src_mean)
    return scale, R, t


def nn_rmse(a: np.ndarray, b: np.ndarray, trim_ratio: float) -> tuple[float, np.ndarray]:
    tree = cKDTree(b)
    dists, idx = tree.query(a, k=1, workers=-1)
    keep = max(16, int(math.ceil(len(dists) * trim_ratio)))
    order = np.argsort(dists)[:keep]
    return float(np.sqrt(np.mean(dists[order] ** 2))), idx[order]


def candidate_rotations(src_axes: np.ndarray, dst_axes: np.ndarray):
    count = 0
    for perm in permutations(range(3)):
        P = np.eye(3)[:, perm]
        for signs in product([-1.0, 1.0], repeat=3):
            S = np.diag(signs)
            R = dst_axes @ P @ S @ src_axes.T
            if np.linalg.det(R) <= 0:
                continue
            yield R
            count += 1


def coarse_initialize(src: np.ndarray, dst: np.ndarray, max_candidates: int, trim_ratio: float) -> tuple[float, np.ndarray, np.ndarray, float]:
    src_center, src_axes = pca_frame(src)
    dst_center, dst_axes = pca_frame(dst)
    scale0 = radius_scale(dst) / max(radius_scale(src), 1e-12)

    best = None
    for i, R in enumerate(candidate_rotations(src_axes, dst_axes)):
        if i >= max_candidates:
            break
        t = dst_center - scale0 * (R @ src_center)
        transformed = apply_similarity(src, scale0, R, t)
        rmse, _ = nn_rmse(transformed, dst, trim_ratio)
        if best is None or rmse < best[0]:
            best = (rmse, scale0, R, t)
    assert best is not None
    return best[1], best[2], best[3], best[0]


def refine_similarity_icp(src: np.ndarray, dst: np.ndarray, init_scale: float, init_rot: np.ndarray, init_trans: np.ndarray, trim_ratio: float, iters: int) -> tuple[float, np.ndarray, np.ndarray, list[dict]]:
    scale = init_scale
    rot = init_rot.copy()
    trans = init_trans.copy()
    history: list[dict] = []
    tree = cKDTree(dst)

    for i in range(iters):
        transformed = apply_similarity(src, scale, rot, trans)
        dists, idx = tree.query(transformed, k=1, workers=-1)
        keep = max(16, int(math.ceil(len(dists) * trim_ratio)))
        order = np.argsort(dists)[:keep]
        matched_src = src[order]
        matched_dst = dst[idx[order]]
        new_scale, new_rot, new_trans = umeyama_similarity(matched_src, matched_dst, with_scale=True)

        transformed_new = apply_similarity(src, new_scale, new_rot, new_trans)
        rmse, _ = nn_rmse(transformed_new, dst, trim_ratio)
        history.append(
            {
                "iter": i,
                "rmse": rmse,
                "scale": float(new_scale),
                "translation": new_trans.tolist(),
                "rotation_matrix": new_rot.tolist(),
            }
        )

        if abs(new_scale - scale) < 1e-7 and np.linalg.norm(new_trans - trans) < 1e-7 and np.linalg.norm(new_rot - rot) < 1e-7:
            scale, rot, trans = new_scale, new_rot, new_trans
            break
        scale, rot, trans = new_scale, new_rot, new_trans

    return scale, rot, trans, history


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_path = Path(args.source_ply)
    dst_path = Path(args.target_ply)

    src_vertex = load_vertex(src_path)
    dst_vertex = load_vertex(dst_path)
    src_xyz = xyz_from_vertex(src_vertex)
    dst_xyz = xyz_from_vertex(dst_vertex)

    src_sample = subsample(src_xyz, args.sample_limit, args.seed)
    dst_sample = subsample(dst_xyz, args.sample_limit, args.seed + 1)

    init_scale, init_rot, init_trans, init_rmse = coarse_initialize(
        src_sample, dst_sample, args.coarse_candidates, args.trim_ratio
    )
    final_scale, final_rot, final_trans, history = refine_similarity_icp(
        src_sample,
        dst_sample,
        init_scale,
        init_rot,
        init_trans,
        args.trim_ratio,
        args.icp_iters,
    )

    aligned_full = apply_similarity(src_xyz, final_scale, final_rot, final_trans)
    aligned_path = out_dir / f"{src_path.stem}_aligned_to_{dst_path.stem}.ply"
    write_transformed_vertex(src_path, aligned_path, aligned_full)

    final_rmse, _ = nn_rmse(aligned_full, dst_xyz, args.trim_ratio)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = final_scale * final_rot
    transform[:3, 3] = final_trans

    manifest = {
        "source_ply": str(src_path),
        "target_ply": str(dst_path),
        "source_points": int(len(src_xyz)),
        "target_points": int(len(dst_xyz)),
        "sample_limit": args.sample_limit,
        "trim_ratio": args.trim_ratio,
        "coarse_init_rmse": init_rmse,
        "final_rmse": final_rmse,
        "scale": float(final_scale),
        "rotation_matrix": final_rot.tolist(),
        "translation": final_trans.tolist(),
        "transform_matrix": transform.tolist(),
        "aligned_ply": str(aligned_path),
        "history": history,
    }

    (out_dir / "alignment_manifest.json").write_text(json.dumps(manifest, indent=2))
    np.savez_compressed(
        out_dir / "transform.npz",
        scale=np.array([final_scale]),
        rotation=final_rot,
        translation=final_trans,
        transform=transform,
    )

    print(aligned_path)
    print(out_dir / "alignment_manifest.json")


if __name__ == "__main__":
    main()
