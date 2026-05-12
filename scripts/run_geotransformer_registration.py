#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R


GEOTRANSFORMER_ROOT = Path("/mnt/d/develop/master_thesis/external/GeoTransformer")
GEOTRANSFORMER_EXP = (
    GEOTRANSFORMER_ROOT / "experiments" / "geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_ply", required=True)
    parser.add_argument("--ref_ply", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--src_limit", type=int, default=30000)
    parser.add_argument("--ref_limit", type=int, default=30000)
    return parser.parse_args()


def random_limit(points: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or len(points) <= limit:
        return points
    rng = np.random.default_rng(0)
    idx = rng.choice(len(points), size=limit, replace=False)
    return points[np.sort(idx)]


def load_points(path: Path, limit: int) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float32)
    return random_limit(points, limit)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    return (transform @ pts_h.T).T[:, :3]


def apply_transform_to_full_gs(src_ply: Path, transform: np.ndarray, out_ply: Path) -> None:
    ply = PlyData.read(str(src_ply))
    vdata = ply["vertex"].data
    new_vertex = vdata.copy()
    field_names = vdata.dtype.names

    xyz = np.stack([vdata["x"], vdata["y"], vdata["z"]], axis=1).astype(np.float64)
    rot_global = transform[:3, :3]
    trans_global = transform[:3, 3]
    xyz_new = (rot_global @ xyz.T).T + trans_global[None, :]
    new_vertex["x"] = xyz_new[:, 0].astype(new_vertex["x"].dtype)
    new_vertex["y"] = xyz_new[:, 1].astype(new_vertex["y"].dtype)
    new_vertex["z"] = xyz_new[:, 2].astype(new_vertex["z"].dtype)

    if all(name in field_names for name in ("nx", "ny", "nz")):
        normals = np.stack([vdata["nx"], vdata["ny"], vdata["nz"]], axis=1).astype(np.float64)
        normals_new = (rot_global @ normals.T).T
        new_vertex["nx"] = normals_new[:, 0].astype(new_vertex["nx"].dtype)
        new_vertex["ny"] = normals_new[:, 1].astype(new_vertex["ny"].dtype)
        new_vertex["nz"] = normals_new[:, 2].astype(new_vertex["nz"].dtype)

    rot_names = ["rot_0", "rot_1", "rot_2", "rot_3"]
    if all(name in field_names for name in rot_names):
        q_global_xyzw = R.from_matrix(rot_global).as_quat()
        q_old_wxyz = np.stack([vdata[name] for name in rot_names], axis=1).astype(np.float64)
        q_old_xyzw = np.stack([q_old_wxyz[:, 1], q_old_wxyz[:, 2], q_old_wxyz[:, 3], q_old_wxyz[:, 0]], axis=1)
        r_new = R.from_quat(q_global_xyzw) * R.from_quat(q_old_xyzw)
        q_new_xyzw = r_new.as_quat()
        q_new_wxyz = np.stack([q_new_xyzw[:, 3], q_new_xyzw[:, 0], q_new_xyzw[:, 1], q_new_xyzw[:, 2]], axis=1)
        for i, name in enumerate(rot_names):
            new_vertex[name] = q_new_wxyz[:, i].astype(new_vertex[name].dtype)

    all_elements = [PlyElement.describe(new_vertex, "vertex")]
    for elem in ply.elements[1:]:
        all_elements.append(elem)
    new_ply = PlyData(
        all_elements,
        text=ply.text,
        byte_order=ply.byte_order,
        comments=ply.comments,
        obj_info=ply.obj_info,
    )
    new_ply.write(str(out_ply))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(GEOTRANSFORMER_ROOT))
    sys.path.insert(0, str(GEOTRANSFORMER_EXP))

    from config import make_cfg  # noqa: WPS433
    from model import create_model  # noqa: WPS433
    from geotransformer.utils.data import registration_collate_fn_stack_mode  # noqa: WPS433
    from geotransformer.utils.torch import to_cuda, release_cuda  # noqa: WPS433

    ref_points = load_points(Path(args.ref_ply), args.ref_limit)
    src_points = load_points(Path(args.src_ply), args.src_limit)
    ref_feats = np.ones((ref_points.shape[0], 1), dtype=np.float32)
    src_feats = np.ones((src_points.shape[0], 1), dtype=np.float32)

    np.save(output_dir / "ref_points.npy", ref_points)
    np.save(output_dir / "src_points.npy", src_points)

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats,
        "src_feats": src_feats,
        "transform": np.eye(4, dtype=np.float32),
    }

    cfg = make_cfg()
    neighbor_limits = [38, 36, 36, 38]
    data_dict = registration_collate_fn_stack_mode(
        [data_dict],
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
    )

    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict["model"])
    model.eval()

    with torch.no_grad():
        data_dict = to_cuda(data_dict)
        output_dict = model(data_dict)
        data_dict = release_cuda(data_dict)
        output_dict = release_cuda(output_dict)

    estimated_transform = output_dict["estimated_transform"]
    np.savez(output_dir / "estimated_transform.npz", estimated_transform=estimated_transform)

    src_aligned = transform_points(src_points, estimated_transform)
    src_aligned_pcd = o3d.geometry.PointCloud()
    src_aligned_pcd.points = o3d.utility.Vector3dVector(src_aligned.astype(np.float64))
    o3d.io.write_point_cloud(str(output_dir / "src_aligned_points.ply"), src_aligned_pcd)

    apply_transform_to_full_gs(Path(args.src_ply), estimated_transform, output_dir / "src_aligned_fullgs.ply")

    summary = {
        "src_ply": args.src_ply,
        "ref_ply": args.ref_ply,
        "weights": args.weights,
        "src_points": int(src_points.shape[0]),
        "ref_points": int(ref_points.shape[0]),
        "estimated_transform": estimated_transform.tolist(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
