from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pointcloud_classify.dem import build_dem
from pointcloud_classify.downsample import voxel_downsample_indices
from pointcloud_classify.features_pca import compute_verticality_knn
from pointcloud_classify.height_norm import compute_height_above_ground
from pointcloud_classify.io_ply import inspect_ply, load_ply, write_ply_with_label
from pointcloud_classify.orientation import choose_up_axis, parse_axis_spec, transform_xyz_for_up_axis
from pointcloud_classify.postprocess import smooth_vote
from pointcloud_classify.propagate import propagate_labels_voxel_hash
from pointcloud_classify.rules import classify_rules


@dataclass
class Config:
    voxel_ds: float = 0.10
    voxel_back: float = 0.10
    grid_res: float = 0.5
    ground_stat: str = "p10"
    fill_holes: bool = True
    h_ground_max: float = 0.30
    pca_k: int = 30
    pca_chunk_size: int = 25000
    h_tree_min: float = 3.0
    h_mid: float = 2.0
    v_thr: float = 0.4
    smooth_k: int = 12
    smooth_support_ratio: float = 0.6
    metrics_k: int = 8
    up_axis: str = "auto"


def load_config(path: Path | None) -> Config:
    cfg = Config()
    if not path:
        return cfg
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def compute_isolated_ratio(xyz: np.ndarray, labels: np.ndarray, k: int) -> float:
    k = max(2, min(int(k), xyz.shape[0]))
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k, workers=-1)
    if k == 1:
        idx = idx[:, None]
    neigh = labels[idx]
    counts = np.stack([(neigh == cls).sum(axis=1) for cls in (0, 1, 2)], axis=1)
    majority = counts.argmax(axis=1).astype(np.uint8)
    return float(np.mean(majority != labels))


def label_stats(labels: np.ndarray) -> dict[str, float]:
    total = labels.shape[0]
    return {
        "p_ground": float((labels == 0).sum() / total),
        "p_shrubs": float((labels == 1).sum() / total),
        "p_trees": float((labels == 2).sum() / total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based large PLY classifier")
    parser.add_argument("--in", dest="input_path", required=True, help="Input PLY")
    parser.add_argument("--out", dest="output_path", required=True, help="Output labeled PLY")
    parser.add_argument("--config", dest="config_path", default=str(ROOT / "config.yaml"))
    parser.add_argument("--log", dest="log_path", default=None, help="Optional log path")
    parser.add_argument("--up-axis", dest="up_axis", default=None, help="Override up-axis: auto, x, y, z, -x, -y, -z")
    args = parser.parse_args()

    cfg = load_config(Path(args.config_path) if args.config_path else None)
    if args.up_axis:
        cfg.up_axis = args.up_axis
    log_path = Path(args.log_path) if args.log_path else ROOT / "logs" / f"run_{int(time.time())}.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    info = inspect_ply(args.input_path)
    ply, vertex, xyz = load_ply(args.input_path)
    timings["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    preview_ds = voxel_downsample_indices(xyz, cfg.voxel_ds)
    preview_ds_xyz = xyz[preview_ds["indices"]]
    timings["downsample"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    axis_scores = None
    if str(cfg.up_axis).lower() == "auto":
        up_axis, axis_scores = choose_up_axis(preview_ds_xyz, cfg.grid_res, cfg.h_ground_max, cfg.ground_stat, cfg.fill_holes)
    else:
        up_axis = parse_axis_spec(str(cfg.up_axis))
    xyz_local = transform_xyz_for_up_axis(xyz, up_axis)
    ds = voxel_downsample_indices(xyz_local, cfg.voxel_ds)
    ds_xyz_local = xyz_local[ds["indices"]]
    timings["orientation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    dem = build_dem(ds_xyz_local, cfg.grid_res, cfg.ground_stat, cfg.fill_holes)
    height_ds = compute_height_above_ground(ds_xyz_local, dem)
    timings["dem_height"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    verticality_ds = compute_verticality_knn(ds_xyz_local, cfg.pca_k, cfg.pca_chunk_size)
    timings["verticality"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    labels_ds = classify_rules(
        height_ds,
        verticality_ds,
        cfg.h_ground_max,
        cfg.h_tree_min,
        cfg.h_mid,
        cfg.v_thr,
    )
    labels_ds = smooth_vote(ds_xyz_local, labels_ds, cfg.smooth_k, cfg.smooth_support_ratio)
    timings["classify_postprocess"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    labels_full, miss_count = propagate_labels_voxel_hash(
        xyz_local, ds_xyz_local, labels_ds, cfg.voxel_back, ds["origin"]
    )
    timings["propagate"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    write_ply_with_label(ply, labels_full, args.output_path)
    timings["write"] = time.perf_counter() - t0

    height_full = compute_height_above_ground(xyz_local, dem)
    stats = label_stats(labels_full)
    isolated_ratio = compute_isolated_ratio(ds_xyz_local, labels_ds, cfg.metrics_k)

    summary = {
        "input": info,
        "config": cfg.__dict__,
        "downsampled_points": int(ds_xyz_local.shape[0]),
        "selected_up_axis": up_axis.name,
        "up_axis_scores": axis_scores,
        "voxel_hash_miss_count": miss_count,
        "height_stats": {
            "all_min": float(np.min(height_full)),
            "all_median": float(np.median(height_full)),
            "all_p95": float(np.percentile(height_full, 95)),
            "ground_median": float(np.median(height_full[labels_full == 0])) if np.any(labels_full == 0) else None,
            "shrubs_p50": float(np.median(height_full[labels_full == 1])) if np.any(labels_full == 1) else None,
            "trees_p50": float(np.median(height_full[labels_full == 2])) if np.any(labels_full == 2) else None,
        },
        "label_stats": stats,
        "isolated_ratio_ds": isolated_ratio,
        "label_values": sorted(np.unique(labels_full).astype(int).tolist()),
        "timings_sec": timings,
        "total_sec": time.perf_counter() - started,
        "output_path": str(args.output_path),
    }

    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, indent=2, ensure_ascii=False))
        fh.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
