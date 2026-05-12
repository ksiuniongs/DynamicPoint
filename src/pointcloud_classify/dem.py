from __future__ import annotations

import numpy as np


def _fill_holes(grid: np.ndarray, max_iter: int = 8) -> np.ndarray:
    filled = grid.copy()
    for _ in range(max_iter):
        missing = np.isnan(filled)
        if not missing.any():
            break
        acc = np.zeros_like(filled, dtype=np.float32)
        cnt = np.zeros_like(filled, dtype=np.int32)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                shifted = np.roll(np.roll(filled, dx, axis=0), dy, axis=1)
                valid = ~np.isnan(shifted)
                acc[missing & valid] += shifted[missing & valid]
                cnt[missing & valid] += 1
        update = missing & (cnt > 0)
        filled[update] = acc[update] / cnt[update]
    if np.isnan(filled).any():
        global_default = np.nanmedian(filled)
        if np.isnan(global_default):
            global_default = 0.0
        filled[np.isnan(filled)] = global_default
    return filled


def build_dem(
    xyz: np.ndarray,
    grid_res: float,
    ground_stat: str = "p10",
    fill_holes: bool = True,
) -> dict[str, np.ndarray | float]:
    xy = xyz[:, :2]
    z = xyz[:, 2]
    origin = xy.min(axis=0)
    ij = np.floor((xy - origin[None, :]) / grid_res).astype(np.int32)
    ix, iy = ij[:, 0], ij[:, 1]
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1

    packed = np.empty(ix.shape[0], dtype=[("ix", np.int32), ("iy", np.int32)])
    packed["ix"] = ix
    packed["iy"] = iy
    order = np.argsort(packed, order=("ix", "iy"))
    packed_sorted = packed[order]
    z_sorted = z[order]
    uniq, first_idx, counts = np.unique(packed_sorted, return_index=True, return_counts=True)

    grid = np.full((nx, ny), np.nan, dtype=np.float32)
    for key, start, count in zip(uniq, first_idx, counts):
        vals = z_sorted[start : start + count]
        if ground_stat == "min":
            ground_z = float(vals.min())
        else:
            ground_z = float(np.percentile(vals, 10))
        grid[int(key["ix"]), int(key["iy"])] = ground_z

    if fill_holes:
        grid = _fill_holes(grid)

    return {"grid": grid, "origin": origin.astype(np.float32), "grid_res": float(grid_res)}


def query_z_ground(xy: np.ndarray, dem: dict[str, np.ndarray | float]) -> np.ndarray:
    grid = dem["grid"]
    origin = dem["origin"]
    grid_res = float(dem["grid_res"])
    ij = np.floor((xy - origin[None, :]) / grid_res).astype(np.int32)
    ij[:, 0] = np.clip(ij[:, 0], 0, grid.shape[0] - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, grid.shape[1] - 1)
    return grid[ij[:, 0], ij[:, 1]]

