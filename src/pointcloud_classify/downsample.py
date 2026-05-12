from __future__ import annotations

import numpy as np


def quantize_voxels(xyz: np.ndarray, voxel_size: float, origin: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if origin is None:
        origin = xyz.min(axis=0)
    origin = np.asarray(origin, dtype=np.float32)
    keys = np.floor((xyz - origin[None, :]) / voxel_size).astype(np.int32)
    return keys, origin


def pack_keys(keys: np.ndarray) -> np.ndarray:
    packed = np.empty(keys.shape[0], dtype=[("kx", np.int32), ("ky", np.int32), ("kz", np.int32)])
    packed["kx"] = keys[:, 0]
    packed["ky"] = keys[:, 1]
    packed["kz"] = keys[:, 2]
    return packed


def voxel_downsample_indices(xyz: np.ndarray, voxel_size: float) -> dict[str, np.ndarray]:
    keys, origin = quantize_voxels(xyz, voxel_size)
    packed = pack_keys(keys)
    _, unique_indices, inverse, counts = np.unique(
        packed, return_index=True, return_inverse=True, return_counts=True
    )
    order = np.argsort(unique_indices)
    unique_indices = unique_indices[order]
    remap = np.empty_like(order)
    remap[order] = np.arange(order.shape[0])
    inverse = remap[inverse]
    return {
        "indices": unique_indices,
        "inverse": inverse,
        "counts": counts[order],
        "origin": origin,
        "voxel_size": np.array([voxel_size], dtype=np.float32),
    }

