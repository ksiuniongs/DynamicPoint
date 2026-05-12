from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .downsample import pack_keys, quantize_voxels


def propagate_labels_voxel_hash(
    xyz_full: np.ndarray,
    xyz_ds: np.ndarray,
    labels_ds: np.ndarray,
    voxel_size: float,
    origin: np.ndarray,
) -> tuple[np.ndarray, int]:
    ds_keys = pack_keys(quantize_voxels(xyz_ds, voxel_size, origin)[0])
    full_keys = pack_keys(quantize_voxels(xyz_full, voxel_size, origin)[0])

    order = np.argsort(ds_keys, axis=0, order=("kx", "ky", "kz"))
    sorted_keys = ds_keys[order]

    idx = np.searchsorted(sorted_keys, full_keys)
    valid = idx < sorted_keys.shape[0]
    matched = np.zeros(xyz_full.shape[0], dtype=bool)
    matched[valid] = sorted_keys[idx[valid]] == full_keys[valid]

    labels = np.empty(xyz_full.shape[0], dtype=np.uint8)
    labels[matched] = labels_ds[order[idx[matched]]]

    miss_count = int((~matched).sum())
    if miss_count:
        tree = cKDTree(xyz_ds)
        _, nn_idx = tree.query(xyz_full[~matched], k=1, workers=-1)
        labels[~matched] = labels_ds[nn_idx]
    return labels, miss_count

