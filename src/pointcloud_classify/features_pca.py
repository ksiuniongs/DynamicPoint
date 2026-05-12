from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def compute_verticality_knn(
    xyz: np.ndarray,
    k: int = 30,
    chunk_size: int = 25000,
) -> np.ndarray:
    n_points = xyz.shape[0]
    if n_points == 0:
        return np.empty(0, dtype=np.float32)
    k = max(3, min(int(k), n_points))
    tree = cKDTree(xyz)
    verticality = np.empty(n_points, dtype=np.float32)

    for start in range(0, n_points, chunk_size):
        stop = min(start + chunk_size, n_points)
        _, nn_idx = tree.query(xyz[start:stop], k=k, workers=-1)
        if k == 1:
            nn_idx = nn_idx[:, None]
        neighborhoods = xyz[nn_idx]
        centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", centered, centered) / max(k - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        max_eig_idx = np.argmax(eigvals, axis=1)[:, None, None]
        principal = np.take_along_axis(eigvecs, max_eig_idx.repeat(3, axis=1), axis=2).squeeze(axis=2)
        verticality[start:stop] = 1.0 - np.abs(principal[:, 2])
    return verticality
