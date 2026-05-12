from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def smooth_vote(
    xyz: np.ndarray,
    labels: np.ndarray,
    k: int = 12,
    support_ratio: float = 0.6,
    chunk_size: int = 50000,
) -> np.ndarray:
    if xyz.shape[0] == 0:
        return labels
    k = max(2, min(int(k), xyz.shape[0]))
    tree = cKDTree(xyz)
    smoothed = labels.copy()
    label_space = np.array([0, 1, 2], dtype=np.uint8)

    for start in range(0, xyz.shape[0], chunk_size):
        stop = min(start + chunk_size, xyz.shape[0])
        _, nn_idx = tree.query(xyz[start:stop], k=k, workers=-1)
        if k == 1:
            nn_idx = nn_idx[:, None]
        neigh_labels = labels[nn_idx]
        counts = np.stack([(neigh_labels == cls).sum(axis=1) for cls in label_space], axis=1)
        winner = counts.argmax(axis=1).astype(np.uint8)
        winner_count = counts.max(axis=1)
        update = (winner_count / k) >= support_ratio
        smoothed[start:stop][update] = winner[update]
    return smoothed

