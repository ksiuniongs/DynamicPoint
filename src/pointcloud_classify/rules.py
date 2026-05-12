from __future__ import annotations

import numpy as np


GROUND = np.uint8(0)
SHRUBS = np.uint8(1)
TREES = np.uint8(2)


def classify_rules(
    height: np.ndarray,
    verticality: np.ndarray,
    h_ground_max: float,
    h_tree_min: float,
    h_mid: float,
    v_thr: float,
) -> np.ndarray:
    labels = np.full(height.shape[0], SHRUBS, dtype=np.uint8)
    ground_mask = height <= h_ground_max
    tree_mask = (~ground_mask) & (
        (height >= h_tree_min) | ((height >= h_mid) & (verticality <= v_thr))
    )
    labels[ground_mask] = GROUND
    labels[tree_mask] = TREES
    return labels

