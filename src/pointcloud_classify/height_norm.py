from __future__ import annotations

import numpy as np

from .dem import query_z_ground


def compute_height_above_ground(xyz: np.ndarray, dem: dict[str, np.ndarray | float]) -> np.ndarray:
    ground = query_z_ground(xyz[:, :2], dem)
    return xyz[:, 2] - ground

