from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dem import build_dem
from .height_norm import compute_height_above_ground


@dataclass(frozen=True)
class AxisSpec:
    axis: int
    sign: int

    @property
    def name(self) -> str:
        return f"{'-' if self.sign < 0 else ''}{'xyz'[self.axis]}"


def parse_axis_spec(value: str) -> AxisSpec:
    value = value.strip().lower()
    sign = -1 if value.startswith("-") else 1
    axis_name = value[1:] if value.startswith("-") else value
    if axis_name not in {"x", "y", "z"}:
        raise ValueError(f"Invalid up-axis '{value}'. Expected one of x, y, z, -x, -y, -z, auto.")
    return AxisSpec("xyz".index(axis_name), sign)


def axis_candidates() -> list[AxisSpec]:
    return [
        AxisSpec(0, 1),
        AxisSpec(0, -1),
        AxisSpec(1, 1),
        AxisSpec(1, -1),
        AxisSpec(2, 1),
        AxisSpec(2, -1),
    ]


def transform_xyz_for_up_axis(xyz: np.ndarray, spec: AxisSpec) -> np.ndarray:
    horiz_axes = [idx for idx in range(3) if idx != spec.axis]
    local = np.empty_like(xyz, dtype=np.float32)
    local[:, 0] = xyz[:, horiz_axes[0]]
    local[:, 1] = xyz[:, horiz_axes[1]]
    local[:, 2] = xyz[:, spec.axis] * spec.sign
    return local


def score_up_axis(
    xyz: np.ndarray,
    spec: AxisSpec,
    grid_res: float,
    h_ground_max: float,
    ground_stat: str = "p10",
    fill_holes: bool = True,
) -> dict[str, float | str]:
    local = transform_xyz_for_up_axis(xyz, spec)
    dem = build_dem(local, grid_res, ground_stat, fill_holes)
    h = compute_height_above_ground(local, dem)
    ground_frac = float(np.mean((h >= -0.20) & (h <= h_ground_max)))
    neg_frac = float(np.mean(h < -0.50))
    p50 = float(np.percentile(h, 50))
    p95 = float(np.percentile(h, 95))
    score = ground_frac + 0.05 * p95 - 2.0 * neg_frac - 0.05 * abs(min(p50, 0.0))
    return {
        "axis": spec.name,
        "score": float(score),
        "ground_frac": ground_frac,
        "neg_frac": neg_frac,
        "height_p50": p50,
        "height_p95": p95,
    }


def choose_up_axis(
    xyz: np.ndarray,
    grid_res: float,
    h_ground_max: float,
    ground_stat: str = "p10",
    fill_holes: bool = True,
) -> tuple[AxisSpec, list[dict[str, float | str]]]:
    results = [
        score_up_axis(xyz, spec, grid_res, h_ground_max, ground_stat, fill_holes)
        for spec in axis_candidates()
    ]
    best = max(results, key=lambda row: float(row["score"]))
    return parse_axis_spec(str(best["axis"])), results
