from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData


LABEL_NAMES = {0: "Ground", 1: "Shrubs", 2: "Trees"}
LABEL_COLORS = np.array(
    [
        [166, 124, 82],
        [109, 179, 63],
        [35, 97, 44],
    ],
    dtype=np.float32,
) / 255.0


@dataclass
class Region:
    name: str
    title: str
    x0: float
    y0: float
    window: float
    count: int
    ratios: tuple[float, float, float]
    z_median: float
    z_p95: float


def load_labeled_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertex = PlyData.read(str(path), mmap=True)["vertex"].data
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32, copy=False)
    if {"red", "green", "blue"}.issubset(vertex.dtype.names or ()):
        rgb = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]).astype(np.float32) / 255.0
    else:
        rgb = np.full((xyz.shape[0], 3), 0.7, dtype=np.float32)
    labels = vertex["label"].astype(np.uint8, copy=False)
    return xyz, rgb, labels


def score_candidates(xyz: np.ndarray, labels: np.ndarray, window: float = 2.0, step: float = 1.0) -> list[Region]:
    xmin, ymin = xyz[:, :2].min(axis=0)
    xmax, ymax = xyz[:, :2].max(axis=0)
    xs = np.arange(xmin, xmax - window + 1e-6, step)
    ys = np.arange(ymin, ymax - window + 1e-6, step)
    regions: list[Region] = []
    for x0 in xs:
        for y0 in ys:
            mask = (xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + window) & (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + window)
            count = int(mask.sum())
            if count < 5000:
                continue
            subset_labels = labels[mask]
            ratios = tuple((np.bincount(subset_labels, minlength=3) / count).tolist())
            z = xyz[mask, 2]
            regions.append(
                Region(
                    name="candidate",
                    title="candidate",
                    x0=float(x0),
                    y0=float(y0),
                    window=float(window),
                    count=count,
                    ratios=ratios,
                    z_median=float(np.median(z)),
                    z_p95=float(np.percentile(z, 95)),
                )
            )
    return regions


def select_regions(candidates: list[Region]) -> list[Region]:
    if not candidates:
        raise ValueError("No candidate regions found for visualization.")

    def pick(score_fn, name: str, title: str, chosen: list[Region]) -> Region:
        best = None
        best_score = None
        for region in candidates:
            if any(abs(region.x0 - prev.x0) < 1.0 and abs(region.y0 - prev.y0) < 1.0 for prev in chosen):
                continue
            score = score_fn(region)
            if best is None or score > best_score:
                best = region
                best_score = score
        assert best is not None
        return Region(name, title, best.x0, best.y0, best.window, best.count, best.ratios, best.z_median, best.z_p95)

    chosen: list[Region] = []
    chosen.append(
        pick(
            lambda r: r.ratios[2] * 3.0 + r.z_p95 * 0.1 - r.ratios[0],
            "A",
            "A Dense Trees",
            chosen,
        )
    )
    chosen.append(
        pick(
            lambda r: -abs(r.ratios[2] - 0.35) - abs(r.ratios[1] - 0.45) - abs(r.ratios[0] - 0.20),
            "B",
            "B Sparse Trees + Shrubs",
            chosen,
        )
    )
    chosen.append(
        pick(
            lambda r: r.ratios[0] * 3.0 - r.z_p95 * 0.1 - r.ratios[2],
            "C",
            "C Ground / Rock-like",
            chosen,
        )
    )
    return chosen


def subsample_indices(count: int, max_points: int = 70000) -> np.ndarray:
    if count <= max_points:
        return np.arange(count)
    step = max(1, count // max_points)
    return np.arange(0, count, step)


def region_mask(xyz: np.ndarray, region: Region) -> np.ndarray:
    return (
        (xyz[:, 0] >= region.x0)
        & (xyz[:, 0] < region.x0 + region.window)
        & (xyz[:, 1] >= region.y0)
        & (xyz[:, 1] < region.y0 + region.window)
    )


def render_region_figure(
    xyz: np.ndarray,
    rgb: np.ndarray,
    labels: np.ndarray,
    region: Region,
    out_path: Path,
) -> None:
    mask = region_mask(xyz, region)
    pts = xyz[mask]
    rgb_pts = rgb[mask]
    label_pts = LABEL_COLORS[labels[mask]]
    keep = subsample_indices(pts.shape[0])
    pts = pts[keep]
    rgb_pts = rgb_pts[keep]
    label_pts = label_pts[keep]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    title = (
        f"{region.title}  x=[{region.x0:.2f},{region.x0 + region.window:.2f})  "
        f"y=[{region.y0:.2f},{region.y0 + region.window:.2f})\n"
        f"count={region.count}  ground={region.ratios[0]:.2%}  shrubs={region.ratios[1]:.2%}  "
        f"trees={region.ratios[2]:.2%}"
    )
    fig.suptitle(title, fontsize=12)

    panels = [
        (axes[0, 0], pts[:, 0], pts[:, 1], rgb_pts, "Top View RGB", "x", "y"),
        (axes[0, 1], pts[:, 0], pts[:, 1], label_pts, "Top View Labels", "x", "y"),
        (axes[1, 0], pts[:, 0], pts[:, 2], rgb_pts, "Side View RGB", "x", "z"),
        (axes[1, 1], pts[:, 0], pts[:, 2], label_pts, "Side View Labels", "x", "z"),
    ]
    for ax, xs, ys, colors, panel_title, xlabel, ylabel in panels:
        ax.scatter(xs, ys, s=0.8, c=colors, linewidths=0, alpha=0.9)
        ax.set_title(panel_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.15)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=LABEL_COLORS[i], markersize=8, label=LABEL_NAMES[i])
        for i in range(3)
    ]
    fig.legend(handles=legend_handles, loc="upper right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def render_overview(
    xyz: np.ndarray,
    rgb: np.ndarray,
    labels: np.ndarray,
    regions: list[Region],
    out_path: Path,
) -> None:
    keep = subsample_indices(xyz.shape[0], max_points=120000)
    pts = xyz[keep]
    rgb_pts = rgb[keep]
    label_pts = LABEL_COLORS[labels[keep]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    axes[0].scatter(pts[:, 0], pts[:, 1], s=0.5, c=rgb_pts, linewidths=0, alpha=0.9)
    axes[1].scatter(pts[:, 0], pts[:, 1], s=0.5, c=label_pts, linewidths=0, alpha=0.9)
    axes[0].set_title("Whole Scene RGB Top View")
    axes[1].set_title("Whole Scene Label Top View")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.15)
        for region in regions:
            rect = plt.Rectangle((region.x0, region.y0), region.window, region.window, fill=False, linewidth=1.2, edgecolor="red")
            ax.add_patch(rect)
            ax.text(region.x0, region.y0 + region.window + 0.05, region.name, color="red", fontsize=10, weight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render label-based inspection figures for a labeled PLY.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input labeled PLY")
    parser.add_argument("--out-dir", dest="out_dir", required=True, help="Output directory for PNGs")
    args = parser.parse_args()

    xyz, rgb, labels = load_labeled_ply(args.input_path)
    candidates = score_candidates(xyz, labels)
    regions = select_regions(candidates)
    out_dir = Path(args.out_dir)

    render_overview(xyz, rgb, labels, regions, out_dir / "overview_topdown.png")
    for region in regions:
        render_region_figure(xyz, rgb, labels, region, out_dir / f"region_{region.name}.png")

    manifest = {
        "input": str(args.input_path),
        "regions": [
            {
                "name": region.name,
                "title": region.title,
                "x0": region.x0,
                "y0": region.y0,
                "window": region.window,
                "count": region.count,
                "ground_ratio": region.ratios[0],
                "shrubs_ratio": region.ratios[1],
                "trees_ratio": region.ratios[2],
                "z_median": region.z_median,
                "z_p95": region.z_p95,
            }
            for region in regions
        ],
    }
    with (out_dir / "regions.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
