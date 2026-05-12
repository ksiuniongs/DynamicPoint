from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData, PlyElement


LABEL_RGB = {
    0: (166, 124, 82),
    1: (109, 179, 63),
    2: (35, 97, 44),
}


def load_vertex(path: str | Path):
    ply = PlyData.read(str(path), mmap=True)
    return ply, ply["vertex"].data


def recolor_vertex(vertex: np.ndarray) -> np.ndarray:
    field_names = vertex.dtype.names or ()
    if "label" not in field_names:
        raise ValueError("Input PLY must contain a 'label' field.")
    if not {"red", "green", "blue"}.issubset(field_names):
        raise ValueError("Input PLY must contain red/green/blue fields for recoloring.")

    out = np.empty(vertex.shape[0], dtype=vertex.dtype)
    for name in field_names:
        out[name] = vertex[name]

    labels = vertex["label"].astype(np.uint8, copy=False)
    colors = np.zeros((vertex.shape[0], 3), dtype=np.uint8)
    for label, rgb in LABEL_RGB.items():
        mask = labels == label
        colors[mask] = rgb

    out["red"] = colors[:, 0]
    out["green"] = colors[:, 1]
    out["blue"] = colors[:, 2]
    return out


def write_recolored_ply(ply: PlyData, vertex: np.ndarray, out_path: str | Path) -> None:
    elements = []
    for element in ply.elements:
        if element.name == "vertex":
            elements.append(PlyElement.describe(vertex, "vertex"))
        else:
            elements.append(element)
    out = PlyData(
        elements,
        text=ply.text,
        byte_order=ply.byte_order,
        comments=ply.comments,
        obj_info=ply.obj_info,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(out_path))


def subsample_indices(count: int, max_points: int) -> np.ndarray:
    if count <= max_points:
        return np.arange(count)
    step = max(1, count // max_points)
    return np.arange(0, count, step)


def render_screenshots(vertex: np.ndarray, out_dir: str | Path, max_points: int = 150000) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32, copy=False)
    rgb = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]).astype(np.float32) / 255.0
    keep = subsample_indices(xyz.shape[0], max_points)
    xyz = xyz[keep]
    rgb = rgb[keep]

    scenes = [
        ("topdown_xy.png", xyz[:, 0], xyz[:, 1], "x", "y", "Top View XY"),
        ("side_xz.png", xyz[:, 0], xyz[:, 2], "x", "z", "Side View XZ"),
        ("side_yz.png", xyz[:, 1], xyz[:, 2], "y", "z", "Side View YZ"),
    ]
    for filename, xs, ys, xlabel, ylabel, title in scenes:
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        ax.scatter(xs, ys, c=rgb, s=0.7, linewidths=0, alpha=0.95)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.15)
        fig.savefig(out_dir / filename, dpi=240)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a label-colored PLY and preview screenshots.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input labeled PLY")
    parser.add_argument("--out-ply", dest="output_ply", required=True, help="Output recolored PLY")
    parser.add_argument("--out-dir", dest="output_dir", required=True, help="Output screenshot directory")
    args = parser.parse_args()

    ply, vertex = load_vertex(args.input_path)
    recolored = recolor_vertex(vertex)
    write_recolored_ply(ply, recolored, args.output_ply)
    render_screenshots(recolored, args.output_dir)


if __name__ == "__main__":
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
