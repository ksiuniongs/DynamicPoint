from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def compare_pair(old_path: Path, new_path: Path, out_path: Path, title: str) -> None:
    old = load_image(old_path)
    new = load_image(new_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    axes[0].imshow(old)
    axes[0].set_title(f"Old ({old_path.parent.name})")
    axes[1].imshow(new)
    axes[1].set_title(f"New ({new_path.parent.name})")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare old vs new label screenshots.")
    parser.add_argument("--old-dir", required=True)
    parser.add_argument("--new-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    old_dir = Path(args.old_dir)
    new_dir = Path(args.new_dir)
    out_dir = Path(args.out_dir)
    for name, title in [
        ("topdown_xy.png", "Top View Comparison"),
        ("side_xz.png", "Side View XZ Comparison"),
        ("side_yz.png", "Side View YZ Comparison"),
    ]:
        compare_pair(old_dir / name, new_dir / name, out_dir / name, title)


if __name__ == "__main__":
    main()
