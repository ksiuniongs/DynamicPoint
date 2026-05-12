#!/usr/bin/env python3
from pathlib import Path
import argparse
import os


def main():
    ap = argparse.ArgumentParser(description="Flatten left/right image folders into a COLMAP images directory.")
    ap.add_argument("--left_dir", required=True, help="Directory containing left images")
    ap.add_argument("--right_dir", required=True, help="Directory containing right images")
    ap.add_argument("--output_dir", required=True, help="Flattened COLMAP images directory")
    args = ap.parse_args()

    left_dir = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    left_files = sorted(p for p in left_dir.glob("*.jpg"))
    right_files = sorted(p for p in right_dir.glob("*.jpg"))

    left_names = [p.name for p in left_files]
    right_names = [p.name for p in right_files]
    if left_names != right_names:
        raise SystemExit("left/right image names do not match")

    for src in output_dir.iterdir():
        if src.is_symlink() or src.is_file():
            src.unlink()

    for idx, (left_path, right_path) in enumerate(zip(left_files, right_files)):
        left_name = output_dir / f"{idx:06d}_left.jpg"
        right_name = output_dir / f"{idx:06d}_right.jpg"
        os.symlink(left_path.resolve(), left_name)
        os.symlink(right_path.resolve(), right_name)

    print(f"prepared {len(left_files) * 2} images in {output_dir}")


if __name__ == "__main__":
    main()
