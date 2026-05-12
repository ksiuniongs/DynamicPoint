#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


LEFT_VIEW = "yaw_m030_pitch_000"
RIGHT_VIEW = "yaw_030_pitch_000"


def copy_and_renumber(src_dir: Path, dst_dir: Path, start_index: int | None, count: int | None) -> int:
    files = sorted(src_dir.glob("*.jpg"))
    if start_index is not None:
        files = files[start_index:]
    if count is not None:
        files = files[:count]

    if not files:
        raise SystemExit(f"No JPG files selected from {src_dir}")

    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for idx, src in enumerate(files):
        shutil.copy2(src, dst_dir / f"frame_{idx:06d}.jpg")
    return len(files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select SCGS left/right stereo views and renumber them to frame_XXXXXX.jpg")
    parser.add_argument("--views_root", required=True, help="Root directory containing yaw_*_pitch_* subdirectories")
    parser.add_argument("--out_root", required=True, help="Output root that will contain left/ and right/")
    parser.add_argument("--start_index", type=int, default=None, help="Optional starting frame index within each selected view")
    parser.add_argument("--count", type=int, default=None, help="Optional number of frames to keep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    views_root = Path(args.views_root)
    out_root = Path(args.out_root)

    left_src = views_root / LEFT_VIEW
    right_src = views_root / RIGHT_VIEW
    if not left_src.is_dir():
        raise SystemExit(f"Missing left source view directory: {left_src}")
    if not right_src.is_dir():
        raise SystemExit(f"Missing right source view directory: {right_src}")

    left_count = copy_and_renumber(left_src, out_root / "left", args.start_index, args.count)
    right_count = copy_and_renumber(right_src, out_root / "right", args.start_index, args.count)

    if left_count != right_count:
        raise SystemExit(f"Left/right frame count mismatch: {left_count} vs {right_count}")

    print(f"Prepared stereo views under {out_root}")
    print(f"Frames per side: {left_count}")


if __name__ == "__main__":
    main()
