#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Assemble LaMa outputs into a COLMAP/3DGS input directory.")
    parser.add_argument("--lama_outdir", required=True, help="LaMa output directory")
    parser.add_argument("--outdir", required=True, help="Assembled image directory")
    args = parser.parse_args()

    lama_outdir = Path(args.lama_outdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    output_paths = sorted([p for p in lama_outdir.iterdir() if p.is_file()])
    if not output_paths:
        raise RuntimeError(f"No files found in {lama_outdir}")

    for path in output_paths:
        stem = path.stem
        if not stem.endswith("_mask001"):
            continue

        image_id = stem[:-8]
        if "_" not in image_id:
            raise ValueError(f"Unexpected LaMa output filename: {path.name}")

        side, frame_id = image_id.split("_", 1)
        if side not in {"left", "right"}:
            raise ValueError(f"Unexpected side in LaMa output filename: {path.name}")

        target = outdir / f"{frame_id}_{side}{path.suffix.lower()}"
        shutil.copy2(path, target)


if __name__ == "__main__":
    main()
