#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a single-view static Gaussian Splatting input from VIPE COLMAP text output and LaMa inpainted right-view images."
    )
    parser.add_argument("--vipe_colmap_dir", required=True, help="Directory containing VIPE-exported cameras.txt/images.txt/points3D.txt.")
    parser.add_argument("--lama_dir", required=True, help="Directory containing right_XXXXXX_mask001.png images.")
    parser.add_argument("--output_dir", required=True, help="Output Gaussian Splatting scene directory.")
    return parser.parse_args()


def main():
    args = parse_args()

    vipe_colmap_dir = Path(args.vipe_colmap_dir)
    lama_dir = Path(args.lama_dir)
    output_dir = Path(args.output_dir)
    sparse_dir = output_dir / "sparse" / "0"
    images_dir = output_dir / "images"

    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(vipe_colmap_dir / "cameras.txt", sparse_dir / "cameras.txt")
    shutil.copy2(vipe_colmap_dir / "points3D.txt", sparse_dir / "points3D.txt")

    src_images_txt = vipe_colmap_dir / "images.txt"
    dst_images_txt = sparse_dir / "images.txt"
    with src_images_txt.open("r", encoding="utf-8") as fin, dst_images_txt.open("w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                fout.write(line)
                continue

            parts = stripped.split()
            if len(parts) >= 10:
                parts[9] = Path(parts[9]).name
                fout.write(" ".join(parts) + "\n")
            else:
                fout.write(line)

    right_files = sorted(lama_dir.glob("right_*_mask001.png"))
    if not right_files:
        raise FileNotFoundError(f"No right-view LaMa outputs found under {lama_dir}")

    for src in right_files:
        frame_id = src.stem.split("_")[1]
        dst = images_dir / f"frame_{frame_id}.jpg"
        with Image.open(src) as img:
            img.convert("RGB").save(dst, quality=95)

    print(f"Prepared {len(right_files)} images under {images_dir}")
    print(f"Wrote sparse model under {sparse_dir}")


if __name__ == "__main__":
    main()
