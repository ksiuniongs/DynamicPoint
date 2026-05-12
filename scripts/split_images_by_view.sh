#!/usr/bin/env bash
set -euo pipefail

# 将一个 images 目录下的图片按视角 (yaw/pitch) 拆分到子目录中。
# 文件名格式假定为：frame_000123_yaw_090_pitch_000.jpg
#
# 用法：
#   bash scripts/split_images_by_view.sh [images_dir]
# 不传参数时，默认使用仓库根目录下的 data/gs360/images

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_IMAGES_DIR="$ROOT/data/gs360/images"
IMAGES_DIR="${1:-$DEFAULT_IMAGES_DIR}"

if [[ ! -d "$IMAGES_DIR" ]]; then
  echo "Error: images dir not found: $IMAGES_DIR" >&2
  exit 1
fi

echo "Splitting images in: $IMAGES_DIR"

shopt -s nullglob
cd "$IMAGES_DIR"

for f in frame_*_yaw_*_pitch_*.*; do
  name="$(basename "$f")"
  if [[ "$name" =~ yaw_([0-9]+)_pitch_([0-9]+) ]]; then
    yaw="${BASH_REMATCH[1]}"
    pitch="${BASH_REMATCH[2]}"
    dir="yaw_${yaw}_pitch_${pitch}"
    mkdir -p "$dir"
    mv "$f" "$dir/"
  else
    echo "Skip (no yaw/pitch pattern): $name" >&2
  fi
done

echo "Done. Images have been grouped by view (yaw/pitch) under: $IMAGES_DIR"