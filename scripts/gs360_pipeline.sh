#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  gs360_pipeline.sh [--video <path>] [--workdir <dir>] [--fps <n>] [--yaw <list>] [--pitch <list>] [--fov <deg>] [--size <WxH>] [--matcher <sequential|exhaustive>] [--pre]

Defaults:
  --workdir data/gs360
  --fps 2
  --yaw 0,90,180,270
  --pitch 0
  --fov 90
  --size 800x800
  --matcher sequential
  (steps [1] Extract frames and [2] Equirectangular -> pinhole are skipped by default; pass --pre to run them — in that case, --video is required)
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO=""
WORKDIR="$ROOT/data/gs360"
FPS="2"
YAW="0,90,270"
PITCH="0"
FOV="90"
SIZE="800x800"
MATCHER="sequential"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_PRE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO="$2"; shift 2;;
    --workdir) WORKDIR="$2"; shift 2;;
    --fps) FPS="$2"; shift 2;;
    --yaw) YAW="$2"; shift 2;;
    --pitch) PITCH="$2"; shift 2;;
    --fov) FOV="$2"; shift 2;;
    --size) SIZE="$2"; shift 2;;
    --matcher) MATCHER="$2"; shift 2;;
    --pre) RUN_PRE=true; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if "$RUN_PRE" && [[ -z "$VIDEO" ]]; then
  echo "Error: --video is required when using --pre"
  usage
  exit 1
fi

RUN_DIR="$WORKDIR/$TIMESTAMP"
BASE_FRAMES_DIR="$WORKDIR/frames"
BASE_IMAGES_DIR="$WORKDIR/images"
FRAMES_DIR="$RUN_DIR/frames"
IMAGES_DIR="$RUN_DIR/images"
SPARSE_DIR="$RUN_DIR/sparse"
DB_PATH="$RUN_DIR/database.db"
GS_DIR="$ROOT/submodules/gaussian-splatting"

mkdir -p "$RUN_DIR" "$BASE_FRAMES_DIR" "$BASE_IMAGES_DIR" "$FRAMES_DIR" "$IMAGES_DIR" "$SPARSE_DIR"

if "$RUN_PRE"; then
  echo "[1/4] Extract frames"
  # 在这里启用你想要的 ffmpeg 抽帧命令，例如：
  # ffmpeg -y -i "$VIDEO" -vf "fps=$FPS" -vframes 200 "$BASE_FRAMES_DIR/frame_%06d.jpg"

  echo "[2/4] Equirectangular -> pinhole"
  python3 "$ROOT/scripts/equirect_to_pinhole.py" \
    --input_dir "$BASE_FRAMES_DIR" \
    --output_dir "$BASE_IMAGES_DIR" \
    --yaw "$YAW" \
    --pitch "$PITCH" \
    --fov "$FOV" \
    --size "$SIZE" \
    --ext jpg
fi

echo "[1.5] Prepare run-local frames/images"
if [[ -d "$BASE_FRAMES_DIR" ]]; then
  rm -rf "$FRAMES_DIR"
  mkdir -p "$FRAMES_DIR"
  cp -a "$BASE_FRAMES_DIR/." "$FRAMES_DIR/"
fi
if [[ -d "$BASE_IMAGES_DIR" ]]; then
  rm -rf "$IMAGES_DIR"
  mkdir -p "$IMAGES_DIR"
  cp -a "$BASE_IMAGES_DIR/." "$IMAGES_DIR/"
  # 删除 180° 视角的图片，避免参与 SfM
  find "$IMAGES_DIR" -type f -name '*yaw_180_*' -delete || true
fi

echo "[3/4] COLMAP SfM"
colmap feature_extractor \
  --database_path "$DB_PATH" \
  --image_path "$IMAGES_DIR"

if [[ "$MATCHER" == "sequential" ]]; then
  colmap sequential_matcher --database_path "$DB_PATH"
elif [[ "$MATCHER" == "exhaustive" ]]; then
  colmap exhaustive_matcher --database_path "$DB_PATH"
else
  echo "Unknown matcher: $MATCHER"
  exit 1
fi

colmap mapper \
  --database_path "$DB_PATH" \
  --image_path "$IMAGES_DIR" \
  --output_path "$SPARSE_DIR"

echo "[4/4] Prepare 3DGS layout"
DISTORTED_DIR="$RUN_DIR/distorted"
INPUT_DIR="$RUN_DIR/input"

mkdir -p "$DISTORTED_DIR"

if [[ ! -d "$INPUT_DIR" ]]; then
  ln -s "$IMAGES_DIR" "$INPUT_DIR"
fi

if [[ ! -f "$DISTORTED_DIR/database.db" ]]; then
  ln -s "$DB_PATH" "$DISTORTED_DIR/database.db"
fi

if [[ ! -d "$DISTORTED_DIR/sparse" ]]; then
  ln -s "$SPARSE_DIR" "$DISTORTED_DIR/sparse"
fi

echo "[4/4] Convert for 3DGS"
python3 "$GS_DIR/convert.py" -s "$RUN_DIR" --skip_matching

echo "Done. Output dataset at: $RUN_DIR"
