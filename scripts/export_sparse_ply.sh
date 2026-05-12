#!/usr/bin/env bash
set -euo pipefail

# Usage: export_sparse_ply.sh <workdir> <timestamp>
# Example:
#   ./scripts/export_sparse_ply.sh /mnt/d/develop/master_thesis/DynamicPoint/data/gs360 20260210_150523

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <workdir> <timestamp>" >&2
  exit 1
fi

WORKDIR="$1"
TS="$2"

INPUT_MODEL="$WORKDIR/$TS/sparse/0"
OUTPUT_FILE="$WORKDIR/$TS/ply/points3D.ply"

if [[ ! -d "$INPUT_MODEL" ]]; then
  echo "Error: input model directory not found: $INPUT_MODEL" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"

colmap model_converter \
  --input_path "$INPUT_MODEL" \
  --output_path "$OUTPUT_FILE" \
  --output_type PLY

echo "Exported sparse model to: $OUTPUT_FILE"