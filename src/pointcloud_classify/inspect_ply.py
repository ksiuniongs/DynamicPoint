from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pointcloud_classify.io_ply import inspect_ply
else:
    from .io_ply import inspect_ply


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PLY metadata.")
    parser.add_argument("path", help="Input PLY path")
    args = parser.parse_args()
    print(json.dumps(inspect_ply(args.path), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
