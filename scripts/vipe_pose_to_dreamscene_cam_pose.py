#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert VIPE cam2world pose artifacts into DreamScene4D cam_pose.json."
    )
    parser.add_argument("--vipe_pose_npz", required=True, help="Path to VIPE pose npz.")
    parser.add_argument(
        "--output_json",
        required=True,
        help="Output DreamScene4D camera pose JSON path.",
    )
    parser.add_argument(
        "--output_cam_scales",
        default=None,
        help="Optional output path for DreamScene4D *_cam_scales.npy.",
    )
    parser.add_argument(
        "--default_scale",
        type=float,
        default=1.0,
        help="Fallback per-frame camera scale written to *_cam_scales.npy.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pose_npz = np.load(args.vipe_pose_npz)
    poses = pose_npz["data"]

    cameras = []
    for pose in poses:
        pose = np.asarray(pose, dtype=np.float64)
        rot = pose[:3, :3]
        trans = pose[:3, 3]

        quat_xyzw = R.from_matrix(rot).as_quat()
        cameras.append(
            {
                "pos": trans.tolist(),
                "orientation": quat_xyzw.tolist(),
            }
        )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(cameras, f, indent=2)

    if args.output_cam_scales:
        output_cam_scales = Path(args.output_cam_scales)
        output_cam_scales.parent.mkdir(parents=True, exist_ok=True)
        scales = np.full((len(cameras),), args.default_scale, dtype=np.float32)
        np.save(output_cam_scales, scales)

    print(f"Wrote {len(cameras)} camera poses to {output_json}")
    if args.output_cam_scales:
        print(f"Wrote default camera scales to {args.output_cam_scales}")


if __name__ == "__main__":
    main()
