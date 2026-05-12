#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def rotmat_to_qvec(rmat: np.ndarray) -> np.ndarray:
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rmat.flat
    k = np.array(
        [
            [rxx - ryy - rzz, 0.0, 0.0, 0.0],
            [ryx + rxy, ryy - rxx - rzz, 0.0, 0.0],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0.0],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ],
        dtype=np.float64,
    ) / 3.0
    eigvals, eigvecs = np.linalg.eigh(k)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1.0
    return qvec


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a COLMAP known-pose text model from UniSH cameras.")
    ap.add_argument("--camera_npz", required=True)
    ap.add_argument("--database_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    args = ap.parse_args()

    import sqlite3

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.camera_npz)
    extrinsics = data["extrinsics"].astype(np.float64)
    intrinsics = data["intrinsics"].astype(np.float64)

    conn = sqlite3.connect(args.database_path)
    cur = conn.cursor()
    db_images = list(cur.execute("SELECT image_id, name FROM images ORDER BY image_id"))
    conn.close()

    if len(db_images) != len(extrinsics):
        raise ValueError(
            f"Database has {len(db_images)} images but camera file has {len(extrinsics)} poses"
        )

    # Keep the DB camera layout simple: one shared camera, using the mean UniSH intrinsics.
    fx = float(intrinsics[:, 0, 0].mean())
    fy = float(intrinsics[:, 1, 1].mean())
    cx = float(intrinsics[:, 0, 2].mean())
    cy = float(intrinsics[:, 1, 2].mean())

    with open(output_dir / "cameras.txt", "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {args.width} {args.height} {fx} {fy} {cx} {cy}\n")

    with open(output_dir / "images.txt", "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(db_images)}\n")
        for (image_id, name), w2c in zip(db_images, extrinsics):
            r = w2c[:3, :3]
            t = w2c[:3, 3]
            q = rotmat_to_qvec(r)
            f.write(
                f"{image_id} {q[0]} {q[1]} {q[2]} {q[3]} "
                f"{t[0]} {t[1]} {t[2]} 1 {name}\n"
            )
            f.write("\n")

    with open(output_dir / "points3D.txt", "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0, mean track length: 0\n")


if __name__ == "__main__":
    main()
