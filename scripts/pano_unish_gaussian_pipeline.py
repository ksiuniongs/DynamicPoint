#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from extract_scgs_views import parse_size, project_equirect_to_pinhole


def extract_frames(video_path: Path, frames_dir: Path, fps: float) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-start_number",
        "0",
        str(frames_dir / "frame_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)


def equirect_pixel_to_angles(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    yaw = ((x / width) - 0.5) * 360.0
    pitch = (0.5 - (y / height)) * 180.0
    return yaw, pitch


def circular_blend_deg(prev_deg: float | None, cur_deg: float, alpha: float) -> float:
    if prev_deg is None:
        return cur_deg
    prev_rad = np.deg2rad(prev_deg)
    cur_rad = np.deg2rad(cur_deg)
    prev_vec = np.array([np.cos(prev_rad), np.sin(prev_rad)], dtype=np.float64)
    cur_vec = np.array([np.cos(cur_rad), np.sin(cur_rad)], dtype=np.float64)
    blended = alpha * prev_vec + (1.0 - alpha) * cur_vec
    if np.linalg.norm(blended) < 1e-8:
        return cur_deg
    return float(np.rad2deg(np.arctan2(blended[1], blended[0])))


def linear_blend(prev_value: float | None, cur_value: float, alpha: float) -> float:
    if prev_value is None:
        return cur_value
    return float(alpha * prev_value + (1.0 - alpha) * cur_value)


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return float(inter / (area_a + area_b - inter))


def summarize_track(track: dict) -> dict:
    centers = np.array(track["centers"], dtype=np.float64)
    areas = np.array(track["areas"], dtype=np.float64)
    displacement = 0.0
    if len(centers) >= 2:
        displacement = float(np.linalg.norm(centers[-1] - centers[0]))
    return {
        "track_id": track["track_id"],
        "num_frames": len(track["frames"]),
        "start_frame": track["frames"][0],
        "end_frame": track["frames"][-1],
        "avg_area": float(areas.mean()) if len(areas) else 0.0,
        "max_area": float(areas.max()) if len(areas) else 0.0,
        "displacement_px": displacement,
        "is_moving": displacement > 80.0,
    }


def detect_all_tracks(
    frames_dir: Path,
    yolo_model_path: str,
    target_class: str,
    conf_thres: float,
    max_center_dist: float,
    max_missed: int,
) -> tuple[list[dict], list[dict]]:
    from ultralytics import YOLO

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise SystemExit("no extracted frames found for multi-target detection")

    model = YOLO(yolo_model_path)
    class_names = model.names

    if target_class.isdigit():
        target_class_id = int(target_class)
    else:
        reverse_names = {str(v): int(k) for k, v in class_names.items()}
        if target_class not in reverse_names:
            raise SystemExit(f"unknown target_class '{target_class}', available names: {sorted(reverse_names)}")
        target_class_id = reverse_names[target_class]

    next_track_id = 0
    active_tracks: list[dict] = []
    finished_tracks: list[dict] = []
    frame_records: list[dict] = []

    for frame_index, frame_path in enumerate(frame_paths):
        result = model.predict(
            source=str(frame_path),
            verbose=False,
            classes=[target_class_id],
            conf=conf_thres,
        )[0]

        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for box, conf in zip(xyxy, confs):
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                area = max(1.0, (x2 - x1) * (y2 - y1))
                detections.append(
                    {
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "center_xy": [cx, cy],
                        "area": area,
                        "confidence": float(conf),
                    }
                )

        assignments = {}
        used_track_ids = set()
        used_det_ids = set()

        candidate_pairs = []
        for det_id, det in enumerate(detections):
            det_center = np.array(det["center_xy"], dtype=np.float64)
            for track in active_tracks:
                if track["missed"] > max_missed:
                    continue
                track_center = np.array(track["last_center"], dtype=np.float64)
                center_dist = float(np.linalg.norm(det_center - track_center))
                iou = bbox_iou(det["bbox_xyxy"], track["last_bbox"])
                if center_dist <= max_center_dist or iou > 0.05:
                    score = center_dist - 200.0 * iou
                    candidate_pairs.append((score, det_id, track["track_id"]))

        for _, det_id, track_id in sorted(candidate_pairs, key=lambda x: x[0]):
            if det_id in used_det_ids or track_id in used_track_ids:
                continue
            assignments[det_id] = track_id
            used_det_ids.add(det_id)
            used_track_ids.add(track_id)

        track_lookup = {track["track_id"]: track for track in active_tracks}
        current_frame_records = []

        for det_id, det in enumerate(detections):
            assigned_track_id = assignments.get(det_id)
            if assigned_track_id is None:
                assigned_track_id = next_track_id
                next_track_id += 1
                track = {
                    "track_id": assigned_track_id,
                    "frames": [],
                    "boxes": [],
                    "centers": [],
                    "areas": [],
                    "confs": [],
                    "last_center": det["center_xy"],
                    "last_bbox": det["bbox_xyxy"],
                    "missed": 0,
                }
                active_tracks.append(track)
                track_lookup[assigned_track_id] = track

            track = track_lookup[assigned_track_id]
            track["frames"].append(frame_index)
            track["boxes"].append(det["bbox_xyxy"])
            track["centers"].append(det["center_xy"])
            track["areas"].append(det["area"])
            track["confs"].append(det["confidence"])
            track["last_center"] = det["center_xy"]
            track["last_bbox"] = det["bbox_xyxy"]
            track["missed"] = 0

            current_frame_records.append(
                {
                    "track_id": assigned_track_id,
                    "bbox_xyxy": det["bbox_xyxy"],
                    "center_xy": det["center_xy"],
                    "area": det["area"],
                    "confidence": det["confidence"],
                }
            )

        matched_track_ids = {item["track_id"] for item in current_frame_records}
        still_active = []
        for track in active_tracks:
            if track["track_id"] not in matched_track_ids:
                track["missed"] += 1
            if track["missed"] > max_missed:
                finished_tracks.append(track)
            else:
                still_active.append(track)
        active_tracks = still_active

        frame_records.append(
            {
                "frame": frame_path.name,
                "frame_index": frame_index,
                "detections": current_frame_records,
            }
        )

    finished_tracks.extend(active_tracks)
    finished_tracks = sorted(finished_tracks, key=lambda t: t["track_id"])
    summaries = [summarize_track(track) for track in finished_tracks]
    return frame_records, summaries


def render_detection_overlays(
    frames_dir: Path,
    output_dir: Path,
    frame_records: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    record_lookup = {item["frame"]: item for item in frame_records}
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, 220) if Path(font_path).exists() else ImageFont.load_default()
    for frame_path in sorted(frames_dir.glob("frame_*.jpg")):
        img = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        frame_record = record_lookup.get(frame_path.name, {"detections": []})
        for det in frame_record["detections"]:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            track_id = det["track_id"]
            color = (
                int((53 * (track_id + 1)) % 255),
                int((97 * (track_id + 3)) % 255),
                int((193 * (track_id + 7)) % 255),
            )
            line_width = 14
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            label = f"track {track_id}"
            text_box = draw.textbbox((0, 0), label, font=font, stroke_width=8)
            text_w = text_box[2] - text_box[0]
            text_h = text_box[3] - text_box[1]
            pad_x = 56
            pad_y = 36
            box_w = text_w + 2 * pad_x
            box_h = text_h + 2 * pad_y
            box_x1 = x1 + 20
            box_y1 = max(0, y1 - box_h - 20)
            box_x2 = box_x1 + box_w
            box_y2 = box_y1 + box_h
            draw.rounded_rectangle(
                [box_x1, box_y1, box_x2, box_y2],
                radius=14,
                fill=(0, 0, 0),
                outline=color,
                width=6,
            )
            draw.text(
                (box_x1 + pad_x, box_y1 + pad_y - 10),
                label,
                fill=(255, 255, 255),
                font=font,
                stroke_width=8,
                stroke_fill=(0, 0, 0),
            )
        img.save(output_dir / frame_path.name, quality=95)


def detect_follow_angles(
    frames_dir: Path,
    yolo_model_path: str,
    target_class: str,
    smoothing: float,
) -> list[dict]:
    from ultralytics import YOLO

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise SystemExit("no extracted frames found for YOLO follow mode")

    model = YOLO(yolo_model_path)
    class_names = model.names

    if target_class.isdigit():
        target_class_id = int(target_class)
    else:
        reverse_names = {str(v): int(k) for k, v in class_names.items()}
        if target_class not in reverse_names:
            raise SystemExit(f"unknown target_class '{target_class}', available names: {sorted(reverse_names)}")
        target_class_id = reverse_names[target_class]

    prev_center = None
    prev_yaw = None
    prev_pitch = None
    detections = []

    for frame_path in frame_paths:
        result = model.predict(
            source=str(frame_path),
            verbose=False,
            classes=[target_class_id],
            conf=0.15,
        )[0]

        img = Image.open(frame_path)
        width, height = img.size

        selected = None
        selected_score = None
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for box, conf in zip(xyxy, confs):
                x1, y1, x2, y2 = box.tolist()
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                area = max(1.0, (x2 - x1) * (y2 - y1))
                if prev_center is None:
                    score = area * float(conf)
                else:
                    dist = np.linalg.norm(np.array([cx, cy]) - prev_center)
                    score = (area * float(conf)) / (1.0 + 0.002 * dist)
                if selected_score is None or score > selected_score:
                    selected = (cx, cy, x1, y1, x2, y2, float(conf))
                    selected_score = score

        if selected is None:
            if prev_yaw is None or prev_pitch is None:
                cur_yaw, cur_pitch = 0.0, 0.0
            else:
                cur_yaw, cur_pitch = prev_yaw, prev_pitch
            detections.append(
                {
                    "frame": frame_path.name,
                    "detected": False,
                    "yaw": cur_yaw,
                    "pitch": cur_pitch,
                }
            )
            continue

        cx, cy, x1, y1, x2, y2, conf = selected
        raw_yaw, raw_pitch = equirect_pixel_to_angles(cx, cy, width, height)
        cur_yaw = circular_blend_deg(prev_yaw, raw_yaw, smoothing)
        cur_pitch = linear_blend(prev_pitch, raw_pitch, smoothing)
        prev_center = np.array([cx, cy], dtype=np.float64)
        prev_yaw = cur_yaw
        prev_pitch = cur_pitch
        detections.append(
            {
                "frame": frame_path.name,
                "detected": True,
                "bbox_xyxy": [x1, y1, x2, y2],
                "confidence": conf,
                "center_xy": [cx, cy],
                "yaw": cur_yaw,
                "pitch": cur_pitch,
                "raw_yaw": raw_yaw,
                "raw_pitch": raw_pitch,
            }
        )

    return detections


def load_track_follow_angles(
    frames_dir: Path,
    tracks_json_path: Path,
    selected_track_id: int,
    smoothing: float,
) -> list[dict]:
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise SystemExit("no extracted frames found for selected-track follow mode")

    with tracks_json_path.open("r", encoding="utf-8") as f:
        frame_records = json.load(f)

    record_lookup = {item["frame"]: item for item in frame_records}
    prev_yaw = None
    prev_pitch = None
    detections = []

    for frame_path in frame_paths:
        img = Image.open(frame_path)
        width, height = img.size
        frame_record = record_lookup.get(frame_path.name, {"detections": []})
        selected = None
        for det in frame_record["detections"]:
            if det["track_id"] == selected_track_id:
                selected = det
                break

        if selected is None:
            if prev_yaw is None or prev_pitch is None:
                cur_yaw, cur_pitch = 0.0, 0.0
            else:
                cur_yaw, cur_pitch = prev_yaw, prev_pitch
            detections.append(
                {
                    "frame": frame_path.name,
                    "detected": False,
                    "track_id": selected_track_id,
                    "yaw": cur_yaw,
                    "pitch": cur_pitch,
                }
            )
            continue

        cx, cy = selected["center_xy"]
        raw_yaw, raw_pitch = equirect_pixel_to_angles(cx, cy, width, height)
        cur_yaw = circular_blend_deg(prev_yaw, raw_yaw, smoothing)
        cur_pitch = linear_blend(prev_pitch, raw_pitch, smoothing)
        prev_yaw = cur_yaw
        prev_pitch = cur_pitch
        detections.append(
            {
                "frame": frame_path.name,
                "detected": True,
                "track_id": selected_track_id,
                "bbox_xyxy": selected["bbox_xyxy"],
                "center_xy": selected["center_xy"],
                "yaw": cur_yaw,
                "pitch": cur_pitch,
                "raw_yaw": raw_yaw,
                "raw_pitch": raw_pitch,
            }
        )

    return detections


def render_main_view(
    frames_dir: Path,
    output_dir: Path,
    size: str,
    fov: float,
    yaw: float,
    pitch: float,
) -> None:
    out_w, out_h = parse_size(size)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise SystemExit("no extracted frames found for perspective rendering")

    for frame_path in frame_paths:
        arr = np.array(Image.open(frame_path).convert("RGB"), dtype=np.uint8)
        out = project_equirect_to_pinhole(arr, out_w, out_h, fov, yaw, pitch)
        Image.fromarray(out).save(output_dir / frame_path.name, quality=95)


def render_follow_view(
    frames_dir: Path,
    output_dir: Path,
    size: str,
    fov: float,
    detections: list[dict],
) -> None:
    out_w, out_h = parse_size(size)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_lookup = {item["frame"]: item for item in detections}
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise SystemExit("no extracted frames found for follow rendering")

    for frame_path in frame_paths:
        arr = np.array(Image.open(frame_path).convert("RGB"), dtype=np.uint8)
        item = frame_lookup.get(frame_path.name)
        if item is None:
            raise SystemExit(f"missing detection metadata for {frame_path.name}")
        out = project_equirect_to_pinhole(arr, out_w, out_h, fov, item["yaw"], item["pitch"])
        Image.fromarray(out).save(output_dir / frame_path.name, quality=95)


def create_preview_video(input_dir: Path, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(input_dir / "frame_%06d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def create_scaled_preview_video(input_dir: Path, output_path: Path, fps: float, width: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(input_dir / "frame_%06d.jpg"),
        "-vf",
        f"scale={width}:-2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def save_step1_metadata(output_root: Path, args: argparse.Namespace, step1_meta: dict) -> None:
    metadata = {
        "video_path": str(Path(args.video).resolve()),
        "step1": step1_meta,
        "step2_unish": {"status": "todo"},
        "step3_gaussian": {"status": "todo"},
        "step4_world_alignment": {"status": "todo"},
        "step5_dynamic_refine": {"status": "todo"},
    }
    (output_root / "pipeline_state.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_unish_placeholder(_: Path) -> None:
    # TODO: run UniSH on the validated main-view crop.
    return


def run_gaussian_placeholder(_: Path) -> None:
    # TODO: run the static Gaussian pipeline and export the scene world frame.
    return


def align_worlds_placeholder(_: Path) -> None:
    # TODO: estimate the Sim3 transform between UniSH world and Gaussian world.
    return


def refine_dynamic_placeholder(_: Path) -> None:
    # TODO: refine dynamic-object placement after the coarse world alignment.
    return


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Panorama -> main perspective crop -> UniSH/Gaussian pipeline scaffold"
    )
    ap.add_argument("--video", required=True, help="Input panorama video")
    ap.add_argument("--output_root", required=True, help="Output root directory")
    ap.add_argument("--extract_fps", type=float, default=6.0, help="Frame extraction FPS")
    ap.add_argument("--size", default="700x700", help="Perspective crop size WxH")
    ap.add_argument("--fov", type=float, default=90.0, help="Perspective horizontal FOV")
    ap.add_argument("--yaw", type=float, default=0.0, help="Main-view yaw in degrees")
    ap.add_argument("--pitch", type=float, default=0.0, help="Main-view pitch in degrees")
    ap.add_argument(
        "--follow_yolo",
        action="store_true",
        help="Use YOLO on panorama frames and let the perspective crop follow the target",
    )
    ap.add_argument(
        "--detect_all_yolo",
        action="store_true",
        help="Detect and track all target instances first, then inspect manually before choosing one",
    )
    ap.add_argument("--yolo_model", default="yolo11n.pt", help="YOLO checkpoint for follow mode")
    ap.add_argument("--target_class", default="person", help="YOLO target class name or id")
    ap.add_argument("--yolo_conf", type=float, default=0.15, help="YOLO confidence threshold")
    ap.add_argument("--track_max_center_dist", type=float, default=500.0, help="Max center distance for greedy track association")
    ap.add_argument("--track_max_missed", type=int, default=2, help="Max missed frames before a track closes")
    ap.add_argument("--selected_track_id", type=int, default=None, help="Use an existing tracked subject id for follow-view rendering")
    ap.add_argument("--tracks_json", default=None, help="Path to step1_all_tracks.json for selected-track follow mode")
    ap.add_argument(
        "--follow_smoothing",
        type=float,
        default=0.8,
        help="Exponential smoothing factor for yaw/pitch tracking, in [0,1)",
    )
    ap.add_argument(
        "--skip_extract",
        action="store_true",
        help="Reuse already extracted panorama frames under output_root/frames",
    )
    args = ap.parse_args()

    output_root = Path(args.output_root)
    frames_dir = output_root / "frames"
    main_view_dir = output_root / "step1_main_view"
    preview_path = output_root / "step1_main_view_preview.mp4"
    detections_path = output_root / "step1_follow_detections.json"
    all_tracks_path = output_root / "step1_all_tracks.json"
    all_tracks_summary_path = output_root / "step1_all_tracks_summary.json"
    overlay_dir = output_root / "step1_all_tracks_overlay"
    overlay_preview_path = output_root / "step1_all_tracks_preview.mp4"
    overlay_preview_small_path = output_root / "step1_all_tracks_preview_1920.mp4"

    if not args.skip_extract:
        extract_frames(Path(args.video), frames_dir, args.extract_fps)

    if args.detect_all_yolo:
        frame_records, summaries = detect_all_tracks(
            frames_dir=frames_dir,
            yolo_model_path=args.yolo_model,
            target_class=args.target_class,
            conf_thres=args.yolo_conf,
            max_center_dist=args.track_max_center_dist,
            max_missed=args.track_max_missed,
        )
        all_tracks_path.write_text(
            json.dumps(frame_records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        all_tracks_summary_path.write_text(
            json.dumps(summaries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        render_detection_overlays(frames_dir, overlay_dir, frame_records)
        create_preview_video(overlay_dir, overlay_preview_path, args.extract_fps)
        create_scaled_preview_video(overlay_dir, overlay_preview_small_path, args.extract_fps, width=1920)
        step1_meta = {
            "fps": args.extract_fps,
            "mode": "detect_all_yolo",
            "target_class": args.target_class,
            "yolo_model": args.yolo_model,
            "yolo_conf": args.yolo_conf,
            "tracks_path": str(all_tracks_path),
            "tracks_summary_path": str(all_tracks_summary_path),
            "overlay_preview_path": str(overlay_preview_path),
            "overlay_preview_small_path": str(overlay_preview_small_path),
            "status": "done",
            "note": "Detect all candidate subjects first, then choose a track manually for the follow-view crop.",
        }
    elif args.selected_track_id is not None:
        if not args.tracks_json:
            raise SystemExit("--tracks_json is required when --selected_track_id is used")
        detections = load_track_follow_angles(
            frames_dir=frames_dir,
            tracks_json_path=Path(args.tracks_json),
            selected_track_id=args.selected_track_id,
            smoothing=args.follow_smoothing,
        )
        detections_path.write_text(
            json.dumps(detections, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        render_follow_view(
            frames_dir=frames_dir,
            output_dir=main_view_dir,
            size=args.size,
            fov=args.fov,
            detections=detections,
        )
        step1_meta = {
            "fps": args.extract_fps,
            "size": args.size,
            "fov": args.fov,
            "mode": "follow_selected_track",
            "selected_track_id": args.selected_track_id,
            "tracks_json": args.tracks_json,
            "follow_smoothing": args.follow_smoothing,
            "detections_path": str(detections_path),
            "status": "done",
            "note": "Perspective crop follows the manually selected track from prior panorama detection results.",
        }
    elif args.follow_yolo:
        detections = detect_follow_angles(
            frames_dir=frames_dir,
            yolo_model_path=args.yolo_model,
            target_class=args.target_class,
            smoothing=args.follow_smoothing,
        )
        detections_path.write_text(
            json.dumps(detections, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        render_follow_view(
            frames_dir=frames_dir,
            output_dir=main_view_dir,
            size=args.size,
            fov=args.fov,
            detections=detections,
        )
        step1_meta = {
            "fps": args.extract_fps,
            "size": args.size,
            "fov": args.fov,
            "mode": "follow_yolo",
            "target_class": args.target_class,
            "yolo_model": args.yolo_model,
            "follow_smoothing": args.follow_smoothing,
            "detections_path": str(detections_path),
            "status": "done",
            "note": "Perspective crop follows the YOLO target center in the panorama.",
        }
    else:
        render_main_view(
            frames_dir=frames_dir,
            output_dir=main_view_dir,
            size=args.size,
            fov=args.fov,
            yaw=args.yaw,
            pitch=args.pitch,
        )
        step1_meta = {
            "fps": args.extract_fps,
            "size": args.size,
            "fov": args.fov,
            "mode": "manual",
            "yaw": args.yaw,
            "pitch": args.pitch,
            "status": "done",
            "note": "Manual main-view crop for validation.",
        }

    if not args.detect_all_yolo:
        create_preview_video(main_view_dir, preview_path, args.extract_fps)
    save_step1_metadata(output_root, args, step1_meta)

    run_unish_placeholder(output_root)
    run_gaussian_placeholder(output_root)
    align_worlds_placeholder(output_root)
    refine_dynamic_placeholder(output_root)

    if args.detect_all_yolo:
        print(f"Step 1 complete. Detection preview video: {overlay_preview_path}")
        print(f"Scaled inspection preview: {overlay_preview_small_path}")
    else:
        print(f"Step 1 complete. Preview video: {preview_path}")
    print(f"Pipeline state: {output_root / 'pipeline_state.json'}")


if __name__ == "__main__":
    main()
