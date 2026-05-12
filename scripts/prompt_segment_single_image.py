import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Single-image text-prompt segmentation.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--prompt", required=True, help="Text prompt, e.g. 'girl'.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument(
        "--grounding_model",
        default="IDEA-Research/grounding-dino-tiny",
        help="GroundingDINO model id.",
    )
    parser.add_argument(
        "--sam_model",
        default="facebook/sam-vit-base",
        help="SAM model id.",
    )
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.20)
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Keep at most this many detections after thresholding.",
    )
    return parser.parse_args()


def ensure_prompt_suffix(prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt.endswith("."):
        prompt = prompt + "."
    return prompt


def to_xyxy_list(box_tensor: torch.Tensor):
    return [float(x) for x in box_tensor.detach().cpu().tolist()]


def color_for_index(index: int):
    colors = [
        (255, 99, 71),
        (65, 105, 225),
        (60, 179, 113),
        (238, 130, 238),
        (255, 215, 0),
    ]
    return colors[index % len(colors)]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)
    height, width = image_np.shape[:2]

    grounding_processor = AutoProcessor.from_pretrained(args.grounding_model)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.grounding_model
    ).to(device)
    grounding_model.eval()

    sam_processor = SamProcessor.from_pretrained(args.sam_model)
    sam_model = SamModel.from_pretrained(args.sam_model).to(device)
    sam_model.eval()

    text_prompt = ensure_prompt_suffix(args.prompt)
    inputs = grounding_processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = grounding_model(**inputs)

    results = grounding_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=[(height, width)],
    )[0]

    scores = results["scores"]
    labels = results["labels"]
    boxes = results["boxes"]

    if len(scores) == 0:
        (output_dir / "detections.json").write_text(
            json.dumps(
                {
                    "image": str(Path(args.image).resolve()),
                    "prompt": args.prompt,
                    "num_detections": 0,
                    "detections": [],
                },
                indent=2,
            )
        )
        cv2.imwrite(str(output_dir / "mask_union.png"), np.zeros((height, width), dtype=np.uint8))
        cv2.imwrite(str(output_dir / "overlay.png"), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        print("No detections found.")
        return

    order = torch.argsort(scores, descending=True)[: args.top_k]
    scores = scores[order]
    labels = [labels[i] for i in order.tolist()]
    boxes = boxes[order]

    union_mask = np.zeros((height, width), dtype=np.uint8)
    overlay = image_np.copy()
    per_detection = []

    for det_idx, (score, label, box_tensor) in enumerate(zip(scores, labels, boxes)):
        box = to_xyxy_list(box_tensor)
        sam_inputs = sam_processor(
            image,
            input_boxes=[[box]],
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            sam_outputs = sam_model(**sam_inputs)

        masks = sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )[0][0]
        iou_scores = sam_outputs.iou_scores[0, 0].detach().cpu()
        best_mask = masks[int(torch.argmax(iou_scores))].numpy() > 0
        union_mask[best_mask] = 255

        color = color_for_index(det_idx)
        overlay[best_mask] = (0.55 * overlay[best_mask] + 0.45 * np.array(color)).astype(np.uint8)
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            overlay,
            f"{label}: {float(score):.3f}",
            (x0, max(22, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

        det_mask = np.zeros((height, width), dtype=np.uint8)
        det_mask[best_mask] = 255
        det_mask_path = output_dir / f"mask_{det_idx:02d}.png"
        cv2.imwrite(str(det_mask_path), det_mask)

        per_detection.append(
            {
                "rank": det_idx,
                "label": str(label),
                "score": float(score),
                "box_xyxy": box,
                "mask_path": str(det_mask_path.resolve()),
            }
        )

    mask_union_path = output_dir / "mask_union.png"
    overlay_path = output_dir / "overlay.png"
    metadata_path = output_dir / "detections.json"

    cv2.imwrite(str(mask_union_path), union_mask)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    metadata_path.write_text(
        json.dumps(
            {
                "image": str(Path(args.image).resolve()),
                "prompt": args.prompt,
                "grounding_model": args.grounding_model,
                "sam_model": args.sam_model,
                "box_threshold": args.box_threshold,
                "text_threshold": args.text_threshold,
                "num_detections": len(per_detection),
                "detections": per_detection,
            },
            indent=2,
        )
    )

    print(f"Saved overlay to {overlay_path}")
    print(f"Saved union mask to {mask_union_path}")
    print(f"Saved detection metadata to {metadata_path}")
    print(f"Detections kept: {len(per_detection)}")


if __name__ == "__main__":
    main()
