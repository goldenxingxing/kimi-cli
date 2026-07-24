#!/usr/bin/env python3
"""
Grounding DINO + SAM Segmentation & Background Replace Skill

Pipeline:
  1. Grounding DINO detects bounding boxes from text prompt
  2. SAM (Segment Anything) generates precise pixel masks using those boxes
  3. Background replaced with specified color (or transparent)
  4. Output cropped to the tightest bounding rect of the subject mask
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
from PIL import Image

try:
    import torch
    from transformers import (
        AutoProcessor,
        AutoModelForZeroShotObjectDetection,
        SamModel,
        SamProcessor,
    )
except Exception:
    torch = None
    AutoProcessor = None
    AutoModelForZeroShotObjectDetection = None
    SamModel = None
    SamProcessor = None

_DINO_CACHE: Dict[str, Any] = {}
_SAM_CACHE: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(args=None):
    p = argparse.ArgumentParser(
        description=(
            "Grounding DINO + SAM: detect subject by text, segment it, "
            "replace background with a solid color, and crop to the subject."
        )
    )
    p.add_argument("--image", "-i", required=True, help="Input image path")
    p.add_argument(
        "--text", "-t", required=True,
        help="Subject description, e.g. 'a cat' or 'the main product'",
    )
    p.add_argument("--output", "-o", required=True, help="Output image path")
    p.add_argument(
        "--bg-color", type=str, default="255,255,255",
        help=(
            'Background color as R,G,B or "transparent". '
            'e.g. "255,255,255" for white, "0,0,0" for black, "transparent" for alpha'
        ),
    )
    p.add_argument(
        "--no-crop", action="store_true", default=False,
        help="Skip tight-crop step; output full-size image with replaced background",
    )
    p.add_argument(
        "--padding", type=int, default=0,
        help="Extra pixel padding around the cropped subject (default 0)",
    )
    p.add_argument(
        "--multi", action="store_true", default=False,
        help="Process all detected boxes (union mask). Default: best detection only.",
    )
    p.add_argument("--threshold", type=float, default=0.3, help="DINO detection threshold")
    p.add_argument("--text-threshold", type=float, default=0.25, help="DINO text match threshold")
    p.add_argument(
        "--dino-model", type=str, default="IDEA-Research/grounding-dino-tiny",
        help="Grounding DINO HuggingFace model ID",
    )
    p.add_argument(
        "--sam-model", type=str, default="facebook/sam-vit-base",
        help="SAM HuggingFace model ID",
    )
    p.add_argument("--device", type=str, default="auto", help="Device: auto / cpu / cuda / mps")
    p.add_argument("--json-output", type=str, default=None, help="Path to write JSON metadata")
    return p.parse_args(args)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_bg_color(color_str: str) -> Optional[Tuple[int, int, int]]:
    """Return (R, G, B) or None for transparent."""
    s = color_str.strip().lower()
    if s in ("transparent", "none", "alpha"):
        return None
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"bg_color must be 'R,G,B' or 'transparent', got: {color_str}")
    return tuple(np.clip(parts, 0, 255).tolist())


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_dino(model_id: str, device: str) -> Dict[str, Any]:
    key = f"{model_id}@{device}"
    if key in _DINO_CACHE:
        return _DINO_CACHE[key]
    if torch is None:
        raise RuntimeError("PyTorch / transformers not installed.")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    entry = {"processor": processor, "model": model, "device": device}
    _DINO_CACHE[key] = entry
    return entry


def load_sam(model_id: str, device: str) -> Dict[str, Any]:
    key = f"{model_id}@{device}"
    if key in _SAM_CACHE:
        return _SAM_CACHE[key]
    if torch is None:
        raise RuntimeError("PyTorch / transformers not installed.")
    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id).to(device)
    model.eval()
    entry = {"processor": processor, "model": model, "device": device}
    _SAM_CACHE[key] = entry
    return entry


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_objects(
    image: Image.Image,
    text_prompt: str,
    dino: Dict[str, Any],
    threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    processor, model, device = dino["processor"], dino["model"], dino["device"]
    w, h = image.size
    inputs = processor(images=image, text=[[text_prompt]], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[(h, w)],
    )[0]

    boxes = results.get("boxes", torch.empty(0)).cpu().numpy()
    scores = results.get("scores", torch.empty(0)).cpu().numpy()
    text_labels = results.get("text_labels", [])

    detections = [
        {
            "box": [float(v) for v in box.tolist()],
            "score": float(scores[i]),
            "label": text_labels[i] if i < len(text_labels) else text_prompt,
        }
        for i, box in enumerate(boxes)
    ]
    detections.sort(key=lambda x: x["score"], reverse=True)
    return detections


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def clamp_box(box: List[float], w: int, h: int) -> List[int]:
    x0, y0, x1, y1 = box
    x0 = max(0, min(w, int(round(x0))))
    y0 = max(0, min(h, int(round(y0))))
    x1 = max(0, min(w, int(round(x1))))
    y1 = max(0, min(h, int(round(y1))))
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def segment_with_sam(
    image: Image.Image,
    boxes: List[List[float]],
    sam: Dict[str, Any],
) -> np.ndarray:
    """
    Run SAM with box prompts. Returns a single merged binary mask (H, W) bool.
    Each box yields one mask; all masks are OR-merged.
    """
    processor, model, device = sam["processor"], sam["model"], sam["device"]
    w, h = image.size
    clamped = [clamp_box(b, w, h) for b in boxes]

    # SAM processor expects input_boxes as [[[x0,y0,x1,y1], ...]] per image
    inputs = processor(
        images=image,
        input_boxes=[[[b] for b in clamped]],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # post_process_masks returns list[list[tensor(3,H,W)]] – one per image, per box
    masks_per_image = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    # masks_per_image[0] shape: (num_boxes, 3, H, W)  (3 candidate masks per box)
    # We pick the candidate with the highest iou_score for each box.
    iou_scores = outputs.iou_scores.cpu()  # (1, num_boxes, 3)
    iou_scores = iou_scores[0]  # (num_boxes, 3)

    masks_tensor = masks_per_image[0]  # (num_boxes, 3, H, W)
    merged = np.zeros((h, w), dtype=bool)

    for box_idx in range(masks_tensor.shape[0]):
        best_candidate = int(iou_scores[box_idx].argmax())
        mask = masks_tensor[box_idx, best_candidate].numpy().astype(bool)
        merged |= mask

    return merged


# ---------------------------------------------------------------------------
# Compose output
# ---------------------------------------------------------------------------

def apply_mask_and_replace_bg(
    image: Image.Image,
    mask: np.ndarray,
    bg_color: Optional[Tuple[int, int, int]],
) -> Image.Image:
    """
    Replace pixels where mask==False with bg_color.
    If bg_color is None, output is RGBA with transparent background.
    """
    img_arr = np.array(image.convert("RGBA"))
    alpha_channel = (mask * 255).astype(np.uint8)

    if bg_color is None:
        img_arr[:, :, 3] = alpha_channel
        return Image.fromarray(img_arr, "RGBA")
    else:
        bg = np.array(bg_color, dtype=np.uint8)
        result = img_arr[:, :, :3].copy()
        result[~mask] = bg
        return Image.fromarray(result, "RGB")


def tight_crop(
    image: Image.Image,
    mask: np.ndarray,
    padding: int = 0,
) -> Image.Image:
    """Crop image to the bounding box of the True region in mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return image  # mask is empty, return as-is
    y0, y1 = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))

    h, w = mask.shape
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(w, x1 + 1 + padding)
    y1 = min(h, y1 + 1 + padding)

    return image.crop((x0, y0, x1, y1))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def seg_and_replace(
    image_path: str,
    text: str,
    output_path: str,
    bg_color: Union[Tuple[int, int, int], None] = (255, 255, 255),
    crop: bool = True,
    padding: int = 0,
    multi: bool = False,
    threshold: float = 0.3,
    text_threshold: float = 0.25,
    dino_model_id: str = "IDEA-Research/grounding-dino-tiny",
    sam_model_id: str = "facebook/sam-vit-base",
    device: str = "auto",
) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch / transformers not installed.")
    device = resolve_device(device)

    image = Image.open(image_path).convert("RGB")
    dino = load_dino(dino_model_id, device)
    sam = load_sam(sam_model_id, device)

    detections = detect_objects(image, text, dino, threshold, text_threshold)
    if not detections:
        raise RuntimeError(f"No objects detected for: '{text}'")

    target_dets = detections if multi else detections[:1]
    boxes = [d["box"] for d in target_dets]

    mask = segment_with_sam(image, boxes, sam)
    result_img = apply_mask_and_replace_bg(image, mask, bg_color)

    if crop:
        result_img = tight_crop(result_img, mask, padding)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    result_img.save(output_path)

    return {
        "success": True,
        "output_path": output_path,
        "output_size": result_img.size,
        "detections": target_dets,
        "mask_coverage": float(mask.mean()),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    bg_color = parse_bg_color(args.bg_color)
    device = resolve_device(args.device)

    try:
        meta = seg_and_replace(
            image_path=args.image,
            text=args.text,
            output_path=args.output,
            bg_color=bg_color,
            crop=not args.no_crop,
            padding=args.padding,
            multi=args.multi,
            threshold=args.threshold,
            text_threshold=args.text_threshold,
            dino_model_id=args.dino_model,
            sam_model_id=args.sam_model,
            device=device,
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    print(
        f"Saved: {meta['output_path']} "
        f"size={meta['output_size']} "
        f"mask_coverage={meta['mask_coverage']:.3f}"
    )


if __name__ == "__main__":
    main()
