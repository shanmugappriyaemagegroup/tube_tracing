"""
pseudo_label.py - use your TRAINED model to auto-label new, unlabeled frames
as YOLO-seg polygons, ready to import into CVAT for correction. Nothing is
passed on the command line - edit CONFIG below and run:

    python pseudo_label.py

This is the "bootstrap loop": hand-label a first batch, train, then let the
model label the NEXT batch for you - you only correct mistakes instead of
drawing from scratch, and each round gets faster as the model improves.

  mask (from the trained model, tiled the same way as training)
      -> contour (cv2.findContours)
      -> simplified polygon (cv2.approxPolyDP - raw contours can have
         hundreds of points, which makes them painful to edit by hand in
         CVAT; this keeps the shape but drops redundant points)
      -> normalised YOLO-seg line, written with the class id straight from
         your model (already in your data.yaml's 0=head/1=tail/2=tubing
         order - no name-matching needed here, unlike measure_from_masks.py,
         because we are WRITING labels in that exact scheme, not reading an
         unknown one)

Output is a full images/ + labels/ pair, self-contained and ready to:
  - zip and import into CVAT as pre-annotations (Upload annotations ->
    Ultralytics YOLO Segmentation) onto a task with the same images, or
  - drop straight into your training dataset after you've corrected it in
    CVAT and exported again.

Images with ZERO detections are printed separately at the end - these are
exactly the frames most likely to need your attention (an unusual angle, a
tube type the model hasn't seen, poor lighting), so check those first.
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from measure_from_masks import masks_from_model

# ============================== CONFIG — edit these ==============================
MODEL_PATH = r"C:\Users\EmageVision\Documents\head_tail_classification\runs\segment\train\weights\best.pt"
IMAGES_DIR = r"C:\Users\EmageVision\Documents\head_tail_classification\extracted_frames"

OUT_DIR = r"C:\Users\EmageVision\Documents\head_tail_classification\pseudo_labeled"
COPY_IMAGES = True     # also copy the source images into OUT_DIR/images - makes
                       # the output folder self-contained and ready to zip for CVAT

DEVICE = "0"           # "0" = first GPU, "cpu" = CPU
CONF = 0.40            # higher than normal inference: better to MISS a weak
                       # detection (you draw it) than write a wrong one (you
                       # have to notice AND delete it - drawing is the safer default)
TILE = 640             # must match training
TILE_OVERLAP = 0.3

MIN_CONTOUR_AREA = 20  # px^2 - drop tiny noise fragments before writing a label
SIMPLIFY_FRACTION = 0.004   # of the contour's perimeter; higher = fewer points,
                           # coarser polygon. 0 disables simplification.
# ===================================================================================


def mask_to_yolo_lines(cls_id, mask, w, h):
    """One mask -> zero or more 'cls x1 y1 x2 y2 ...' lines (normalised 0..1).
    A mask can produce more than one contour if the model's prediction has a
    gap in it; each becomes its own polygon line, same class id."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in contours:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            continue
        if SIMPLIFY_FRACTION > 0:
            eps = SIMPLIFY_FRACTION * cv2.arcLength(c, True)
            c = cv2.approxPolyDP(c, eps, True)
        if len(c) < 3:
            continue
        pts = c.reshape(-1, 2).astype(np.float64)
        pts[:, 0] = np.clip(pts[:, 0] / w, 0, 1)
        pts[:, 1] = np.clip(pts[:, 1] / h, 0, 1)
        lines.append(f"{cls_id} " + " ".join(f"{v:.6f}" for v in pts.reshape(-1)))
    return lines


def main():
    model = YOLO(MODEL_PATH)
    if DEVICE:
        model.to(f"cuda:{DEVICE}" if DEVICE.isdigit() else DEVICE)
    print(f"[device] model is running on: {model.device}")
    print(f"[classes] {model.names}  (label ids written as-is, matching your data.yaml)")

    out = Path(OUT_DIR)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    if COPY_IMAGES:
        (out / "images").mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in Path(IMAGES_DIR).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not paths:
        raise SystemExit(f"no images found in {IMAGES_DIR}")

    counts = {name: 0 for name in model.names.values()}
    empty_images = []

    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"[skip] cannot read {p.name}")
            continue
        h, w = bgr.shape[:2]
        instances = masks_from_model(model, bgr, CONF, True, TILE, TILE_OVERLAP)

        lines = []
        for cls_id, mask in instances:
            new_lines = mask_to_yolo_lines(cls_id, mask, w, h)
            lines.extend(new_lines)
            counts[model.names[cls_id]] += len(new_lines)

        (out / "labels" / f"{p.stem}.txt").write_text("\n".join(lines))
        if COPY_IMAGES:
            shutil.copy2(p, out / "images" / p.name)
        if not lines:
            empty_images.append(p.name)

        print(f"{p.name}: {len(lines)} polygon(s)")

    print(f"\n[summary] {len(paths)} images -> {out}")
    for name, n in counts.items():
        print(f"  {name:10s} {n}")
    if empty_images:
        print(f"\n[review first] {len(empty_images)} image(s) got NO detections at all:")
        for name in empty_images:
            print(f"  {name}")
    print(f"\nNext: open {out} in CVAT (Upload annotations -> Ultralytics YOLO "
         "Segmentation onto a task with these images, or create a new task from "
         f"{out}/images and import {out}/labels the same way) and correct the "
         "polygons - focus on the flagged images first.")


if __name__ == "__main__":
    main()