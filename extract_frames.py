"""
Extract 1 frame per second from a video, save the RAW frame (no drawings),
and write YOLO-seg polygon labels predicted by the model.

Output layout (imports directly into CVAT as "Ultralytics YOLO Segmentation"):
    OUT_DIR/
        data.yaml
        images/train/<date>_frame00000.jpg
        labels/train/<date>_frame00000.txt
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- settings ----------------
VIDEO = r"C:\Users\EmageVision\Documents\head_training_dataset\videos\new_video\video_23-10-2026.mp4"
MODEL = r"C:\Users\EmageVision\Documents\head_training_dataset\priya_weights\tube_pt\epoch25_finetune_spark.pt"
OUT_DIR = r"C:\Users\EmageVision\Documents\extracted_frames\23-10-2026_video"

FRAMES_PER_SEC = 1      # how many frames to keep per second of video
CONF = 0.3              # minimum confidence to keep a prediction
IOU = 0.5               # NMS IoU
SIMPLIFY = 1.0          # polygon simplification in pixels (0 = keep every point)
DEVICE = 0              # 0 = GPU, "cpu" = CPU
# ------------------------------------------

img_dir = os.path.join(OUT_DIR, "images", "train")
lbl_dir = os.path.join(OUT_DIR, "labels", "train")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(lbl_dir, exist_ok=True)

model = YOLO(MODEL)
names = model.names  # {0: 'HEAD', 1: 'TAIL'}

date = input("Enter date (YYYY-MM-DD): ").strip()

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {VIDEO}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
every_n = max(1, int(round(fps / FRAMES_PER_SEC)))

i = saved = labelled = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    if i % every_n == 0:
        stem = f"{date}_frame{saved:05d}"
        cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), frame)  # raw frame

        r = model.predict(frame, conf=CONF, iou=IOU, agnostic_nms=True,
                          retina_masks=True, device=DEVICE, verbose=False)[0]
        h, w = frame.shape[:2]
        lines = []
        if r.masks is not None:
            for poly, cls in zip(r.masks.xy, r.boxes.cls.int().tolist()):
                if len(poly) < 3:
                    continue
                pts = poly.astype(np.float32)
                if SIMPLIFY > 0:
                    pts = cv2.approxPolyDP(pts.reshape(-1, 1, 2), SIMPLIFY, True).reshape(-1, 2)
                    if len(pts) < 3:
                        continue
                pts[:, 0] /= w
                pts[:, 1] /= h
                pts = np.clip(pts, 0, 1)
                coords = " ".join(f"{v:.6f}" for v in pts.flatten())
                lines.append(f"{cls} {coords}")

        # empty .txt = background frame (still useful for training)
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(lines))
        labelled += bool(lines)
        saved += 1

    i += 1

cap.release()

with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
    f.write(f"path: {OUT_DIR}\ntrain: images/train\nval: images/train\nnames:\n")
    for k in sorted(names):
        f.write(f"  {k}: {names[k]}\n")

print(f"Saved {saved} frames ({labelled} with labels) to {OUT_DIR}")