"""
Run the head/tail model on EVERY frame of a video and save:
  frames/frame_00000.jpg        annotated frame (boxes + class + confidence)
  detections.csv                one row per detection (frame, time, class, conf, box)
  per_frame_summary.csv         one row per frame (counts + avg/min/max conf per class)
  summary.csv                   overall averages for the whole video
  confidence_map.jpg            average confidence heatmap of where boxes appear
  box_density_map.jpg           how often each pixel was inside a box
  report.xlsx                   all three tables as Excel sheets
"""
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ---------------- settings ----------------
VIDEO = r"C:\Users\EmageVision\Downloads\session_2026-10-21_17-42-26.mp4"
MODEL = r"C:\Users\EmageVision\Videos\best_21-10-2026_named.pt"
OUT_DIR = r"C:\Users\EmageVision\Documents\detection_reports\session_2026-10-21_17-42-26"

CONF = 0.25          # minimum confidence to keep a detection
IOU = 0.5            # NMS IoU
AGNOSTIC_NMS = True  # merge overlapping HEAD/TAIL duplicates
DEVICE = 0           # 0 = GPU, "cpu" = CPU
JPG_QUALITY = 90
# ------------------------------------------

COLORS = {0: (255, 80, 0), 1: (0, 220, 255)}  # BGR: HEAD blue, TAIL yellow

frames_dir = os.path.join(OUT_DIR, "frames")
os.makedirs(frames_dir, exist_ok=True)

model = YOLO(MODEL)
names = model.names
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {VIDEO}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

conf_sum = np.zeros((H, W), np.float32)   # summed confidence inside boxes
hit_count = np.zeros((H, W), np.float32)  # number of boxes covering each pixel
last_frame = None

det_rows, frame_rows = [], []
idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    last_frame = frame
    t = idx / fps

    r = model.predict(frame, conf=CONF, iou=IOU, agnostic_nms=AGNOSTIC_NMS,
                      device=DEVICE, verbose=False)[0]
    vis = frame.copy()
    per_cls = {c: [] for c in names}

    for box, cf, cl in zip(r.boxes.xyxy.cpu().numpy(),
                           r.boxes.conf.cpu().numpy(),
                           r.boxes.cls.cpu().numpy().astype(int)):
        x1, y1, x2, y2 = box
        xi1, yi1 = max(0, int(x1)), max(0, int(y1))
        xi2, yi2 = min(W, int(x2)), min(H, int(y2))
        conf_sum[yi1:yi2, xi1:xi2] += cf
        hit_count[yi1:yi2, xi1:xi2] += 1
        per_cls[cl].append(float(cf))

        det_rows.append({
            "frame": idx, "time_s": round(t, 3), "class_id": cl,
            "class": names[cl], "confidence": round(float(cf), 4),
            "x1": round(float(x1), 1), "y1": round(float(y1), 1),
            "x2": round(float(x2), 1), "y2": round(float(y2), 1),
            "width": round(float(x2 - x1), 1), "height": round(float(y2 - y1), 1),
            "cx": round(float((x1 + x2) / 2), 1), "cy": round(float((y1 + y2) / 2), 1),
        })

        color = COLORS.get(cl, (0, 255, 0))
        cv2.rectangle(vis, (xi1, yi1), (xi2, yi2), color, 2)
        label = f"{names[cl]} {cf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(th + 4, yi1)
        cv2.rectangle(vis, (xi1, ty - th - 4), (xi1 + tw + 4, ty), color, -1)
        cv2.putText(vis, label, (xi1 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    row = {"frame": idx, "time_s": round(t, 3), "total_detections": len(r.boxes)}
    all_conf = [c for v in per_cls.values() for c in v]
    row["avg_conf_all"] = round(float(np.mean(all_conf)), 4) if all_conf else None
    for c, n in names.items():
        v = per_cls[c]
        row[f"{n}_count"] = len(v)
        row[f"{n}_avg_conf"] = round(float(np.mean(v)), 4) if v else None
        row[f"{n}_min_conf"] = round(float(np.min(v)), 4) if v else None
        row[f"{n}_max_conf"] = round(float(np.max(v)), 4) if v else None
    frame_rows.append(row)

    header = f"frame {idx}  t={t:.2f}s  dets={len(r.boxes)}" + \
             (f"  avg conf={row['avg_conf_all']:.2f}" if all_conf else "")
    cv2.putText(vis, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(vis, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(frames_dir, f"frame_{idx:05d}.jpg"), vis,
                [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])

    idx += 1
    if idx % 50 == 0:
        print(f"{idx}/{total} frames")

cap.release()

# ---------------- tables ----------------
det_df = pd.DataFrame(det_rows)
frame_df = pd.DataFrame(frame_rows)

summary = {"video": VIDEO, "model": MODEL, "frames": idx, "fps": round(fps, 2),
           "conf_threshold": CONF, "total_detections": len(det_df),
           "frames_with_detections": int((frame_df["total_detections"] > 0).sum()) if idx else 0,
           "avg_detections_per_frame": round(frame_df["total_detections"].mean(), 3) if idx else 0,
           "avg_conf_all": round(det_df["confidence"].mean(), 4) if len(det_df) else None}
for c, n in names.items():
    d = det_df[det_df["class_id"] == c] if len(det_df) else det_df
    summary[f"{n}_detections"] = len(d)
    summary[f"{n}_avg_per_frame"] = round(len(d) / idx, 3) if idx else 0
    summary[f"{n}_avg_conf"] = round(d["confidence"].mean(), 4) if len(d) else None
    summary[f"{n}_median_conf"] = round(d["confidence"].median(), 4) if len(d) else None
    summary[f"{n}_avg_box_w"] = round(d["width"].mean(), 1) if len(d) else None
    summary[f"{n}_avg_box_h"] = round(d["height"].mean(), 1) if len(d) else None
summary_df = pd.DataFrame(list(summary.items()), columns=["metric", "value"])

det_df.to_csv(os.path.join(OUT_DIR, "detections.csv"), index=False)
frame_df.to_csv(os.path.join(OUT_DIR, "per_frame_summary.csv"), index=False)
summary_df.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
try:
    with pd.ExcelWriter(os.path.join(OUT_DIR, "report.xlsx")) as xw:
        summary_df.to_excel(xw, sheet_name="summary", index=False)
        frame_df.to_excel(xw, sheet_name="per_frame", index=False)
        det_df.to_excel(xw, sheet_name="detections", index=False)
except ImportError:
    print("openpyxl not installed - skipped report.xlsx (pip install openpyxl)")

# ---------------- average maps ----------------
def save_heatmap(values, path, title):
    norm = values / values.max() if values.max() > 0 else values
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = last_frame if last_frame is not None else np.zeros((H, W, 3), np.uint8)
    out = cv2.addWeighted(base, 0.45, heat, 0.55, 0)
    cv2.putText(out, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(out, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 95])

avg_conf = np.divide(conf_sum, hit_count, out=np.zeros_like(conf_sum), where=hit_count > 0)
save_heatmap(avg_conf, os.path.join(OUT_DIR, "confidence_map.jpg"),
             f"average confidence inside boxes (max {avg_conf.max():.2f})")
save_heatmap(hit_count, os.path.join(OUT_DIR, "box_density_map.jpg"),
             f"box density over {idx} frames (max {int(hit_count.max())} hits)")

print(f"\nDone: {idx} frames, {len(det_df)} detections -> {OUT_DIR}")
print(summary_df.to_string(index=False))