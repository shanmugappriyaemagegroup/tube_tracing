"""
Build one row per frame with the frame's JPG path and every HEAD / TAIL confidence.
Uses the detections.csv already produced by video_detection_report.py (no re-run).

Output: frame_confidences.csv and frame_confidences.xlsx in the same report folder.
Columns: frame, time_s, image, HEAD_count, HEAD_1, HEAD_2, ..., HEAD_max, HEAD_avg,
         TAIL_count, TAIL_1, TAIL_2, ..., TAIL_max, TAIL_avg
"""
import os
import pandas as pd

REPORT_DIR = r"C:\Users\EmageVision\Documents\detection_reports\session_2026-10-21_17-42-26"
CLASSES = ["HEAD", "TAIL"]

det = pd.read_csv(os.path.join(REPORT_DIR, "detections.csv"))
frames = pd.read_csv(os.path.join(REPORT_DIR, "per_frame_summary.csv"))[["frame", "time_s"]]

out = frames.copy()
out["image"] = out["frame"].apply(
    lambda i: os.path.join(REPORT_DIR, "frames", f"frame_{i:05d}.jpg"))

for cls in CLASSES:
    d = det[det["class"] == cls].sort_values(["frame", "confidence"], ascending=[True, False])
    d = d.assign(rank=d.groupby("frame").cumcount() + 1)
    wide = d.pivot(index="frame", columns="rank", values="confidence")
    wide.columns = [f"{cls}_{c}" for c in wide.columns]
    stats = d.groupby("frame")["confidence"].agg(["count", "max", "mean"]).round(4)
    stats.columns = [f"{cls}_count", f"{cls}_max", f"{cls}_avg"]

    out = out.merge(stats[[f"{cls}_count"]], on="frame", how="left")
    out = out.merge(wide, on="frame", how="left")
    out = out.merge(stats[[f"{cls}_max", f"{cls}_avg"]], on="frame", how="left")
    out[f"{cls}_count"] = out[f"{cls}_count"].fillna(0).astype(int)

csv_path = os.path.join(REPORT_DIR, "frame_confidences.csv")
out.to_csv(csv_path, index=False)
try:
    out.to_excel(os.path.join(REPORT_DIR, "frame_confidences.xlsx"), index=False)
except ImportError:
    print("openpyxl not installed - skipped xlsx")

print(f"Saved {len(out)} rows -> {csv_path}")
print(out.head(5).to_string(index=False))