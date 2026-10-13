import cv2, numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\EmageVision\Documents\extracted_frames_29-09-2026")
IMG_DIR = BASE / "images"
LBL_DIR = BASE / "labels"
OUT = BASE / "label_check"
OUT.mkdir(exist_ok=True)

# 0 = head (green), 1 = tail (red)
COL = {0: (0, 255, 0), 1: (0, 0, 255)}
NAME = {0: "head", 1: "tail"}

n_ok = n_missing = 0
for ip in sorted(IMG_DIR.iterdir()):
    if ip.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
        continue
    lp = LBL_DIR / (ip.stem + ".txt")
    if not lp.exists():
        print("no label:", ip.name)
        n_missing += 1
        continue

    img = cv2.imread(str(ip)); h, w = img.shape[:2]
    for line in lp.read_text().splitlines():
        v = line.split()
        if not v:
            continue
        c = int(float(v[0])); xy = np.array(v[1:], float)
        pts = (xy.reshape(-1, 2) * [w, h]).astype(np.int32)
        col = COL.get(c, (255, 255, 255))
        cv2.polylines(img, [pts], True, col, 2)
        cv2.putText(img, NAME.get(c, str(c)), tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    cv2.imwrite(str(OUT / ip.name), img)
    n_ok += 1

print(f"Done: {n_ok} images drawn -> {OUT}  |  {n_missing} images without labels")