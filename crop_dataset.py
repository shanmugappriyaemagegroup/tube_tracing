"""
crop_dataset.py - make CROPPED copies of a YOLO dataset (images + labels), N per video,
so the model sees heads/tails larger and with less background. No zoom_in_ copies.

Groups frames the same way as add_background.py:
    20261027_145107_441_video_frame02070   -> video 20261027_145107_441
    2045-10-26_frame00000                  -> 2045-10-26

For every picked frame it writes
    cropped_<stem>.jpg / .txt   tight crop around all labelled objects + margin
                                (native pixels, no resizing)
Labels are clipped and moved with the crop (boxes and polygons). An object that keeps
less than --min-keep of its area is dropped from that label file.

Output: OUT_DIR/images and OUT_DIR/labels. Originals are never touched.
Add these to TRAIN only (never val).

    python crop_dataset.py                    # 20 crops per video
    python crop_dataset.py --per-video 30
    python crop_dataset.py --per-video 0      # every labelled frame
    python crop_dataset.py --margin 0.4 --min-size 480
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

DATASET = Path(r"C:\Users\EmageVision\Documents\dataset")
OUT_DIR = Path(r"C:\Users\EmageVision\Documents\cropped_one")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SKIP_PREFIX = ("cropped_", "zoom_in_", "bg_", "bgcrop_")
VIDEO_RE = re.compile(r"(\d{8}_\d{6}_\d{3})")                     # 20261027_145107_441


# ------------------------------------------------------------------ grouping
def group_key(stem):
    s = stem.split("_frame")[0].rstrip("_")
    m = VIDEO_RE.search(s)
    if m:
        return m.group(1)
    return re.sub(r"_(video|depth)$", "", s) or stem


def frame_no(p):
    m = re.search(r"_frame(\d+)", p.stem)
    return int(m.group(1)) if m else 0


def spread(items, n):
    """n items spread evenly over the list (n = 0 -> all)."""
    if n <= 0 or len(items) <= n:
        return list(items)
    return [items[int(k * len(items) / n)] for k in range(n)]


# ---------------------------------------------------------------- label I/O
def read_labels(path, W, H):
    """-> [(class_id, Nx2 float array in PIXELS, is_box)]"""
    objs = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls, v = int(float(p[0])), np.array(p[1:], dtype=np.float64)
        if len(v) == 4:
            cx, cy, w, h = v
            x1, y1, x2, y2 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
            objs.append((cls, np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]), True))
        elif len(v) >= 6 and len(v) % 2 == 0:
            objs.append((cls, v.reshape(-1, 2) * [W, H], False))
    return objs


def poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clip_polygon(pts, x1, y1, x2, y2):
    """Sutherland-Hodgman clip of a polygon to the rectangle."""
    def clip(poly, inside, cross):
        out = []
        for i in range(len(poly)):
            a, b = poly[i - 1], poly[i]
            if inside(b):
                if not inside(a):
                    out.append(cross(a, b))
                out.append(b)
            elif inside(a):
                out.append(cross(a, b))
        return out

    def cx(x):
        return lambda a, b: np.array([x, a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0])])

    def cy(y):
        return lambda a, b: np.array([a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]), y])

    poly = [np.asarray(p, float) for p in pts]
    for inside, cross in ((lambda p: p[0] >= x1, cx(x1)), (lambda p: p[0] <= x2, cx(x2)),
                          (lambda p: p[1] >= y1, cy(y1)), (lambda p: p[1] <= y2, cy(y2))):
        if not poly:
            break
        poly = clip(poly, inside, cross)
    return np.array(poly) if len(poly) >= 3 else None


def labels_for_window(objs, x1, y1, x2, y2, min_keep):
    """Clip objects to the window, move to its origin, normalise -> label lines."""
    w, h = x2 - x1, y2 - y1
    lines = []
    for cls, pts, is_box in objs:
        full = poly_area(pts)
        c = clip_polygon(pts, x1, y1, x2, y2)
        if c is None or full <= 0 or poly_area(c) < min_keep * full:
            continue
        c = np.clip((c - [x1, y1]) / [w, h], 0, 1)
        if is_box:
            bx1, by1 = c.min(0)
            bx2, by2 = c.max(0)
            vals = [(bx1 + bx2) / 2, (by1 + by2) / 2, bx2 - bx1, by2 - by1]
        else:
            vals = c.reshape(-1)
        lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in vals))
    return lines


# -------------------------------------------------------------------- window
def crop_window(objs, W, H, margin, min_size):
    """Tight crop around all objects + margin, at least min_size per side, kept inside."""
    allp = np.vstack([p for _, p, _ in objs])
    bx1, by1 = allp.min(0)
    bx2, by2 = allp.max(0)
    w = min(max((bx2 - bx1) * (1 + 2 * margin), min_size), W)
    h = min(max((by2 - by1) * (1 + 2 * margin), min_size), H)
    x1 = min(max((bx1 + bx2) / 2 - w / 2, 0), W - w)
    y1 = min(max((by1 + by2) / 2 - h / 2, 0), H - h)
    return int(round(x1)), int(round(y1)), int(round(x1 + w)), int(round(y1 + h))


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(DATASET), help="folder with images/ and labels/")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--per-video", type=int, default=20, help="crops per video, spread evenly (0 = all)")
    ap.add_argument("--margin", type=float, default=0.25, help="space around the objects, fraction of their size")
    ap.add_argument("--min-size", type=int, default=640, help="smallest crop side in px")
    ap.add_argument("--min-keep", type=float, default=0.4, help="drop an object keeping less than this of its area")
    a = ap.parse_args()

    root, out = Path(a.dataset), Path(a.out)
    img_in, lbl_in = root / "images", root / "labels"
    img_out, lbl_out = out / "images", out / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for p in img_in.iterdir():
        if p.suffix.lower() not in IMG_EXT or p.stem.startswith(SKIP_PREFIX) or "_depth" in p.stem:
            continue
        lp = lbl_in / f"{p.stem}.txt"
        if lp.exists() and lp.read_text().strip():          # only frames with objects
            groups[group_key(p.stem)].append(p)

    total = 0
    print(f"{'video':<24}{'labelled':>9}{'crops':>7}")
    for g in sorted(groups):
        frames = sorted(groups[g], key=frame_no)
        made = 0
        for p in spread(frames, a.per_video):
            img = cv2.imread(str(p))
            if img is None:
                continue
            H, W = img.shape[:2]
            objs = read_labels(lbl_in / f"{p.stem}.txt", W, H)
            if not objs:
                continue
            x1, y1, x2, y2 = crop_window(objs, W, H, a.margin, a.min_size)
            lines = labels_for_window(objs, x1, y1, x2, y2, a.min_keep)
            if not lines:
                continue
            name = f"cropped_{p.stem}"
            cv2.imwrite(str(img_out / f"{name}{p.suffix}"), img[y1:y2, x1:x2])
            (lbl_out / f"{name}.txt").write_text("\n".join(lines) + "\n")
            made += 1
        total += made
        print(f"{g:<24}{len(frames):>9}{made:>7}")

    print(f"\ndone: {total} cropped images")
    print(f"  images -> {img_out}\n  labels -> {lbl_out}")


if __name__ == "__main__":
    main()