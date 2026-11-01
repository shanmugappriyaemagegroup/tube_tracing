"""
add_background.py - add background (negative) images with EMPTY label files,
N per video (or per date), so the model learns what is NOT a head/tail
(fewer overkills / false positives).

Works with both frame-name styles:
    20261027_145107_441_video_frame02070   -> video 20261027_145107_441, date 2026-10-27
    2045-10-26_frame00000                  -> video 2045-10-26,          date 2045-10-26

For each video (or date) group:
  1. WHOLE backgrounds (default 3): full frames with no label file or an empty one,
     spread over the video
       -> bg_<stem>.jpg + bg_<stem>.txt (empty)
  2. CROPPED backgrounds (default 7): a window from frames spread over the video
     that does not touch any labelled object
       -> bgcrop_<stem>_<k>.jpg + .txt (empty)

Output: OUT_DIR/images and OUT_DIR/labels. Originals are never touched.
Add these to TRAIN only (never val).

    python add_background.py                       # 3 whole + 7 crops per video
    python add_background.py --whole 5 --crops 10
    python add_background.py --group-by date       # per date instead of per video
    python add_background.py --only-real           # whole frames only, no crops
"""
import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

DATASET = Path(r"C:\Users\EmageVision\Documents\dataset")
OUT_DIR = Path(r"C:\Users\EmageVision\Documents\background_one")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SKIP_PREFIX = ("cropped_", "zoom_in_", "bg_", "bgcrop_")

VIDEO_RE = re.compile(r"(\d{8}_\d{6}_\d{3})")                     # 20261027_145107_441
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}|(?<!\d)\d{8}(?!\d)")


def group_key(stem, by):
    s = stem.split("_frame")[0].rstrip("_")          # drop _frame02070
    if by == "video":
        m = VIDEO_RE.search(s)
        if m:
            return m.group(1)
        return re.sub(r"_(video|depth)$", "", s) or stem
    m = DATE_RE.search(s)
    if m:
        d = m.group(0)
        return f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d
    return s or stem


def spread(items, n):
    """n items spread evenly over the (sorted) list."""
    if len(items) <= n:
        return list(items)
    return [items[int(k * len(items) / n)] for k in range(n)]


def object_boxes(label_path, W, H):
    """Pixel boxes (x1, y1, x2, y2) of every labelled object (box or polygon)."""
    boxes = []
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        v = np.array(p[1:], dtype=float)
        if len(v) == 4:
            cx, cy, w, h = v
            boxes.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
        else:
            pts = v.reshape(-1, 2) * [W, H]
            boxes.append((*pts.min(0), *pts.max(0)))
    return boxes


def free_window(boxes, W, H, side, pad, tries=200):
    """Random side x side window that does not touch any box (+pad). None if not found."""
    side = int(min(side, W, H))
    for _ in range(tries):
        x1 = random.randint(0, W - side)
        y1 = random.randint(0, H - side)
        x2, y2 = x1 + side, y1 + side
        if all(x2 <= bx1 - pad or x1 >= bx2 + pad or y2 <= by1 - pad or y1 >= by2 + pad
               for bx1, by1, bx2, by2 in boxes):
            return x1, y1, x2, y2
    return None


def frame_no(p):
    m = re.search(r"_frame(\d+)", p.stem)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--group-by", choices=["video", "date"], default="video")
    ap.add_argument("--whole", type=int, default=3, help="whole-frame backgrounds per video/date (default 3)")
    ap.add_argument("--crops", type=int, default=7, help="cropped backgrounds per video/date (default 7)")
    ap.add_argument("--crop-size", type=int, default=640, help="side of a background crop in px (default 640)")
    ap.add_argument("--pad", type=int, default=40, help="min distance in px between a crop and any object")
    ap.add_argument("--only-real", action="store_true", help="only real empty frames, no background crops")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)

    root, out = Path(a.dataset), Path(a.out)
    img_in, lbl_in = root / "images", root / "labels"
    img_out, lbl_out = out / "images", out / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    files = [p for p in img_in.iterdir()
             if p.suffix.lower() in IMG_EXT
             and not p.stem.startswith(SKIP_PREFIX)
             and "_depth" not in p.stem]
    files.sort(key=lambda p: (group_key(p.stem, a.group_by), frame_no(p)))

    empty, labelled = defaultdict(list), defaultdict(list)
    for p in files:
        lp = lbl_in / f"{p.stem}.txt"
        has_obj = lp.exists() and lp.read_text().strip()
        (labelled if has_obj else empty)[group_key(p.stem, a.group_by)].append(p)

    total_real = total_crop = 0
    print(f"{'group':<24}{'frames':>8}{'empty':>7}{'real':>6}{'crops':>7}")
    for g in sorted(set(empty) | set(labelled)):
        real = spread(empty[g], a.whole)
        for p in real:
            shutil.copy2(p, img_out / f"bg_{p.stem}{p.suffix}")
            (lbl_out / f"bg_{p.stem}.txt").write_text("")
        need = 0 if a.only_real else a.crops

        made = 0
        # crops can come from any frame of the video (objects are avoided via the labels)
        source = sorted(labelled[g] + empty[g], key=frame_no)
        if need > 0 and source:
            # walk frames spread over the whole video, one crop per frame
            order = source
            step = max(1, len(order) // max(need, 1))
            # spread first, then the rest; if the video is short, a frame can give
            # more than one crop (different random window) on later passes
            first = order[::step]
            candidates = first + [p for p in order if p not in set(first)] + order * 3
            for p in candidates:
                if made >= need:
                    break
                img = cv2.imread(str(p))
                if img is None:
                    continue
                H, W = img.shape[:2]
                lp = lbl_in / f"{p.stem}.txt"
                boxes = object_boxes(lp, W, H) if lp.exists() else []
                win = free_window(boxes, W, H, a.crop_size, a.pad)
                if win is None:
                    continue
                x1, y1, x2, y2 = win
                name = f"bgcrop_{p.stem}_{made}"
                cv2.imwrite(str(img_out / f"{name}{p.suffix}"), img[y1:y2, x1:x2])
                (lbl_out / f"{name}.txt").write_text("")
                made += 1

        total_real += len(real)
        total_crop += made
        n_frames = len(empty[g]) + len(labelled[g])
        warn = []
        if len(real) < a.whole:
            warn.append(f"only {len(real)} empty frames")
        if made < need:
            warn.append(f"only {made} crops (no free area)")
        warn = ("   <- " + ", ".join(warn)) if warn else ""
        print(f"{g:<24}{n_frames:>8}{len(empty[g]):>7}{len(real):>6}{made:>7}{warn}")

    print(f"\ndone: {total_real} real background frames + {total_crop} background crops")
    print(f"  images -> {img_out}\n  labels -> {lbl_out} (empty .txt files)")


if __name__ == "__main__":
    main()