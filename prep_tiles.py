"""
prep_tiles.py - cut a YOLO-seg dataset into overlapping tiles around the labels.

Your ends are 10-20 px inside a ~1920 px photo. Resized to imgsz=1024 they are
6-10 px, and YOLO's mask head works at a quarter of that, so barely 2 px of mask
survives. Cutting 640 px tiles around each label and training on those makes the
same objects 3-5x larger relative to the image, which is the single biggest
accuracy win available without new photos.

Tiles that contain no label are skipped, except for a random fraction (--bg)
that is kept with an empty label file as background, which reduces false
positives. A polygon clipped by a tile border is kept only if `--min-keep` of
its area is still inside, so half objects do not teach the model wrong shapes.

  python prep_tiles.py --src dataset_01 --dst dataset_tiles --tile 640 --overlap 0.3 --bg 0.1

Train on the tiles at the tile size:
  yolo segment train model=yolo11s-seg.pt data=tiles.yaml imgsz=640 ...
and run inference with measure_from_masks.py --tile 640.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def polys_of(txt, w, h):
    out = []
    if not Path(txt).exists():
        return out
    for ln in Path(txt).read_text().splitlines():
        v = ln.split()
        if len(v) < 7:
            continue
        pts = np.array(v[1:], np.float32).reshape(-1, 2) * [w, h]
        out.append((int(v[0]), pts))
    return out


def clip(poly, x0, y0, tile, min_keep):
    """Rasterise the polygon, crop to the tile, and read the contour back."""
    x1, y1 = poly[:, 0].min(), poly[:, 1].min()
    x2, y2 = poly[:, 0].max(), poly[:, 1].max()
    if x2 < x0 or x1 > x0 + tile or y2 < y0 or y1 > y0 + tile:
        return []
    pad = 4
    ox, oy = int(x1) - pad, int(y1) - pad
    W, H = int(x2 - x1) + 2 * pad, int(y2 - y1) + 2 * pad
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [(poly - [ox, oy]).round().astype(np.int32)], 1)
    full_area = int(m.sum())
    if full_area < 9:
        return []
    keep = np.zeros_like(m)
    kx0, ky0 = max(x0 - ox, 0), max(y0 - oy, 0)
    kx1, ky1 = min(x0 + tile - ox, W), min(y0 + tile - oy, H)
    if kx1 <= kx0 or ky1 <= ky0:
        return []
    keep[ky0:ky1, kx0:kx1] = 1
    m = m * keep
    if m.sum() < min_keep * full_area:
        return []
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        if cv2.contourArea(c) < 6:
            continue
        p = c.reshape(-1, 2).astype(np.float32) + [ox - x0, oy - y0]
        if len(p) >= 3:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dataset root with images/ and labels/")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.3)
    ap.add_argument("--min-keep", type=float, default=0.6,
                    help="fraction of a polygon that must stay inside the tile")
    ap.add_argument("--bg", type=float, default=0.1,
                    help="fraction of empty tiles to keep as background")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    np.random.seed(a.seed)

    src, dst = Path(a.src), Path(a.dst)
    step = max(int(a.tile * (1 - a.overlap)), 32)
    made = kept = bg = 0
    for img in sorted(src.rglob("*")):
        if img.suffix.lower() not in IMG_EXT or "images" not in img.parts:
            continue
        rel = img.relative_to(src)
        lbl = src / str(rel).replace("images", "labels", 1)
        lbl = lbl.with_suffix(".txt")
        im = cv2.imread(str(img))
        if im is None:
            continue
        h, w = im.shape[:2]
        polys = polys_of(lbl, w, h)
        if not polys:
            continue
        xs = list(range(0, max(w - a.tile, 0) + 1, step)) or [0]
        ys = list(range(0, max(h - a.tile, 0) + 1, step)) or [0]
        if xs[-1] + a.tile < w:
            xs.append(w - a.tile)
        if ys[-1] + a.tile < h:
            ys.append(h - a.tile)
        for yi in ys:
            for xi in xs:
                lines = []
                for cls, p in polys:
                    for q in clip(p, xi, yi, a.tile, a.min_keep):
                        q = np.clip(q, 0, a.tile - 1) / a.tile
                        lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in q.reshape(-1)))
                made += 1
                if not lines and np.random.rand() > a.bg:   # keep a fraction of empty tiles
                    continue
                if not lines:
                    bg += 1
                crop = im[yi:yi + a.tile, xi:xi + a.tile]
                if crop.shape[0] != a.tile or crop.shape[1] != a.tile:
                    crop = cv2.copyMakeBorder(crop, 0, a.tile - crop.shape[0],
                                              0, a.tile - crop.shape[1],
                                              cv2.BORDER_CONSTANT, value=(114, 114, 114))
                name = f"{img.stem}_x{xi}_y{yi}"
                oi = dst / rel.parent / f"{name}{img.suffix}"
                ol = dst / str(rel.parent).replace("images", "labels", 1) / f"{name}.txt"
                oi.parent.mkdir(parents=True, exist_ok=True)
                ol.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(oi), crop)
                ol.write_text("\n".join(lines) + ("\n" if lines else ""))
                kept += 1
    print(f"{kept} tiles written to {dst} ({kept - bg} with labels, {bg} background) "
          f"out of {made} candidate tiles")


if __name__ == "__main__":
    main()
