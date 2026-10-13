"""
build_head_tail_dataset.py - merge mixed-schema YOLO labels into a 2-class
grayscale dataset (0: head, 1: tail), dropping 'tubing'.

Source label schemas
  GRAY  (tube_tracing/dataset_gray):  0 head_blue, 1 head_black, 2 tail_open, 3 tubing
  TILES (head_tail_classification):   0 head,      1 tail,       2 tubing

Target
  0 head   <- head_blue, head_black, head
  1 tail   <- tail_open, tail
  (tubing lines are removed)

How each label file's schema is decided (from the label file only):
  contains class 3 (tubing) -> GRAY  (4-class)
  otherwise                 -> TILES (3-class)
  Files treated as TILES are listed in <out>_tiles_files.txt so you can check them.

Train/val split
  Split by VIDEO, not by frame: every frame with the same prefix before "_frame"
  (e.g. "2030-09-26") goes to the same side, so near-identical neighbouring frames
  never end up in both train and val. Whole videos are moved to val until about
  --val of all frames are there.

Usage (paths below are the defaults, so plain `python build_head_tail_dataset.py` works)
  python build_head_tail_dataset.py
  python build_head_tail_dataset.py --dry-run          # counts only, writes nothing
  python build_head_tail_dataset.py --val 0.2 --seed 1 # different videos in val
"""
import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

MAP_GRAY = {0: 0, 1: 0, 2: 1, 3: None}    # None = drop
MAP_TILES = {0: 0, 1: 1, 2: None}
NAMES = {0: "head", 1: "tail"}


def pick_schema(class_ids):
    # class 3 (tubing) only exists in the 4-class schema
    return "GRAY" if 3 in class_ids else "TILES"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=r"C:\Users\EmageVision\Documents\head_training_dataset\image")
    ap.add_argument("--labels", default=r"C:\Users\EmageVision\Documents\head_training_dataset\label")
    ap.add_argument("--out", default=r"C:\Users\EmageVision\Documents\head_training_dataset\dataset_headtail")
    ap.add_argument("--val", type=float, default=0.2, help="validation fraction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-empty", action="store_true",
                    help="keep images whose labels become empty (background images)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    img_dir, lbl_dir, out = Path(a.images), Path(a.labels), Path(a.out)
    images = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT)

    items, schema_count, cls_before, cls_after = [], Counter(), Counter(), Counter()
    missing, unreadable, emptied, bad = 0, 0, 0, 0
    tiles_files = []

    for ip in images:
        lp = lbl_dir / ip.relative_to(img_dir).with_suffix(".txt")
        if not lp.exists():
            lp = lbl_dir / (ip.stem + ".txt")
        if not lp.exists():
            missing += 1
            continue

        img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if img is None:
            unreadable += 1
            print(f"[skip] cannot read {ip}")
            continue

        rows = [ln.split() for ln in lp.read_text().splitlines() if ln.strip()]
        ids = {int(float(r[0])) for r in rows}
        schema = pick_schema(ids)
        if schema == "TILES":
            tiles_files.append(lp.name)
        mapping = MAP_GRAY if schema == "GRAY" else MAP_TILES
        schema_count[schema] += 1

        new_rows = []
        for r in rows:
            c = int(float(r[0]))
            cls_before[(schema, c)] += 1
            if c not in mapping:
                bad += 1
                print(f"[warn] {lp.name}: class {c} not in {schema} schema, line dropped")
                continue
            nc = mapping[c]
            if nc is None:
                continue
            cls_after[nc] += 1
            new_rows.append(" ".join([str(nc)] + r[1:]))   # works for boxes and polygons

        if not new_rows:
            emptied += 1
            if not a.keep_empty:
                continue
        items.append((ip, img, new_rows))

    print(f"\nimages found: {len(images)} | no label: {missing} | unreadable: {unreadable}")
    print(f"schema: {dict(schema_count)}")
    print("source classes:", {f'{s}:{c}': n for (s, c), n in sorted(cls_before.items())})
    print("output classes:", {NAMES[c]: n for c, n in sorted(cls_after.items())})
    print(f"images left with no head/tail: {emptied} "
          f"({'kept' if a.keep_empty else 'dropped'}) | unknown-class lines: {bad}")
    print(f"images to write: {len(items)}")
    tl = out.parent / (out.name + "_tiles_files.txt")
    tl.parent.mkdir(parents=True, exist_ok=True)
    tl.write_text("\n".join(tiles_files))
    print(f"{len(tiles_files)} label files treated as 3-class (no class 3) -> {tl}")
    if a.dry_run:
        return

    # ---- split by video: all frames with the same prefix go to the same side ----
    groups = {}
    for it in items:
        vid = it[0].stem.split("_frame")[0]          # "2030-09-26_frame00010" -> "2030-09-26"
        groups.setdefault(vid, []).append(it)
    vids = sorted(groups)
    random.Random(a.seed).shuffle(vids)

    target = len(items) * a.val
    val_vids, n = [], 0
    for v in vids:                                   # add whole videos until ~val fraction
        if n >= target:
            break
        val_vids.append(v)
        n += len(groups[v])
    if len(val_vids) == len(vids) and len(vids) > 1:  # never put everything in val
        val_vids.pop()

    splits = {"val":   [it for v in val_vids for it in groups[v]],
              "train": [it for v in vids if v not in val_vids for it in groups[v]]}
    print("\nvideos:", {v: len(groups[v]) for v in sorted(groups)})
    print("val videos  :", sorted(val_vids))
    print("train videos:", sorted(v for v in vids if v not in val_vids))

    if out.exists():
        shutil.rmtree(out)
    for split, group in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for ip, img, rows in group:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(out / "images" / split / ip.name),
                        cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
            (out / "labels" / split / (ip.stem + ".txt")).write_text(
                "\n".join(rows) + ("\n" if rows else ""))

    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n  0: head\n  1: tail\n")
    print(f"train {len(splits['train'])} / val {len(splits['val'])} -> {out}")


if __name__ == "__main__":
    main()