"""
remove_duplicates.py - remove duplicate / near-duplicate images (and their labels)
from a YOLO folder.

Each image gets a perceptual hash (dHash, 64 bit). Two images whose hashes differ in
<= --threshold bits are treated as duplicates; the first one (sorted by name) is kept.
Only images of the same type are compared (cropped_ with cropped_, zoom_in_ with
zoom_in_), so a crop and a zoom of the same frame are never counted as duplicates.

By default duplicates are MOVED (not deleted) to <folder>/duplicates/images and
/labels so you can check them first. Use --delete to delete them for real.

    python remove_duplicates.py                     # dry run: only lists what it would remove
    python remove_duplicates.py --move              # move duplicates to crop_one\duplicates
    python remove_duplicates.py --delete            # delete duplicates
    python remove_duplicates.py --threshold 0 --move    # only (almost) exact copies
    python remove_duplicates.py --threshold 8 --move    # also very similar frames
"""
import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\EmageVision\Documents\head_training_dataset\crop_one")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def dhash(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()          # 64 bools


def img_type(stem):
    for p in ("cropped_", "zoom_in_"):
        if stem.startswith(p):
            return p
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT), help="folder with images/ and labels/")
    ap.add_argument("--threshold", type=int, default=4,
                    help="max differing bits (of 64) to call two images duplicates (default 4)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--move", action="store_true", help="move duplicates to <root>/duplicates")
    g.add_argument("--delete", action="store_true", help="delete duplicates")
    a = ap.parse_args()

    root = Path(a.root)
    img_dir, lbl_dir = root / "images", root / "labels"
    dup_img, dup_lbl = root / "duplicates" / "images", root / "duplicates" / "labels"

    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    print(f"{len(files)} images in {img_dir}")

    kept = defaultdict(list)          # type -> list of hashes
    kept_names = defaultdict(list)
    dups = []                         # (duplicate, kept_original)
    for i, f in enumerate(files, 1):
        h = dhash(f)
        if h is None:
            print(f"  [skip] cannot read {f.name}")
            continue
        t = img_type(f.stem)
        if kept[t]:
            dist = np.count_nonzero(np.array(kept[t]) != h, axis=1)
            j = int(dist.argmin())
            if dist[j] <= a.threshold:
                dups.append((f, kept_names[t][j]))
                continue
        kept[t].append(h)
        kept_names[t].append(f.name)
        if i % 500 == 0:
            print(f"  hashed {i}/{len(files)}")

    print(f"\n{len(dups)} duplicates found, {len(files) - len(dups)} images kept")
    for f, orig in dups[:30]:
        print(f"  {f.name}  ==  {orig}")
    if len(dups) > 30:
        print(f"  ... and {len(dups) - 30} more")

    if not (a.move or a.delete):
        print("\nDry run - nothing changed. Add --move or --delete to remove them.")
        return

    if a.move:
        dup_img.mkdir(parents=True, exist_ok=True)
        dup_lbl.mkdir(parents=True, exist_ok=True)
    for f, _ in dups:
        lbl = lbl_dir / f"{f.stem}.txt"
        if a.move:
            shutil.move(str(f), str(dup_img / f.name))
            if lbl.exists():
                shutil.move(str(lbl), str(dup_lbl / lbl.name))
        else:
            f.unlink()
            if lbl.exists():
                lbl.unlink()
    print(f"\n{'Moved' if a.move else 'Deleted'} {len(dups)} images + their labels"
          + (f" -> {root / 'duplicates'}" if a.move else ""))


if __name__ == "__main__":
    main()