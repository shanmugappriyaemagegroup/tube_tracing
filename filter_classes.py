"""
filter_classes.py - keep only chosen class ids (default 0 and 1) in a YOLO-seg dataset.

Your originals are never touched. A new dataset is written to the --out folder:
  <out>/images/...   copy of every image (sub-folders such as train/val are kept)
  <out>/labels/...   same label files, with every line whose class id is not kept removed
  <out>/data.yaml    written only if images/train and images/val exist

Usage (Windows, from any terminal):
  python filter_classes.py
  python filter_classes.py --images "C:\\path\\images" --labels "C:\\path\\labels" --out "C:\\path\\dataset_01"
  python filter_classes.py --keep 0 1
"""
import argparse
import shutil
from collections import Counter
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_IMAGES = r"C:\Users\EmageVision\Documents\dataset\images"
DEFAULT_LABELS = r"C:\Users\EmageVision\Documents\dataset\labels"
DEFAULT_OUT = r"C:\Users\EmageVision\Documents\dataset_01"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=DEFAULT_IMAGES)
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--keep", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--names", nargs="+", default=["head", "tail"],
                    help="class names for data.yaml, in the same order as --keep")
    a = ap.parse_args()

    images = Path(a.images)
    labels = Path(a.labels)
    out = Path(a.out)
    keep = set(a.keep)

    if not images.is_dir() or not labels.is_dir():
        raise SystemExit(f"Folder not found:\n  images: {images}\n  labels: {labels}")
    if out.resolve() in (images.resolve(), labels.resolve()):
        raise SystemExit("--out must be a new folder, not your original images or labels.")
    if out.exists():
        shutil.rmtree(out)

    # ---- copy images ----
    n_img = 0
    for p in images.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            dest = out / "images" / p.relative_to(images)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            n_img += 1

    # ---- filter labels ----
    kept, removed = Counter(), Counter()
    n_files = n_emptied = 0
    for f in labels.rglob("*.txt"):
        if f.name.lower() == "classes.txt":
            continue
        lines_out = []
        had_lines = False
        for line in f.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            had_lines = True
            try:
                cid = int(float(parts[0]))
            except ValueError:
                continue
            if cid in keep:
                lines_out.append(line)
                kept[cid] += 1
            else:
                removed[cid] += 1
        dest = out / "labels" / f.relative_to(labels)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
        n_files += 1
        if had_lines and not lines_out:
            n_emptied += 1

    # ---- data.yaml (only when train/val split folders exist) ----
    wrote_yaml = False
    if (out / "images" / "train").is_dir() and (out / "images" / "val").is_dir():
        ids = sorted(keep)
        names = a.names if len(a.names) == len(ids) else [f"class_{i}" for i in ids]
        # ids are not remapped, so the yaml must cover 0..max(keep)
        name_map = {i: n for i, n in zip(ids, names)}
        lines = [f"path: {out.as_posix()}", "train: images/train", "val: images/val",
                 f"nc: {max(ids) + 1}", "names:"]
        for i in range(max(ids) + 1):
            lines.append(f"  {i}: {name_map.get(i, f'unused_{i}')}")
        (out / "data.yaml").write_text("\n".join(lines) + "\n")
        wrote_yaml = True

    print(f"images copied : {n_img}")
    print(f"label files   : {n_files}  (of which {n_emptied} had only removed classes and are now empty)")
    print(f"kept lines    : {dict(sorted(kept.items()))}")
    print(f"removed lines : {dict(sorted(removed.items()))}")
    print(f"output folder : {out}")
    if wrote_yaml:
        print(f"data.yaml     : {out / 'data.yaml'}")
    else:
        print("data.yaml     : not written (no images/train and images/val folders found)")


if __name__ == "__main__":
    main()