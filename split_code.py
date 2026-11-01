"""
Split a YOLO dataset into train / val (default 60 / 40).

Input: either layout works
    <src>\\images\\*.jpg + <src>\\labels\\*.txt
    <src>\\*.jpg + <src>\\*.txt              (images and labels side by side)

Output:
    <out>\\images\\train   <out>\\labels\\train
    <out>\\images\\val     <out>\\labels\\val

Usage:
    python split_dataset.py "C:\\path\\to\\src" "C:\\path\\to\\out"
    python split_dataset.py "C:\\path\\to\\src" "C:\\path\\to\\out" --train 0.6 --seed 42
    python split_dataset.py "C:\\path\\to\\src" "C:\\path\\to\\out" --move     # move instead of copy
    python split_dataset.py "C:\\path\\to\\src" "C:\\path\\to\\out" --dry-run
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    p = argparse.ArgumentParser(description="Train/val split for YOLO datasets.")
    p.add_argument("src", help="Dataset folder")
    p.add_argument("out", help="Output folder")
    p.add_argument("--train", type=float, default=0.6, help="Train ratio (default 0.6)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for a repeatable split")
    p.add_argument("--move", action="store_true", help="Move files instead of copying")
    p.add_argument("--dry-run", action="store_true", help="Show counts, change nothing")
    args = p.parse_args()

    src, out = Path(args.src), Path(args.out)
    img_dir = src / "images" if (src / "images").is_dir() else src
    lbl_dir = src / "labels" if (src / "labels").is_dir() else img_dir

    images = sorted(f for f in img_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"No images found in {img_dir}")

    random.Random(args.seed).shuffle(images)
    n_train = round(len(images) * args.train)
    splits = {"train": images[:n_train], "val": images[n_train:]}

    op = shutil.move if args.move else shutil.copy2
    no_label = 0
    for split, files in splits.items():
        if not args.dry_run:
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in files:
            lbl = lbl_dir / (img.stem + ".txt")
            if args.dry_run:
                no_label += not lbl.exists()
                continue
            op(str(img), str(out / "images" / split / img.name))
            dst_lbl = out / "labels" / split / lbl.name
            if lbl.exists():
                op(str(lbl), str(dst_lbl))
            else:
                dst_lbl.touch()          # no label -> treat as background
                no_label += 1

    tag = "[dry-run] " if args.dry_run else ""
    print(f"{tag}Total: {len(images)} | train: {len(splits['train'])} | val: {len(splits['val'])}")
    if no_label:
        print(f"{tag}{no_label} image(s) had no label file -> empty label (background)")
    print(f"\ndata.yaml paths:\n  train: {out / 'images' / 'train'}\n  val:   {out / 'images' / 'val'}")


if __name__ == "__main__":
    main()