"""
Check a YOLO dataset folder: image/label names, duplicates, and label-file problems.

Usage (PowerShell):
    python check_dataset.py
    python check_dataset.py --hash                 # also find images with identical content
    python check_dataset.py --also "C:\\path\\crop_one" "C:\\path\\background_one"
    python check_dataset.py --fix-duplicate-lines  # rewrite label files without repeated lines

What it checks
  1. images with no label file, and label files with no image
  2. duplicate names: same name twice (different case) or same stem with different extensions
  3. label files with repeated identical lines (the "duplicate labels removed" warning)
  4. malformed label lines (wrong number of values, class id not a number, values outside 0-1)
  5. empty label files (background images - fine, only counted)
  6. (--also) file names that also exist in other folders (the "same file name in train" error)
  7. (--hash) different file names with identical image content

A text report is written next to this script: dataset_check_report.txt
Nothing is changed unless you pass --fix-duplicate-lines (a .bak copy of each file is kept).
"""
import argparse
import hashlib
import shutil
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_ROOT = Path(r"C:\Users\EmageVision\Documents\head_training_dataset\dataset_15-10-2026")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_SHOW = 15                      # how many examples to print per problem


def list_files(folder, exts=None):
    if not folder.exists():
        sys.exit(f"folder not found: {folder}")
    files = [p for p in folder.iterdir() if p.is_file()]
    if exts is not None:
        files = [p for p in files if p.suffix.lower() in exts]
    return sorted(files)


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class Report:
    def __init__(self):
        self.lines = []
        self.problems = 0

    def out(self, text=""):
        print(text)
        self.lines.append(text)

    def section(self, title, items, show=MAX_SHOW):
        """items: list of strings. Prints OK or the count + first examples."""
        if not items:
            self.out(f"[ OK ] {title}: none")
            return
        self.problems += 1
        self.out(f"[FAIL] {title}: {len(items)}")
        for it in items[:show]:
            self.out(f"         {it}")
        if len(items) > show:
            self.out(f"         ... and {len(items) - show} more (see report file)")
        self.lines.extend(f"         {it}" for it in items[show:])


def check_label_file(path):
    """Return (repeated_lines, malformed_lines, n_objects, class_ids) for one label file."""
    seen, repeated, malformed, n_obj, classes = set(), [], [], 0, set()
    for n, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            malformed.append(f"line {n}: not numbers -> {line[:60]}")
            continue
        # detect: class + 4 values; segment: class + even number (>= 6) of polygon values
        ok_len = len(vals) == 4 or (len(vals) >= 6 and len(vals) % 2 == 0)
        if not ok_len:
            malformed.append(f"line {n}: {len(vals)} values after class id -> {line[:60]}")
            continue
        if any(v < 0 or v > 1 for v in vals):
            malformed.append(f"line {n}: value outside 0-1 -> {line[:60]}")
            continue
        if line in seen:
            repeated.append(n)
        seen.add(line)
        n_obj += 1
        classes.add(cls)
    return repeated, malformed, n_obj, classes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=str(DEFAULT_ROOT), help="dataset folder containing images/ and labels/")
    ap.add_argument("--also", nargs="*", default=[], help="other dataset folders (with images/) to compare names against")
    ap.add_argument("--hash", action="store_true", help="also look for identical image content (slower)")
    ap.add_argument("--fix-duplicate-lines", action="store_true", help="remove repeated lines from label files (keeps .bak)")
    args = ap.parse_args()

    root = Path(args.root)
    img_dir, lbl_dir = root / "images", root / "labels"
    r = Report()
    r.out(f"dataset: {root}")

    images = list_files(img_dir, IMG_EXT)
    labels = list_files(lbl_dir, {".txt"})
    other_in_images = [p for p in list_files(img_dir) if p.suffix.lower() not in IMG_EXT]
    r.out(f"images: {len(images)}   label files: {len(labels)}")
    r.out()

    img_by_stem = defaultdict(list)
    for p in images:
        img_by_stem[p.stem].append(p)
    lbl_by_stem = {p.stem: p for p in labels}

    # 1. missing pairs
    r.section("images WITHOUT a label file", [p.name for p in images if p.stem not in lbl_by_stem])
    r.section("label files WITHOUT an image", [p.name for p in labels if p.stem not in img_by_stem])
    r.section("non-image files inside images/", [p.name for p in other_in_images])

    # 2. duplicate names
    same_stem = [", ".join(x.name for x in v) for v in img_by_stem.values() if len(v) > 1]
    r.section("same stem with different extensions (e.g. a.jpg + a.png)", same_stem)
    lower = defaultdict(list)
    for p in images:
        lower[p.name.lower()].append(p.name)
    r.section("names that differ only by upper/lower case", [", ".join(v) for v in lower.values() if len(v) > 1])
    lower_l = defaultdict(list)
    for p in labels:
        lower_l[p.name.lower()].append(p.name)
    r.section("label names that differ only by upper/lower case", [", ".join(v) for v in lower_l.values() if len(v) > 1])

    # 3-5. label contents
    repeated_files, malformed_files, empty = [], [], 0
    total_obj, classes = 0, defaultdict(int)
    for p in labels:
        rep, bad, n_obj, cls = check_label_file(p)
        total_obj += n_obj
        for c in cls:
            classes[c] += 1
        if n_obj == 0 and not bad:
            empty += 1
        if rep:
            repeated_files.append((p, rep))
        if bad:
            malformed_files.append(f"{p.name}: " + "; ".join(bad[:3]))
    r.section("label files with REPEATED identical lines",
              [f"{p.name}: lines {rep}" for p, rep in repeated_files])
    r.section("label files with malformed lines", malformed_files)
    r.out(f"[info] empty label files (background images): {empty}")
    r.out(f"[info] objects total: {total_obj};  class ids found (files containing each): "
          + ", ".join(f"{c}: {n}" for c, n in sorted(classes.items())))

    if args.fix_duplicate_lines and repeated_files:
        for p, _ in repeated_files:
            shutil.copy2(p, p.with_suffix(".txt.bak"))
            seen, keep = set(), []
            for raw in p.read_text(errors="replace").splitlines():
                line = raw.strip()
                if line and line in seen:
                    continue
                seen.add(line)
                if line:
                    keep.append(line)
            p.write_text("\n".join(keep) + ("\n" if keep else ""))
        r.out(f"[done] removed repeated lines in {len(repeated_files)} label files (.bak copies kept)")

    # 6. names shared with other folders
    for other in args.also:
        o_dir = Path(other) / "images"
        if not o_dir.exists():
            r.out(f"[warn] {o_dir} not found, skipped")
            continue
        names_here = {p.name for p in images}
        clash = [p.name for p in list_files(o_dir, IMG_EXT) if p.name in names_here]
        r.section(f"file names ALSO present in {other}", clash)

    # 7. identical content
    if args.hash:
        r.out("hashing images (can take a while) ...")
        by_hash = defaultdict(list)
        for i, p in enumerate(images, 1):
            by_hash[md5(p)].append(p.name)
            if i % 1000 == 0:
                print(f"  hashed {i}/{len(images)}")
        r.section("different names, IDENTICAL image content",
                  [", ".join(v) for v in by_hash.values() if len(v) > 1])

    r.out()
    r.out("RESULT: " + ("all checks passed" if r.problems == 0 else f"{r.problems} check(s) need attention"))
    report = Path(__file__).with_name("dataset_check_report.txt")
    report.write_text("\n".join(r.lines), encoding="utf-8")
    print(f"report saved: {report}")


if __name__ == "__main__":
    main()