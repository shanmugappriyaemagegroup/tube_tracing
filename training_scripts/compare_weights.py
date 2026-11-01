"""
compare_weights.py - compare several YOLOv8-seg weights on the SAME validation set.

The val set is built exactly like train_finetune.py does it:
originals only (no cropped_/zoom_in_/bg_/bgcrop_, no _depth), grouped per video,
the LAST 20 % of each video's frames -> val.
So the numbers here are comparable with the training run.

    python compare_weights.py
    python compare_weights.py --weights a.pt b.pt c.pt --batch 8
"""
import argparse
import hashlib
import re
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

DATASET = "/home/wsw/Desktop/priya/dataset"
RUNS = "/home/wsw/Desktop/priya/training_runs"
WEIGHTS = [
    f"{RUNS}/finetune_oak/weights/epoch25.pt",
    f"{RUNS}/finetune_spark_seg_m/weights/best.pt",
    f"{RUNS}/finetune_spark_seg_m/weights/closest_gap.pt",
    f"{RUNS}/finetune_spark_seg_m/weights/_rolling/epoch9.pt",
    f"{RUNS}/finetune_spark_seg_m/weights/_rolling/epoch8.pt",
]
NAMES = ["head", "tail"]
VAL_FRAC = 0.20
IMGSZ = 1280
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SKIP_PREFIX = ("cropped_", "zoom_in_", "bg_", "bgcrop_")
VIDEO_RE = re.compile(r"(\d{8}_\d{6}_\d{3})")


def group_key(stem):
    s = stem.split("_frame")[0].rstrip("_")
    m = VIDEO_RE.search(s)
    if m:
        return m.group(1)
    return re.sub(r"_(video|depth)$", "", s) or stem


def frame_no(stem):
    m = re.search(r"_frame(\d+)", stem) or re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else 0


def build_val(dataset, out):
    img_in, lbl_in = dataset / "images", dataset / "labels"
    if not img_in.exists():
        raise SystemExit(f"not found: {img_in}")
    groups = defaultdict(list)
    for p in img_in.iterdir():
        if p.suffix.lower() in IMG_EXT and not p.stem.startswith(SKIP_PREFIX) and "_depth" not in p.stem:
            groups[group_key(p.stem)].append(p)
    val = []
    for items in groups.values():
        items.sort(key=lambda p: (frame_no(p.stem), p.stem))
        n_val = int(round(len(items) * VAL_FRAC)) if len(items) >= 5 else 0
        val += items[len(items) - n_val:]

    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)
    for p in val:
        shutil.copy2(p, out / "images" / p.name)
        lp = lbl_in / f"{p.stem}.txt"
        dst = out / "labels" / f"{p.stem}.txt"
        if lp.exists():
            shutil.copy2(lp, dst)
        else:
            dst.write_text("")
    data = out / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(out), "train": "images", "val": "images",
                                    "names": {i: n for i, n in enumerate(NAMES)}}, sort_keys=False))
    print(f"val set: {len(val)} original frames from {len(groups)} videos -> {out}")
    return data


def file_hash(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--weights", nargs="+", default=WEIGHTS)
    ap.add_argument("--out", default=None, help="where to build the val set (default: <dataset>/../compare_val)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    from ultralytics import YOLO

    dataset = Path(a.dataset)
    out = Path(a.out) if a.out else dataset.parent / "compare_val"
    data = build_val(dataset, out)

    rows, seen = [], {}
    for w in map(Path, a.weights):
        if not w.exists():
            print(f"[skip] not found: {w}")
            continue
        label = w.name if w.parent.name != "_rolling" else f"_rolling/{w.name}"
        h = file_hash(w)
        if h in seen:
            print(f"[same] {label} has the same weights as {seen[h]} - not run again")
            rows.append((label, None, seen[h]))
            continue
        seen[h] = label
        print(f"\nvalidating {label} ...")
        m = YOLO(str(w)).val(data=str(data), imgsz=IMGSZ, batch=a.batch, device=a.device,
                             plots=False, verbose=False)
        fit = (0.1 * m.box.map50 + 0.9 * m.box.map) + (0.1 * m.seg.map50 + 0.9 * m.seg.map)
        rows.append((label, (m.box.mp, m.box.mr, m.box.map50, m.box.map,
                             m.seg.map50, m.seg.map, fit), None))

    print("\n" + "=" * 100)
    print(f"{'weights':<28}{'box P':>7}{'box R':>7}{'box50':>8}{'box50-95':>10}"
          f"{'mask50':>8}{'mask50-95':>11}{'fitness':>9}")
    print("-" * 100)
    scored = [r for r in rows if r[1]]
    best = max(scored, key=lambda r: r[1][6])[0] if scored else None
    for label, v, same in rows:
        if v is None:
            print(f"{label:<28}  = same weights as {same}")
            continue
        P, R, b50, b, m50, mm, fit = v
        star = "  <- best" if label == best else ""
        print(f"{label:<28}{P:>7.3f}{R:>7.3f}{b50:>8.3f}{b:>10.3f}{m50:>8.3f}{mm:>11.3f}{fit:>9.3f}{star}")
    print("=" * 100)
    print("fitness = Ultralytics' own score: (0.1*mAP50 + 0.9*mAP50-95) for box + the same for mask")


if __name__ == "__main__":
    main()