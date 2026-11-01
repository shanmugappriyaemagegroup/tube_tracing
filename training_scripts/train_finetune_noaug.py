"""
train_finetune_noaug.py - fine-tune a YOLOv8-m SEGMENTATION model (epoch25_finetune_spark.pt)
on E:\\head_tail_priya\\dataset with NO AUGMENTATION.

The model sees every training image exactly as it is on disk:
  * no albumentations (no brightness/noise/blur/compression ...)
  * Ultralytics' own augmentation switched off: no mosaic, no flips, no rotation, no scale,
    no translate, no shear, no HSV colour change, no copy-paste, no mixup
Everything else is the same as train_fine_tuning1.py (lr 0.0002, 40 epochs, patience 10,
batch 8, imgsz 1280, every epoch kept, loss-match + comparison + winner).

Data:
  * originals: last 20 % of each video -> val (ORIGINALS ONLY), the rest -> train
  * cropped_one/ zoom_one/ background_one/ -> train only (as before)
  * --originals-only : leave the crop/zoom/background images out as well (only the
    original frames in train)

    python train_finetune_noaug.py --split-only
    python train_finetune_noaug.py                   # originals + crops/zooms/backgrounds
    python train_finetune_noaug.py --originals-only  # originals only
    python train_finetune_noaug.py --compare-only
"""
import argparse
import csv
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(_here, "ultralytics")) and not os.path.exists(os.path.join(_here, "ultralytics", "__init__.py")):
    sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _here]
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256,garbage_collection_threshold:0.8")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import yaml

# ------------------------------------------------------------------ paths / settings
DOCS = Path(r"E:\head_tail_priya")
START = DOCS / "training_runs" / "weights" / "epoch25_finetune_spark.pt"
DATASET = DOCS / "dataset"                # originals: images/ + labels/
CROP_DIR = DOCS / "cropped_one"           # cropped_            (train only)
ZOOM_DIR = DOCS / "zoom_one"              # zoom_in_            (train only)
BG_DIR = DOCS / "background_one"          # bg_ / bgcrop_       (train only, empty labels)
FT_DIR = DOCS / "split_train_noaug"       # created here (your folders never change)
PROJECT = DOCS / "training_runs"
SELECT_DIR = DOCS / "selected_weights"    # the winning weights are copied here
RUN_NAME = "finetune_spark_seg_m_noaug"

NAMES = ["head", "tail"]  # class order (0, 1, ...)
VAL_FRAC = 0.20           # last 20 % of each video -> val  (80 / 20 split)
IMGSZ = 1280
EPOCHS = 40
PATIENCE = 10
LR0 = 0.0002              # was 0.001 - too high for an already trained model
SAVE_PERIOD = 5
BATCH = 3
WORKERS = min(8, os.cpu_count() or 4)
AROUND = 2                # loss-match: the epoch + AROUND before + AROUND after
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
# ------------------------------------------------------------------------------------

SKIP_PREFIX = ("cropped_", "zoom_in_", "bg_", "bgcrop_")
VIDEO_RE = re.compile(r"(\d{8}_\d{6}_\d{3})")                     # 20261027_145107_441
# extra image -> its source frame (to keep val frames out of train)
SOURCE_RES = [re.compile(r"^cropped_(.+)$"),
              re.compile(r"^zoom_in_(.+?)(?:_obj\d+)?$"),
              re.compile(r"^bgcrop_(.+)_\d+(?:_dup\d+)?$"),
              re.compile(r"^bg_(.+)$")]


def images_in(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT) if folder.exists() else []


def group_key(stem):
    s = stem.split("_frame")[0].rstrip("_")
    m = VIDEO_RE.search(s)
    if m:
        return m.group(1)
    return re.sub(r"_(video|depth)$", "", s) or stem


def frame_no(stem):
    m = re.search(r"_frame(\d+)", stem) or re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else 0


def source_stem(stem):
    for r in SOURCE_RES:
        m = r.match(stem)
        if m:
            return m.group(1)
    return None


def max_class_id(*label_dirs):
    m = -1
    for t in (t for d in label_dirs if d.exists() for t in d.glob("*.txt")):
        for line in t.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                m = max(m, int(float(p[0])))
    return m


def label_for(img):
    """images/<name>.jpg -> labels/<name>.txt in the same source folder."""
    return img.parent.parent / "labels" / (img.stem + ".txt")


def copy_pair(img, img_dir, lbl_dir):
    dst = img_dir / img.name
    if dst.exists():
        raise SystemExit(f"two images with the same name: {dst.name} - rename one")
    shutil.copy2(img, dst)
    lbl = label_for(img)
    if lbl.exists():
        shutil.copy2(lbl, lbl_dir / (img.stem + ".txt"))
    else:
        (lbl_dir / (img.stem + ".txt")).write_text("")          # no label = background image


def fresh(path):
    if path.exists():
        shutil.rmtree(path)
    (path / "images").mkdir(parents=True)
    (path / "labels").mkdir(parents=True)
    return path / "images", path / "labels"


def write_yaml(path, train_rel, val_rel, names):
    path.write_text(yaml.safe_dump({"path": str(FT_DIR), "train": train_rel, "val": val_rel,
                                    "names": {i: n for i, n in enumerate(names)}}, sort_keys=False))


def split_groups(items_by_group):
    """Last VAL_FRAC of each video (sorted by frame number) -> val, rest -> train."""
    train, val, counts = [], [], {}
    for g, items in sorted(items_by_group.items()):
        items.sort(key=lambda p: (frame_no(p.stem), p.stem))
        n_val = int(round(len(items) * VAL_FRAC)) if len(items) >= 5 else 0
        train += items[:len(items) - n_val]
        val += items[len(items) - n_val:]
        counts[g] = (len(items) - n_val, n_val)
    return train, val, counts


def build_split(originals_only=False):
    for p in (DATASET / "images", DATASET / "labels"):
        if not p.exists():
            raise SystemExit(f"not found: {p}")
    max_cls = max_class_id(DATASET / "labels", CROP_DIR / "labels", ZOOM_DIR / "labels")
    if max_cls + 1 > len(NAMES):
        raise SystemExit(f"labels use class ids up to {max_cls}, but NAMES has {NAMES} - edit NAMES")

    all_ds = [p for p in images_in(DATASET / "images") if "_depth" not in p.stem]
    originals = [p for p in all_ds if not p.stem.startswith(SKIP_PREFIX)]
    ds_extras = [p for p in all_ds if p.stem.startswith(SKIP_PREFIX)]   # extras placed in dataset/

    groups = defaultdict(list)
    for p in originals:
        groups[group_key(p.stem)].append(p)
    train_o, val_o, counts = split_groups(groups)
    val_stems = {p.stem for p in val_o}

    def keep(p):
        s = source_stem(p.stem)
        return s is None or s not in val_stems

    extras = {
        "cropped": [p for p in images_in(CROP_DIR / "images") + ds_extras if p.stem.startswith("cropped_")],
        "zoom_in": [p for p in images_in(ZOOM_DIR / "images") + ds_extras if p.stem.startswith("zoom_in_")],
        "background": [p for p in images_in(BG_DIR / "images") + ds_extras if p.stem.startswith(("bg_", "bgcrop_"))],
    }
    if originals_only:
        extras = {k: [] for k in extras}
    extras_ok = {k: [p for p in v if keep(p)] for k, v in extras.items()}

    assert not [p for p in val_o if p.stem.startswith(SKIP_PREFIX)], "extras ended up in val"

    ti, tl = fresh(FT_DIR / "train")
    for p in train_o + [p for v in extras_ok.values() for p in v]:
        copy_pair(p, ti, tl)
    vi, vl = fresh(FT_DIR / "val")
    for p in val_o:
        copy_pair(p, vi, vl)
    write_yaml(FT_DIR / "data.yaml", "train/images", "val/images", NAMES)

    n_empty_val = sum(1 for p in val_o if not (label_for(p).exists() and label_for(p).read_text().strip()))
    print(f"classes: {NAMES}")
    print(f"{'video':<24}{'train':>7}{'val':>6}")
    for g, (t, v) in counts.items():
        print(f"{g:<24}{t:>7}{v:>6}")
    print(f"originals:   {len(train_o)} train + {len(val_o)} val   [{len(groups)} videos]"
          f"   ({n_empty_val} empty frames in val = false-positive check)")
    for k in extras:
        print(f"{k:<12} {len(extras_ok[k])} -> train   ({len(extras[k]) - len(extras_ok[k])} left out: source in val)")
    n_train = len(train_o) + sum(len(v) for v in extras_ok.values())
    print(f"TOTAL: train {n_train}  |  val {len(val_o)} (originals only) -> {FT_DIR}")
    return FT_DIR / "data.yaml"


# ---------------------------------------------------------------- per-epoch saving
class EpochSaver:
    """After each epoch: keep a copy of the weights (all_epochs/epochN.pt, N = 1-based like
    results.csv) and log train loss vs val loss. At the end: the epoch with the smallest
    gap + AROUND before + AROUND after are copied to loss_match/."""

    def __init__(self):
        self.gaps = {}

    def on_epoch(self, trainer):
        ep = trainer.epoch + 1
        wdir = Path(trainer.wdir)
        last = wdir / "last.pt"
        if not last.exists():
            return
        keep = wdir / "all_epochs"
        keep.mkdir(exist_ok=True)
        shutil.copy2(last, keep / f"epoch{ep}.pt")

        tr = trainer.label_loss_items(trainer.tloss, prefix="train")
        t_loss = sum(float(v) for k, v in tr.items() if k.endswith("_loss"))
        v_loss = sum(float(v) for k, v in (trainer.metrics or {}).items()
                     if k.startswith("val/") and k.endswith("_loss"))
        if v_loss <= 0:
            return
        gap = abs(t_loss - v_loss) / max(t_loss, v_loss)
        self.gaps[ep] = gap
        best_ep = min(self.gaps, key=self.gaps.get)

        log = wdir / "loss_gap.csv"
        new = not log.exists()
        with log.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["epoch", "train_loss", "val_loss", "gap"])
            w.writerow([ep, f"{t_loss:.4f}", f"{v_loss:.4f}", f"{gap:.3f}"])
        print(f"[loss-gap] epoch {ep}: train {t_loss:.4f}  val {v_loss:.4f}  gap {gap:.1%}"
              f"   (smallest so far: epoch {best_ep}, {self.gaps[best_ep]:.1%})")

    def on_end(self, trainer):
        save_loss_match(Path(trainer.wdir))


def read_gaps(wdir):
    log = wdir / "loss_gap.csv"
    if not log.exists():
        return {}
    with log.open() as f:
        return {int(r["epoch"]): float(r["gap"]) for r in csv.DictReader(f)}


def save_loss_match(wdir):
    """Epoch with the smallest train/val gap + AROUND before/after -> loss_match/."""
    gaps = read_gaps(wdir)
    keep = wdir / "all_epochs"
    if not gaps or not keep.exists():
        return []
    e = min(gaps, key=gaps.get)
    d = wdir / "loss_match"
    d.mkdir(exist_ok=True)
    saved = []
    for ep in (e - AROUND, e, e + AROUND):
        src = keep / f"epoch{ep}.pt"
        if src.exists():
            shutil.copy2(src, d / src.name)
            saved.append(ep)
    print(f"[loss-match] smallest gap at epoch {e} ({gaps[e]:.1%}) -> saved epochs {saved} to {d}")
    return saved


NO_AUG = dict(
    # geometry
    degrees=0.0, flipud=0.0, fliplr=0.0, scale=0.0, translate=0.0, shear=0.0, perspective=0.0,
    # colour
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    # multi-image
    mosaic=0.0, mixup=0.0, copy_paste=0.0, close_mosaic=0,
)


def train(data_yaml, run_name):
    import torch
    from ultralytics import YOLO
    torch.backends.cudnn.benchmark = True
    args = dict(
        data=str(data_yaml), project=str(PROJECT), name=run_name, exist_ok=False,
        imgsz=IMGSZ, epochs=EPOCHS, patience=PATIENCE, save_period=SAVE_PERIOD,
        batch=BATCH, workers=WORKERS, cache=False, amp=True, deterministic=False,
        optimizer="AdamW", lr0=LR0, lrf=0.1, cos_lr=True, warmup_epochs=1, weight_decay=0.0005,
        warmup_bias_lr=LR0,                      # default 0.1 is far too high for fine-tuning
        box=7.5, cls=0.5, dfl=1.5, max_det=50,
        overlap_mask=True, mask_ratio=4,
        plots=True,
        **NO_AUG,
    )

    def make_model():
        m = YOLO(str(START))
        if m.task != "segment":
            raise SystemExit(f"{START.name} is a '{m.task}' model, not segmentation")
        saver = EpochSaver()
        m.add_callback("on_fit_epoch_end", saver.on_epoch)
        m.add_callback("on_train_end", saver.on_end)
        return m

    print("[aug] ALL augmentation off: " + ", ".join(f"{k}={v}" for k, v in NO_AUG.items()))
    try:
        # augmentations=[] -> no albumentations at all (otherwise Ultralytics adds its
        # small default set: Blur / MedianBlur / ToGray / CLAHE at p=0.01)
        res = make_model().train(**args, augmentations=[])
    except (TypeError, ValueError, SyntaxError) as e:      # older ultralytics: no 'augmentations'
        print(f"[aug] 'augmentations' not supported ({e}) - Ultralytics may still apply its tiny"
              " default albumentations (p=0.01); uninstall albumentations to remove them too")
        res = make_model().train(**args)
    return Path(res.save_dir) if res is not None else None


# ---------------------------------------------------------------- comparison
def best_map_epoch(run_dir):
    """Epoch with the highest Ultralytics fitness in results.csv (= the best.pt epoch)."""
    rc = run_dir / "results.csv"
    if not rc.exists():
        return None
    best, best_fit = None, -1
    with rc.open() as f:
        for r in csv.DictReader(f):
            r = {k.strip(): v for k, v in r.items()}
            try:
                fit = (0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"])
                       + 0.1 * float(r["metrics/mAP50(M)"]) + 0.9 * float(r["metrics/mAP50-95(M)"]))
            except (KeyError, ValueError):
                continue
            if fit > best_fit:
                best, best_fit = int(float(r["epoch"])), fit
    return best


def compare(run_dir):
    from ultralytics import YOLO
    if run_dir is None or not run_dir.exists():
        raise SystemExit("no training run found")
    wdir = run_dir / "weights"
    keep = wdir / "all_epochs"

    cands = [("start (old model)", START, None)]
    bm = best_map_epoch(run_dir)
    if bm and (keep / f"epoch{bm}.pt").exists():
        cands.append((f"epoch {bm} (best mAP)", keep / f"epoch{bm}.pt", bm))
    for ep in save_loss_match(wdir):
        if ep != bm:
            cands.append((f"epoch {ep} (loss match)", keep / f"epoch{ep}.pt", ep))

    rows = []
    for label, w, ep in cands:
        if not Path(w).exists():
            print(f"[skip] not found: {w}")
            continue
        print(f"\nvalidating {label} ...")
        m = YOLO(str(w)).val(data=str(FT_DIR / "data.yaml"), imgsz=IMGSZ, batch=BATCH,
                             plots=False, verbose=False)
        fit = (0.1 * m.box.map50 + 0.9 * m.box.map) + (0.1 * m.seg.map50 + 0.9 * m.seg.map)
        rows.append((label, Path(w), ep, (m.box.mp, m.box.mr, m.box.map50, m.box.map,
                                         m.seg.map50, m.seg.map, fit)))
    if not rows:
        return

    win = max(rows, key=lambda r: r[3][6])
    print("\n" + "=" * 104)
    print(f"{'weights':<30}{'box P':>7}{'box R':>7}{'box50':>8}{'box50-95':>10}"
          f"{'mask50':>8}{'mask50-95':>11}{'fitness':>9}")
    print("-" * 104)
    for label, _, _, (P, R, b50, b, m50, mm, fit) in rows:
        star = "  <- best" if label == win[0] else ""
        print(f"{label:<30}{P:>7.3f}{R:>7.3f}{b50:>8.3f}{b:>10.3f}{m50:>8.3f}{mm:>11.3f}{fit:>9.3f}{star}")
    print("=" * 104)

    SELECT_DIR.mkdir(parents=True, exist_ok=True)
    label, src, ep, _ = win
    name = f"selected_{run_dir.name}_epoch{ep}.pt" if ep else f"selected_start_{START.stem}.pt"
    shutil.copy2(src, SELECT_DIR / name)
    with (SELECT_DIR / f"compare_{run_dir.name}.csv").open("w") as f:
        f.write("weights,path,box_P,box_R,box_mAP50,box_mAP50-95,mask_mAP50,mask_mAP50-95,fitness,selected\n")
        for lab, w, _, v in rows:
            f.write(f"{lab},{w}," + ",".join(f"{x:.4f}" for x in v) + f",{int(lab == label)}\n")
    if ep is None:
        print("\nThe OLD model is still the best -> keep using it (fine-tune did not help).")
    print(f"WINNER: {label}\n  copied to -> {SELECT_DIR / name}\n  use in your config:  model_path: {name}")


def latest_run():
    runs = sorted((p for p in PROJECT.glob(f"{RUN_NAME}*") if (p / "weights").exists()),
                  key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-only", action="store_true", help="only build split_train/")
    ap.add_argument("--compare-only", action="store_true", help="only the comparison for the latest run")
    ap.add_argument("--originals-only", action="store_true",
                    help="train on original frames only (no cropped_/zoom_in_/background images)")
    a = ap.parse_args()
    if a.compare_only:
        compare(latest_run())
    else:
        data = build_split(a.originals_only)
        if not a.split_only:
            compare(train(data, RUN_NAME + ("_originals" if a.originals_only else "")))