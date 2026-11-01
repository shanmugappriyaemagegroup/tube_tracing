r"""
find_redo_frames.py - run the head/tail analyzer (run_video_frames.py + config.yaml)
over a folder of images and collect every image that does NOT show exactly
2 heads and 2 tails, together with its label file and the annotated frame.

    python find_redo_frames.py --config config.yaml --images D:\data\images
    python find_redo_frames.py --config config.yaml --images D:\data\images --labels D:\data\labels
    python find_redo_frames.py --config config.yaml --images frames --sequence   # frames of ONE video, in order

Output (default C:\Users\EmageVision\Documents\redo_frames):
    images\      the original image
    labels\      its label .txt (if one was found)
    annotated\   the image with the analyzer's boxes / cards drawn on it
    results\     the analyzer's JSON result for that image
    report.csv   EVERY processed image: heads/tails found, heads/tails in the label, OK / REDO

Two modes:
  default     each image is analysed on its own (tracking is reset per image and
              min_frames = 1, so a single detection counts straight away)
  --sequence  the images are consecutive frames of one video: tracking runs across
              them like on a video (min_frames from the config applies, so the first
              few frames show 0 confirmed ends and will be listed as REDO)

Label files are found as (first match wins):
  --labels\<name>.txt   ->   <name>.txt next to the image   ->   ...\labels\... instead of ...\images\...
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from head_tail_analyzer.run_video_frames import (
        AnalyzerError, StreamState, analyze_frame, configure_for_model,
        load_model, load_stream_config)
except ImportError:          # script placed next to run_video_frames.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_video_frames import (
        AnalyzerError, StreamState, analyze_frame, configure_for_model,
        load_model, load_stream_config)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_OUT = r"C:\Users\EmageVision\Documents\redo_frames"


# ------------------------------------------------------------------ file helpers
def read_image(path):
    """cv2.imread fails on non-ASCII Windows paths; this does not."""
    data = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def write_image(path, img):
    ext = path.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"cannot encode {path}")
    buf.tofile(str(path))


def list_images(folder, recursive):
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in Path(folder).glob(pattern)
                  if p.is_file() and p.suffix.lower() in IMG_EXT)


def find_label(img, images_dir, labels_dir):
    cands = []
    if labels_dir:
        cands.append(Path(labels_dir) / img.relative_to(images_dir).with_suffix(".txt"))
        cands.append(Path(labels_dir) / (img.stem + ".txt"))
    cands.append(img.with_suffix(".txt"))
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):          # YOLO layout: .../images/... -> .../labels/...
        if parts[i].lower() == "images":
            cands.append(Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt"))
            break
    return next((c for c in cands if c.is_file()), None)


def label_counts(label_path, role_ids):
    """How many HEAD / TAIL objects the label file says the image has."""
    counts = {"HEAD": 0, "TAIL": 0}
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        tok = line.split()
        if not tok:
            continue
        try:
            role = role_ids.get(int(float(tok[0])))
        except ValueError:
            continue
        if role:
            counts[role] += 1
    return counts


def out_name(img, images_dir):
    """Unique flat file name (sub-folders joined with __ when --recursive)."""
    return "__".join(img.relative_to(images_dir).with_suffix("").parts)


# ------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Collect images that do not have the expected heads/tails count.")
    ap.add_argument("--config", help="config.yaml used by run_video_frames / main.py")
    ap.add_argument("--model", help="model .pt (overrides config)")
    ap.add_argument("--images", required=True, help="folder with the images")
    ap.add_argument("--labels", help="folder with the label .txt files (optional, see docstring)")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"output folder (default {DEFAULT_OUT})")
    ap.add_argument("--heads", type=int, default=2, help="expected heads per image (default 2)")
    ap.add_argument("--tails", type=int, default=2, help="expected tails per image (default 2)")
    ap.add_argument("--sequence", action="store_true", help="images are consecutive video frames: track across them")
    ap.add_argument("--recursive", action="store_true", help="also look in sub-folders of --images")
    ap.add_argument("--no-roi", action="store_true", help="ignore the config's ROI, count the whole image")
    ap.add_argument("--device", help="0 = first GPU, cpu = CPU")
    ap.add_argument("--conf", type=float, help="confidence threshold (overrides config)")
    a = ap.parse_args(argv)

    images_dir = Path(a.images).expanduser().resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"images folder not found: {images_dir}")
    images = list_images(images_dir, a.recursive)
    if not images:
        raise SystemExit(f"no images ({', '.join(sorted(IMG_EXT))}) in {images_dir}")

    overrides = dict(model_path=a.model, device=a.device, conf=a.conf)
    if not a.sequence:                       # single images: confirm on the first sighting
        overrides.update(min_frames=1, min_frames_tail=1)
    try:
        cfg, roi = load_stream_config(a.config, use_roi=not a.no_roi, **overrides)
    except AnalyzerError as e:
        raise SystemExit(f"[config] {e}")
    cfg.print_frames = False                 # no per-frame terminal spam
    cfg.latest_json = None                   # this script writes its own outputs
    cfg.history_jsonl = None

    entry = load_model(cfg.model_path, device=cfg.device, imgsz=int(cfg.imgsz or cfg.tile or 640))
    masks_ok = configure_for_model(cfg, entry)

    out = Path(a.out)
    dirs = {k: out / k for k in ("images", "labels", "annotated", "results")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"[run] {len(images)} images from {images_dir}")
    print(f"[run] expected {a.heads} heads + {a.tails} tails, "
          f"mode={'sequence' if a.sequence else 'single images'}, ROI={'on' if roi is not None else 'off'}")
    print(f"[run] redo frames -> {out}\n")

    state = StreamState()
    rows, n_redo = [], 0
    for i, img_path in enumerate(images, 1):
        img = read_image(img_path)
        if img is None:
            print(f"  [{i}/{len(images)}] SKIP (cannot read) {img_path.name}")
            rows.append({"image": str(img_path), "status": "UNREADABLE"})
            continue
        if not a.sequence:
            state.reset()

        result, vis = analyze_frame(entry, img, state, roi, cfg,
                                    frame_index=i, annotate=True, masks=masks_ok)
        heads, tails = result["heads_count"], result["tails_count"]
        redo = heads != a.heads or tails != a.tails

        label = find_label(img_path, images_dir, a.labels)
        gt = label_counts(label, entry.role_ids) if label else None
        reasons = []
        if heads != a.heads:
            reasons.append(f"heads {heads}/{a.heads}")
        if tails != a.tails:
            reasons.append(f"tails {tails}/{a.tails}")

        rows.append({
            "image": str(img_path),
            "status": "REDO" if redo else "OK",
            "heads_found": heads,
            "tails_found": tails,
            "heads_in_label": gt["HEAD"] if gt else "",
            "tails_in_label": gt["TAIL"] if gt else "",
            "label_file": str(label) if label else "NOT FOUND",
            "reason": ", ".join(reasons),
            "bigger_tube": result["bigger_tube"]["status"],
        })

        if redo:
            n_redo += 1
            name = out_name(img_path, images_dir)
            shutil.copy2(img_path, dirs["images"] / (name + img_path.suffix))
            if label:
                shutil.copy2(label, dirs["labels"] / (name + ".txt"))
            write_image(dirs["annotated"] / (name + ".jpg"), vis)
            (dirs["results"] / (name + ".json")).write_text(json.dumps(result, indent=2))

        gt_txt = f"  label: {gt['HEAD']}H {gt['TAIL']}T" if gt else "  label: -"
        print(f"  [{i}/{len(images)}] {'REDO' if redo else 'ok  '}  "
              f"found {heads}H {tails}T{gt_txt}  {img_path.name}")

    report = out / "report.csv"
    fields = ["image", "status", "heads_found", "tails_found", "heads_in_label",
              "tails_in_label", "label_file", "reason", "bigger_tube"]
    with open(report, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)

    print(f"\n[done] {len(images)} images, {n_redo} REDO, {len(images) - n_redo} OK")
    print(f"  redo frames : {out}")
    print(f"  report      : {report}")


if __name__ == "__main__":
    main()