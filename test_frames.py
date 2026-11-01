"""
test_frames.py - run the head/tail analyzer on a FOLDER OF IMAGES with any .pt
and save an annotated copy of every image + one JSON line per image.

    python test_frames.py --config config.yaml ^
        --model "C:\\Users\\EmageVision\\Documents\\head_training_dataset\\priya_weights\\tube_pt\\epoch25_finetune_spark.pt" ^
        --frames "C:\\Users\\EmageVision\\Documents\\frames_test"

Output (default: <frames folder>\\results_<model name>):
    <image name>.jpg      annotated image (boxes, IDs, width, Bigger/Smaller)
    results.jsonl         one line per image: file, heads_count, tails_count, detections, ...
    summary printed at the end

--independent  (default) every image is analysed on its own: tracks are reset
               and an end counts on the first sighting (min_frames = 1).
               Use for random frames / photos.
--sequence     the images are consecutive frames of one video (sorted by name):
               tracking + min_frames from config.yaml apply, like a video.
"""
import argparse
import json
import time
from pathlib import Path

import cv2

try:
    from head_tail_analyzer.main import HeadTailAnalyzer, AnalyzerError   # installed wheel
except ImportError:
    from main import HeadTailAnalyzer, AnalyzerError                      # next to main.py

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Test a model on a folder of images.")
    ap.add_argument("--config", required=True, help="config.yaml (thresholds, tile, gray, ...)")
    ap.add_argument("--model", help=".pt to test (overrides model_path in the config)")
    ap.add_argument("--frames", required=True, help="folder with the images")
    ap.add_argument("--out", help="output folder (default: <frames>/results_<model name>)")
    ap.add_argument("--device", help="0 = first GPU, cpu = CPU")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--independent", action="store_true", help="each image on its own (default)")
    mode.add_argument("--sequence", action="store_true", help="images are consecutive video frames")
    a = ap.parse_args(argv)

    frames_dir = Path(a.frames).expanduser()
    if not frames_dir.is_dir():
        raise SystemExit(f"frames folder not found: {frames_dir}")
    images = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not images:
        raise SystemExit(f"no images ({', '.join(sorted(IMG_EXT))}) in {frames_dir}")

    sequence = a.sequence
    overrides = dict(model_path=a.model, device=a.device, print_frames=False)
    if not sequence:
        overrides["min_frames"] = 1          # one image = one sighting -> count it right away
    try:
        analyzer = HeadTailAnalyzer(a.config, preload=True, **overrides)   # None values are ignored
    except AnalyzerError as e:
        raise SystemExit(f"error: {e}")
    # this script writes its own outputs; don't overwrite the video run's latest.json / history.jsonl
    analyzer.cfg.latest_json = analyzer.cfg.history_jsonl = analyzer.cfg.output_video = None

    model_name = Path(analyzer.cfg.model_path).stem
    out_dir = Path(a.out).expanduser() if a.out else frames_dir / f"results_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    c = analyzer.cfg
    print(f"[test] model   : {c.model_path}")
    print(f"[test] images  : {len(images)} in {frames_dir}")
    print(f"[test] mode    : {'sequence (tracking on)' if sequence else 'independent (each image alone)'}")
    print(f"[test] settings: tile={c.tile} imgsz={c.imgsz or c.tile} gray={c.gray} "
          f"conf_head={c.conf_head if c.conf_head is not None else c.conf} "
          f"conf_tail={c.conf_tail if c.conf_tail is not None else c.conf} "
          f"min_frames={c.min_frames} device={c.device}")
    print(f"[test] output  : {out_dir}\n")

    totals = {"images": 0, "with_head": 0, "with_tail": 0, "empty": 0, "heads": 0, "tails": 0}
    t0 = time.time()
    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as jf:
        for i, img_path in enumerate(images, 1):
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"  [skip] cannot read {img_path.name}")
                continue
            if not sequence:
                analyzer.reset()             # forget the previous image
            result, vis = analyzer.process_frame_annotated(frame, frame_index=i)
            cv2.imwrite(str(out_dir / f"{img_path.stem}.jpg"), vis)

            h, t = result["heads_count"], result["tails_count"]
            totals["images"] += 1
            totals["heads"] += h
            totals["tails"] += t
            totals["with_head"] += h > 0
            totals["with_tail"] += t > 0
            totals["empty"] += (h == 0 and t == 0)
            jf.write(json.dumps({"file": img_path.name, **result}) + "\n")

            confs = " ".join(f"{d['role'][0]}{d['conf']:.2f}" for d in result["detections"])
            print(f"  {i:>4}/{len(images)}  {img_path.name:<40} HEADS={h} TAILS={t}  {confs}")

    n = max(totals["images"], 1)
    print(f"\n[done] {totals['images']} images in {time.time() - t0:.1f}s   model: {model_name}")
    print(f"  images with a head : {totals['with_head']} ({100 * totals['with_head'] / n:.0f}%)")
    print(f"  images with a tail : {totals['with_tail']} ({100 * totals['with_tail'] / n:.0f}%)")
    print(f"  images with nothing: {totals['empty']} ({100 * totals['empty'] / n:.0f}%)")
    print(f"  heads found: {totals['heads']}   tails found: {totals['tails']}")
    print(f"  annotated images + results.jsonl: {out_dir}")


if __name__ == "__main__":
    main()