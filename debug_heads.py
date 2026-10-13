"""
debug_heads.py - find WHERE heads disappear between the model and the output.

Run from the same folder as run_video_frames.py:
    python debug_heads.py

It reuses the CONFIG from run_video_frames.py and counts HEAD / TAIL at each stage:
  1. raw     - plain YOLO on 640 tiles (independent of your pipeline code)
  2. analyse - after measure_from_masks.analyse()
  3. tip     - after the `"tip" in e` filter
  4. cluster - after keep_largest_clusters()
The stage where heads drop to ~0 is the culprit.
"""
import collections

import cv2
from ultralytics import YOLO

from measure_from_masks import analyse, build_class_map, masks_from_model
from run_video_frames import (MODEL_PATH, VIDEO_PATH, DEVICE, CONF, TILE, TILE_OVERLAP,
                              CLUSTER_DIST, EXPECTED_TUBE_COUNT, keep_largest_clusters)

N_FRAMES = 60          # how many frames to inspect
PRINT_EVERY = 10       # per-frame detail every N frames


def tile_origins(size, tile, overlap):
    last = max(size - tile, 0)
    step = max(int(tile * (1 - overlap)), 1)
    xs = list(range(0, last + 1, step))
    if xs[-1] != last:
        xs.append(last)
    return xs


def raw_counts(model, frame):
    h, w = frame.shape[:2]
    c = collections.Counter()
    for y in tile_origins(h, TILE, TILE_OVERLAP):
        for x in tile_origins(w, TILE, TILE_OVERLAP):
            r = model.predict(frame[y:y + TILE, x:x + TILE], conf=CONF,
                              device=DEVICE, verbose=False)[0]
            for k in r.boxes.cls.tolist():
                c[model.names[int(k)].lower()] += 1
    return c


def role_of(e):
    label = str(e.get("role") or e.get("class") or "").lower()
    if "head" in label:
        return "head"
    if "tail" in label:
        return "tail"
    return label or "?"


def count(ends):
    return collections.Counter(role_of(e) for e in ends)


def main():
    model = YOLO(MODEL_PATH)
    class_map = build_class_map(model.names)
    print(f"[classes] model.names = {model.names}")
    print(f"[classes] class_map   = {class_map}\n")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {VIDEO_PATH}")

    totals = {s: collections.Counter() for s in ("raw", "analyse", "tip", "cluster")}
    headless_status = collections.Counter()   # why heads had no tip
    head_keys = set()

    for idx in range(1, N_FRAMES + 1):
        ok, frame = cap.read()
        if not ok:
            break

        raw = raw_counts(model, frame)
        inst = masks_from_model(model, frame, CONF, True, TILE, TILE_OVERLAP)
        ends = analyse(frame, inst, None, gap=3, class_map=class_map)
        with_tip = [e for e in ends if "tip" in e]
        clustered = keep_largest_clusters(with_tip, CLUSTER_DIST, EXPECTED_TUBE_COUNT)

        for e in ends:
            if role_of(e) == "head":
                head_keys.update(e.keys())
                if "tip" not in e:
                    headless_status[str(e.get("status"))] += 1

        stages = {"raw": raw, "analyse": count(ends), "tip": count(with_tip),
                  "cluster": count(clustered)}
        for s, c in stages.items():
            totals[s].update(c)

        if idx % PRINT_EVERY == 0 or idx == 1:
            line = "  ".join(f"{s}: H={c.get('head', 0)} T={c.get('tail', 0)}"
                             for s, c in stages.items())
            print(f"frame {idx:4d}  {line}")

    cap.release()

    print("\n=== TOTALS over", idx, "frames ===")
    for s, c in totals.items():
        print(f"  {s:8s} heads={c.get('head', 0):5d}  tails={c.get('tail', 0):5d}  other={dict((k, v) for k, v in c.items() if k not in ('head', 'tail'))}")
    print(f"\n  keys present on head entries from analyse(): {sorted(head_keys)}")
    print(f"  heads dropped for missing 'tip', by status: {dict(headless_status)}")

    r, a, t, c = (totals[s].get("head", 0) for s in ("raw", "analyse", "tip", "cluster"))
    print("\n=== VERDICT ===")
    if r == 0:
        print("  Model finds no heads even on raw tiles -> wrong weights file, or the video "
              "differs from training (colour vs grey, resolution). Check MODEL_PATH.")
    elif a == 0:
        print("  Model finds heads, analyse() drops them -> check build_class_map / role "
              "mapping (see [classes] lines above).")
    elif t < 0.2 * a:
        print("  analyse() returns heads but WITHOUT a 'tip' -> the tip/width logic fails on "
              "heads (likely needs a tubing mask, and 'tubing' has only 2 training labels). "
              "See the status breakdown above.")
    elif c < 0.2 * t:
        print("  keep_largest_clusters() is removing the heads -> set "
              "FILTER_ISOLATED_DETECTIONS = False (heads and tails form separate clusters "
              "in this setup).")
    else:
        print("  Heads survive every per-frame stage -> the loss is in tracking/confirmation. "
              "Check the HEAD section of print_diagnosis() in run_video_frames.py.")


if __name__ == "__main__":
    main()