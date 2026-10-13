"""
fix_names.py - write real class names into a YOLO checkpoint that was trained
without them (names show up as class_0, class_1, ...).

  python fix_names.py --model best_640_2classes.pt --names head tail
  python fix_names.py --model best_640_4classes.pt --names head head tail tubing

Names are given in class-id order (0, 1, 2, ...), the same order as in the
data.yaml / CVAT export used for training. A new file <name>_named.pt is written
next to the original; the original is not touched.

run_video_frames.py needs classes named exactly "head" and "tail" (case does not
matter). Two head types (e.g. blue and black) can both be called "head".
"""
import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--names", nargs="+", required=True, help="names in class-id order")
    ap.add_argument("--out", help="output .pt (default: <model>_named.pt)")
    a = ap.parse_args()

    src = Path(a.model)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    names = {i: n for i, n in enumerate(a.names)}

    changed = False
    for key in ("model", "ema"):
        m = ckpt.get(key)
        if m is None:
            continue
        old = getattr(m, "names", None)
        if old is not None and len(old) != len(names):
            raise SystemExit(f"model has {len(old)} classes {old}, but you gave {len(names)} names")
        m.names = names
        changed = True
    if not changed:
        raise SystemExit("no model found inside this checkpoint")

    out = Path(a.out) if a.out else src.with_name(src.stem + "_named.pt")
    torch.save(ckpt, out)
    print(f"{src.name}: {old} -> {names}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()