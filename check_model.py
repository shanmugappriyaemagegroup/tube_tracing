"""
check_model.py - quick sanity check that a trained YOLO model detects heads/tails.

Runs the model on images (a file, a folder, or frames grabbed from a video) with
NO tracking or merging, and reports per-class counts and confidences.
Two modes are compared for each image:
    tiled   - 640 px tiles stitched back (what run_video_frames.py does)
    whole   - the full frame in one pass at --imgsz
--scale shrinks the frame first (use ~0.33 when the ends look huge).
--roi roi.json counts only detections whose centre is inside the polygon;
--draw-roi lets you click that polygon first (on the first frame tested).

    python check_model.py --model best.pt --source video.mp4 --frames 60 300 --scale 0.33 --draw-roi
    python check_model.py --model best.pt --source video.mp4 --frames 60 300 --scale 0.33 --roi roi.json
    python check_model.py --model best.pt --source frame.jpg
    python check_model.py --model best.pt --source C:/data/images/val
    python check_model.py --model best.pt --source part_4.avi --frames 0 200 400 800
    python check_model.py --model best.pt --source part_4.avi --frames 200 --conf 0.1 --color

Annotated images are written to --out (default: check_model_out).
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    from .run_video_frames import draw_roi_interactive, load_roi    # installed package
except ImportError:
    from run_video_frames import draw_roi_interactive, load_roi     # run as a script

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def load_inputs(source, frames):
    p = Path(source)
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                yield f.name, cv2.imread(str(f))
    elif p.suffix.lower() in IMG_EXT:
        yield p.name, cv2.imread(str(p))
    else:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise SystemExit(f"cannot open {p}")
        for n in frames or [0]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ok, img = cap.read()
            if ok:
                yield f"{p.stem}_frame{n}.jpg", img
            else:
                print(f"[skip] frame {n} not readable")
        cap.release()


def tile_origins(size, tile, overlap):
    last = max(size - tile, 0)
    step = max(int(tile * (1 - overlap)), 1)
    xs = list(range(0, last + 1, step))
    if xs[-1] != last:
        xs.append(last)
    return xs


def predict_tiled(model, img, a):
    h, w = img.shape[:2]
    crops, offs = [], []
    for y in tile_origins(h, a.tile, a.overlap):
        for x in tile_origins(w, a.tile, a.overlap):
            crops.append(img[y:y + a.tile, x:x + a.tile])
            offs.append((x, y))
    res = model.predict(crops, conf=a.conf, imgsz=a.tile, device=a.device, verbose=False)
    out = []
    for r, (ox, oy) in zip(res, offs):
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
                           r.boxes.conf.cpu().numpy()):
            out.append((int(c), float(s), [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]))
    return out


def predict_whole(model, img, a):
    r = model.predict(img, conf=a.conf, imgsz=a.imgsz, device=a.device, verbose=False)[0]
    return [(int(c), float(s), list(b)) for b, c, s in
            zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]


def summarize(label, dets, names):
    parts = []
    for cid, name in names.items():
        confs = [s for c, s, _ in dets if c == cid]
        if confs:
            parts.append(f"{name}={len(confs)} (conf {min(confs):.2f}-{max(confs):.2f})")
        else:
            parts.append(f"{name}=0")
    print(f"  {label:6s} " + "  ".join(parts))


def inside(det, roi):
    if roi is None:
        return True
    x1, y1, x2, y2 = det[2]
    return cv2.pointPolygonTest(roi, ((x1 + x2) / 2, (y1 + y2) / 2), False) >= 0


def draw(img, dets, names, roi=None):
    vis = img.copy()
    if roi is not None:
        cv2.polylines(vis, [roi.reshape(-1, 1, 2)], True, (0, 220, 0), 4, cv2.LINE_AA)
    for cid, s, (x1, y1, x2, y2) in dets:
        color = ((255, 40, 20), (230, 220, 0), (0, 200, 0))[cid % 3]
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(vis, f"{names[cid]} {s:.2f}", (int(x1), max(int(y1) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True, help="trained weights (.pt)")
    ap.add_argument("--source", required=True, help="image, folder of images, or video")
    ap.add_argument("--frames", type=int, nargs="*", help="frame numbers to test (video only)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=1280, help="size for the whole-frame pass")
    ap.add_argument("--device", default="0", help="0 = GPU, cpu = CPU")
    ap.add_argument("--color", action="store_true", help="feed colour (default: grayscale, like run_video_frames)")
    ap.add_argument("--scale", type=float, default=1.0, help="shrink the frame before inference, e.g. 0.33")
    ap.add_argument("--roi", help="roi.json polygon: only detections inside it are counted")
    ap.add_argument("--draw-roi", action="store_true", help="click an ROI polygon first (saved to --roi or roi.json)")
    ap.add_argument("--out", default="check_model_out")
    a = ap.parse_args()

    model = YOLO(a.model)
    print(f"[model] {a.model}\n[classes] {model.names}\n[conf] {a.conf}  "
          f"[input] {'colour' if a.color else 'grayscale'}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    roi_path = a.roi or ("roi.json" if a.draw_roi else None)
    if a.draw_roi:
        first = a.frames[0] if a.frames else 0
        if draw_roi_interactive(a.source, roi_path, first) is None:
            raise SystemExit("no ROI drawn - stopped")
    roi = load_roi(roi_path) if roi_path else None
    if roi is not None:
        print(f"[roi] {len(roi)}-point polygon from {roi_path}: counting only inside it")

    k = a.scale
    n = 0
    for name, img in load_inputs(a.source, a.frames):
        if img is None:
            continue
        n += 1
        if not a.color:
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k != 1 else img
        print(f"\n{name}  ({img.shape[1]}x{img.shape[0]}, scale {k})")
        for label, fn in (("tiled", predict_tiled), ("whole", predict_whole)):
            dets = [(c, s, [v / k for v in b]) for c, s, b in fn(model, small, a)]   # original pixels
            if roi is not None:
                summarize(label + " all", dets, model.names)
                dets = [d for d in dets if inside(d, roi)]
            summarize(label + (" ROI" if roi is not None else ""), dets, model.names)
            cv2.imwrite(str(out / f"{Path(name).stem}_{label}.jpg"), draw(img, dets, model.names, roi))
    print(f"\n[done] {n} image(s) -> {out.resolve()}")
    if n == 0:
        print("nothing was read - check --source")


if __name__ == "__main__":
    main()
