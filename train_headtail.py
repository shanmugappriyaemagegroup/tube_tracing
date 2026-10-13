"""
train_headtail.py - train a YOLO-seg model for tube head / tail on grayscale video frames.

Dataset: dataset_headtail (built by build_head_tail_dataset.py), classes 0 head, 1 tail,
all images 3-channel grayscale.

  python train_headtail.py
  python train_headtail.py --imgsz 960 --batch 4          # if GPU memory is short
  python train_headtail.py --model yolo11m-seg.pt         # bigger model, more accurate, slower

Augmentation choices for this case
  - Grayscale data  -> no hue/saturation jitter (hsv_h = hsv_s = 0), only brightness (hsv_v),
                       so lighting changes between the phone, the bench and the machine camera
                       are covered.
  - Top-down / angled views of loose tubes -> rotation, both flips and a little perspective
                       are safe: a head is a head whichever way it points.
  - Small objects   -> mild scale (0.4) so ends are not shrunk to nothing, mask_ratio 2 for
                       finer masks, mosaic on but switched off for the last 15 epochs so the
                       model finishes on real, uncut frames.
  - Video frames    -> if `pip install albumentations` is done, Ultralytics adds light blur,
                       median blur and CLAHE automatically, which mimics motion blur and
                       compression in video.
  - False positives -> copy_paste and mixup kept low; they paste objects into odd places and
                       can teach the model to fire on background.
"""
import argparse

from ultralytics import YOLO

DATA = "C:/Users/EmageVision/Documents/head_training_dataset/dataset_headtail/data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--model", default="yolo11s-seg.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--name", default="headtail_seg")
    ap.add_argument("--device", default="0", help='"0" for first GPU, "cpu" for CPU')
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    model = YOLO(a.model)
    results = model.train(
        # ---- run ----
        data=a.data,
        imgsz=a.imgsz,
        epochs=a.epochs,
        batch=a.batch,
        device=a.device,
        workers=a.workers,
        project="runs/headtail",
        name=a.name,
        exist_ok=False,
        seed=0,
        deterministic=False,
        cache="disk",             # faster epochs after the first one
        plots=True,

        # ---- optimiser / schedule ----
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        weight_decay=0.0005,
        patience=40,              # stop early if val mAP stops improving
        close_mosaic=15,          # last 15 epochs on real frames only

        # ---- segmentation ----
        overlap_mask=True,
        mask_ratio=2,             # finer masks for small tube ends (default 4)

        # ---- colour / light (grayscale data) ----
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,                # brightness changes: phone vs bench vs machine camera

        # ---- geometry ----
        degrees=20.0,             # tubes lie at any angle
        translate=0.1,
        scale=0.4,                # keep small ends from shrinking too much
        shear=2.0,
        perspective=0.0005,       # angled bench shots
        fliplr=0.5,
        flipud=0.5,               # top-down views, orientation does not matter

        # ---- mixing ----
        mosaic=1.0,
        mixup=0.05,
        copy_paste=0.1,
        erasing=0.2,              # simulates fingers / tape partly hiding an end
    )

    # final check on the val set with the best weights
    # best = YOLO(f"runs/headtail/{a.name}/weights/best.pt")
    best = YOLO(f"{model.trainer.save_dir}/weights/best.pt")
    print("run folder:", model.trainer.save_dir)
    m = best.val(data=a.data, imgsz=a.imgsz, batch=a.batch, device=a.device)
    print(f"\nbox  mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}")
    print(f"mask mAP50 {m.seg.map50:.3f}  mAP50-95 {m.seg.map:.3f}")


if __name__ == "__main__":       # required on Windows for the dataloader workers
    main()