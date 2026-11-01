# train_v6_plus.py - fine-tune the honest tube_v6 model on tube_dataset_v6 PLUS the relabelled tail frames (dataset_plus_os.yaml).
# Keep this file in its own folder (E:\\Sona\\tube_train\\), NOT next to the ultralytics source folder.
# Compare afterwards on val_new.yaml and val_base.yaml (tail MASK AP50) against the starting model.

import argparse
import os
import sys
 
_here = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(_here, "ultralytics")) and not os.path.exists(os.path.join(_here, "ultralytics", "__init__.py")):
    sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _here]
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
 
from ultralytics import YOLO
 
START = r"E:\Sona\training_runs_tube_detection_1280_v6\tube_v6_1280\weights\best.pt"
DATA = r"E:\Sona\tube_dataset_v6\dataset_plus_os.yaml"
PROJECT = r"E:\Sona\training_runs_tube_detection_1280_v6_plus"
 
 
def custom_augmentations():
    try:
        import albumentations as A
    except ImportError:
        print("[aug] albumentations not installed -> built-in augmentations only")
        return None
    new_api = int(A.__version__.split(".")[0]) >= 2
    aug = [A.MotionBlur(blur_limit=(3, 11), p=0.25), A.ToGray(p=0.1)]
    for make in (lambda: A.ImageCompression(quality_range=(60, 95), p=0.2),
                 lambda: (A.GaussNoise(std_range=(0.02, 0.08), p=0.15) if new_api
                          else A.GaussNoise(var_limit=(25.0, 400.0), p=0.15))):
        try:
            aug.append(make())
        except Exception:
            pass
    print(f"[aug] {len(aug)} custom augmentations active (albumentations {A.__version__})")
    return aug
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=3)
    a = ap.parse_args()
 
    args = dict(
        data=DATA, imgsz=1280, device=0,
        epochs=15, patience=6,
        batch=a.batch,
        workers=4,
        cache="disk",
        amp=True, deterministic=False,
        mask_ratio=2,
        optimizer="AdamW", lr0=0.0003, lrf=0.1, cos_lr=True, warmup_epochs=1.0,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        degrees=15, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
        flipud=0.5, fliplr=0.5,
        mosaic=1.0, close_mosaic=6,
        mixup=0.1, copy_paste=0.3, copy_paste_mode="flip",
        project=PROJECT, name="tube_v6_plus", plots=False,
    )
    aug = custom_augmentations()
    if aug:
        try:
            YOLO(START).train(**args, augmentations=aug)
            return
        except (SyntaxError, TypeError, ValueError) as e:
            print(f"[aug] custom augmentations rejected ({e}); training with built-in augmentations only")
    YOLO(START).train(**args)
 
 
if __name__ == "__main__":
    main()