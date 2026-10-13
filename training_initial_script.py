# First i trained the whole your dataset with yolov11m-seg model , then I added more frames and finetuned with this:
# """Fine-tune the 1280 yolo11m-seg tube model on tube_dataset_v4 (imgsz stays 1280), tuned to keep the GPU busy:
# - no slow CPU augmentations (RandomFog ~5 s/image and SunFlare ~250 ms/image starved the GPU)
# - more dataloader workers, so the GPU never waits for data
# - batch sized to fit 11 GB without spilling into shared system memory
# - cudnn autotune + allocator settings for fixed 1280 inputs
# Edit the three paths below for your PC."""
import os
import sys
 
# If this script sits next to an ultralytics *source* folder (e.g. E:/Sona/ultralytics), Python can import that empty
# folder instead of the real package ("cannot import name 'YOLO' from 'ultralytics' (unknown location)").
# Safest fix: keep this script in its own folder. This guard also removes the script folder from the import path.
_here = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(_here, "ultralytics")) and not os.path.exists(os.path.join(_here, "ultralytics", "__init__.py")):
    sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _here]
 
# Must be set before torch is imported (ultralytics imports torch).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256,garbage_collection_threshold:0.8")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
 
import albumentations as A
import torch
from ultralytics import YOLO
 
MODEL = r"E:\Sona\training_runs_tube_detection\tube_v3_y11m_1280-3\weights\best.pt"
DATA = r"E:\Sona\tube_dataset_v4_arranged_final\dataset.yaml"
PROJECT = r"E:\Sona\training_runs_tube_detection_1280"
 
BATCH = 3                                   # 4 needed ~12 GB on an 11 GB card (spills, slow). Raise to 4 only if GPU_mem stays < ~10 GB.
WORKERS = min(8, os.cpu_count() or 4)       # lower to 4 if the data loader runs out of RAM
CACHE = False                               # 'disk' may help if GPU-Util is still < 90% (untested on your ultralytics build); 'ram' crashed there
 
# Pixel-level only (never move/warp pixels), so masks and boxes stay valid. Cheap ones only (ms per 1280x1280 image).
CANDIDATES = [
    lambda: A.RandomBrightnessContrast(brightness_limit=(-0.35, 0.35), contrast_limit=(-0.35, 0.35), p=0.5),  # 2 ms
    lambda: A.RandomGamma(gamma_limit=(55, 160), p=0.4),                                                      # 2 ms
    lambda: A.RandomToneCurve(scale=0.3, p=0.3),                                                              # 2 ms
    lambda: A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.2),                                         # 5 ms
    lambda: A.PlanckianJitter(mode="blackbody", temperature_limit=(3000, 9000), p=0.3),                       # 5 ms
    lambda: A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.25),                         # 5 ms
    lambda: A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2), shadow_dimension=5, shadow_intensity_range=(0.3, 0.7), p=0.3),  # 27 ms
    # GaussNoise changed its argument in albumentations 2.x (std_range); 1.x ignores it silently, so pick by version
    lambda: (A.GaussNoise(std_range=(0.02, 0.08), p=0.3) if int(A.__version__.split(".")[0]) >= 2
             else A.GaussNoise(var_limit=(25.0, 400.0), p=0.3)),                                              # 48 ms
    lambda: A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.15),                               # 86 ms
    lambda: A.MotionBlur(blur_limit=(3, 9), p=0.2),                                                           # 3 ms
    lambda: A.GaussianBlur(blur_limit=(3, 7), p=0.15),                                                        # 3 ms
    lambda: A.Defocus(radius=(2, 5), alias_blur=(0.1, 0.4), p=0.08),                                          # 69 ms
    lambda: A.ImageCompression(quality_range=(35, 95), p=0.3),                                                # 7 ms
    lambda: A.Downscale(scale_range=(0.5, 0.9), p=0.15),                                                      # 2 ms
    lambda: A.Sharpen(alpha=(0.1, 0.4), lightness=(0.8, 1.2), p=0.1),                                         # 4 ms
    lambda: A.ToGray(p=0.03),                                                                                 # 2 ms
]
augs = []
for make in CANDIDATES:
    try:
        augs.append(make())
    except Exception as e:  # different albumentations version
        print("skipping augmentation:", e)
print(f"augmentations active: {len(augs)} of {len(CANDIDATES)} (albumentations {A.__version__})")
 
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True   # fixed 1280 input -> pick the fastest conv kernels once
    model = YOLO(MODEL)
    model.train(
        data=DATA, project=PROJECT, name="tube_v4_ft_1280_fast",
        imgsz=1280, epochs=80, patience=25, batch=BATCH, workers=WORKERS, cache=CACHE, amp=True, deterministic=False,
        optimizer="AdamW", lr0=0.0005, lrf=0.1, warmup_epochs=2, weight_decay=0.0005,
        mask_ratio=2, overlap_mask=True, max_det=50,
        # geometry: tube can lie at any angle, any side up, any distance
        degrees=180, flipud=0.5, fliplr=0.5, scale=0.5, translate=0.1, shear=2.0, perspective=0.0005,
        # colour / lighting handled by ultralytics (strong brightness range) + the list above
        hsv_h=0.02, hsv_s=0.8, hsv_v=0.6,
        mosaic=1.0, copy_paste=0.3, mixup=0.0, close_mosaic=15, bgr=0.0,
        augmentations=augs, plots=True,
    )