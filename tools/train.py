#!/usr/bin/env python3
"""
train.py  —  YOLOv8s Classification Training Script
=====================================================
Trains a YOLOv8s-cls model on the manually collected warped-frame dataset.

Dataset structure expected under ./dataset/:
  safe/           ← warped frames with object safely centred
  warning/        ← warped frames with object near the boundary
  danger_left/    ← warped frames with object on the left edge
  danger_right/   ← warped frames with object on the right edge
  empty/          ← warped frames with no object present

Output:
  runs/classify/sleeping_monitor/   ← best weights, metrics, confusion matrix
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR   = BASE_DIR / "dataset_split" / "train"
VAL_DIR     = BASE_DIR / "dataset_split" / "val"

MODEL       = "yolov8s-cls.pt"   # YOLOv8 Small — best for RTX 2050
EPOCHS      = 50
IMG_SIZE    = 224                 # standard classification input size
BATCH       = 32                  # fits comfortably in RTX 2050 VRAM
WORKERS     = 4
VAL_SPLIT   = 0.15               # 15 % of each class held out for validation
PROJECT     = "runs/classify"
NAME        = "sleeping_monitor"
DEVICE      = 0                  # GPU 0 (RTX 2050)

CLASSES = ["safe", "warning", "danger_left", "danger_right", "empty"]


# ─────────────────────────────────────────────────────────
# STEP 1 — Split dataset into train / val
# ─────────────────────────────────────────────────────────
def prepare_split():
    """
    Splits each class folder into train/val subsets.
    Re-creates the split fresh on every run to avoid stale data.
    """
    print("\n" + "=" * 58)
    print("  STEP 1 — Preparing train / val split")
    print("=" * 58)

    if (BASE_DIR / "dataset_split").exists():
        shutil.rmtree(BASE_DIR / "dataset_split")

    random.seed(42)
    class_counts = {}

    for cls in CLASSES:
        cls_dir = DATASET_DIR / cls
        if not cls_dir.is_dir():
            print(f"  [WARN] Missing class folder: {cls_dir}")
            continue

        images = sorted(cls_dir.glob("*.jpg"))
        if not images:
            print(f"  [WARN] No images found for class: {cls}")
            continue

        random.shuffle(images)
        n_val   = max(1, int(len(images) * VAL_SPLIT))
        val_imgs = images[:n_val]
        trn_imgs = images[n_val:]

        (TRAIN_DIR / cls).mkdir(parents=True, exist_ok=True)
        (VAL_DIR   / cls).mkdir(parents=True, exist_ok=True)

        for img in trn_imgs:
            shutil.copy(img, TRAIN_DIR / cls / img.name)
        for img in val_imgs:
            shutil.copy(img, VAL_DIR   / cls / img.name)

        class_counts[cls] = (len(trn_imgs), len(val_imgs))
        print(f"  {cls:<16} train={len(trn_imgs):>4}  val={len(val_imgs):>3}")

    print()
    return class_counts


# ─────────────────────────────────────────────────────────
# STEP 2 — Train
# ─────────────────────────────────────────────────────────
def train():
    print("=" * 58)
    print("  STEP 2 — Training YOLOv8s-cls")
    print(f"  Model   : {MODEL}")
    print(f"  Epochs  : {EPOCHS}")
    print(f"  Batch   : {BATCH}")
    print(f"  ImgSize : {IMG_SIZE}")
    print(f"  Device  : GPU {DEVICE} (RTX 2050)")
    print("=" * 58 + "\n")

    model = YOLO(MODEL)

    results = model.train(
        data   = str(BASE_DIR / "dataset_split"),
        epochs = EPOCHS,
        imgsz  = IMG_SIZE,
        batch  = BATCH,
        device = DEVICE,
        workers= WORKERS,
        project= PROJECT,
        name   = NAME,
        # ── Augmentation (helps generalize with ~800 imgs/class) ──
        augment    = True,
        flipud     = 0.0,   # don't flip vertically (bed has fixed orientation)
        fliplr     = 0.5,   # horizontal flip is OK (left/right mirroring)
        degrees    = 5.0,   # slight rotation to handle camera angle variation
        translate  = 0.05,  # slight translation
        scale      = 0.1,   # slight zoom
        hsv_h      = 0.015, # hue jitter
        hsv_s      = 0.4,   # saturation jitter (handles lighting changes)
        hsv_v      = 0.3,   # brightness jitter
        # ── Training parameters ──
        lr0        = 0.01,
        lrf        = 0.01,
        momentum   = 0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        patience   = 15,    # early stopping if no improvement for 15 epochs
        save_period= 10,    # save checkpoint every 10 epochs
        verbose    = True,
    )

    return results


# ─────────────────────────────────────────────────────────
# STEP 3 — Report
# ─────────────────────────────────────────────────────────
def report(results, class_counts):
    print("\n" + "=" * 58)
    print("  TRAINING COMPLETE")
    print("=" * 58)

    best_pt = Path(PROJECT) / NAME / "weights" / "best.pt"
    last_pt = Path(PROJECT) / NAME / "weights" / "last.pt"

    print(f"\n  Best weights : {best_pt}")
    print(f"  Last weights : {last_pt}")

    print("\n  Dataset summary:")
    total_train = total_val = 0
    for cls, (tr, va) in class_counts.items():
        print(f"    {cls:<16} train={tr:>4}  val={va:>3}")
        total_train += tr
        total_val   += va
    print(f"    {'TOTAL':<16} train={total_train:>4}  val={total_val:>3}")

    print("\n  Next step:")
    print(f"    Copy  {best_pt}  to your project root")
    print("    Update img_process.py: MODEL = 'best.pt'")
    print("=" * 58 + "\n")


# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    class_counts = prepare_split()
    results      = train()
    report(results, class_counts)
