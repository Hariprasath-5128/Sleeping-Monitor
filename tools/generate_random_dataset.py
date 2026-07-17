import os
import glob
import random
import numpy as np
from PIL import Image

from animate_slp import (
    load_grayscale, estimate_bg_stats, build_patch_library, outpaint_frame, 
    build_trajectory, WIDTH_SCALE, MARGIN_SMALL_FRAC, PATCH_SIZE, PATCH_OVERLAP, 
    PERSON_MARGIN_COL_FRAC, BAND_COUNT, LOW_FREQ_SIGMA
)

INPUT_DATASET_ROOT = r"C:\Projects\sleeping-monitor\data"
OUTPUT_DATASET = r"C:\Projects\sleeping-monitor\data\new_dataset_IR"

MOTIONS = {
    "1_safe_to_left": [
        (1, 0.5), (10, 0.5), (35, 0.0), (45, 0.0)
    ],
    "2_safe_to_right": [
        (1, 0.5), (10, 0.5), (35, 1.0), (45, 1.0)
    ],
    "3_left_to_safe": [
        (1, 0.0), (10, 0.0), (35, 0.5), (45, 0.5)
    ],
    "4_right_to_safe": [
        (1, 1.0), (10, 1.0), (35, 0.5), (45, 0.5)
    ],
    "5_left_to_safe_to_right": [
        (1, 0.0), (5, 0.0), (20, 0.5), (25, 0.5), (40, 1.0), (45, 1.0)
    ],
    "6_right_to_safe_to_left": [
        (1, 1.0), (5, 1.0), (20, 0.5), (25, 0.5), (40, 0.0), (45, 0.0)
    ],
    "7_safe_to_left_to_safe": [
        (1, 0.5), (15, 0.0), (25, 0.0), (40, 0.5), (45, 0.5)
    ],
    "8_safe_to_right_to_safe": [
        (1, 0.5), (15, 1.0), (25, 1.0), (40, 0.5), (45, 0.5)
    ],
    "9_restless": [
        (1, 0.5), (7, 0.5), (15, 0.0), (21, 0.0), (28, 0.5), (31, 0.5), (39, 1.0), (45, 1.0)
    ],
    "10_always_safe": [
        (1, 0.5), (45, 0.5)
    ]
}

def process_sequence(input_dir, out_seq_dir, motion_name, keyframes):
    paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    if not paths:
        return
        
    first_gray, (orig_w, orig_h) = load_grayscale(paths[0])
    target_w = int(orig_w * WIDTH_SCALE)
    extra = target_w - orig_w
    margin_small = 0 # max(4, int(orig_w * MARGIN_SMALL_FRAC))
    person_margin_col = max(PATCH_SIZE + PATCH_OVERLAP, int(orig_w * PERSON_MARGIN_COL_FRAC))
    feather_px = max(4, int(PATCH_SIZE * 1.5))
    
    # Precompute backgrounds
    frame_data = []
    for path in paths:
        gray, (w, h) = load_grayscale(path)
        if (w, h) != (orig_w, orig_h):
            continue
        low_freq, bg_level = estimate_bg_stats(gray, LOW_FREQ_SIGMA)
        library = build_patch_library(gray, low_freq, person_margin_col, PATCH_SIZE, PATCH_OVERLAP, BAND_COUNT)
        filename = os.path.basename(path)
        frame_data.append((filename, gray, low_freq, bg_level, library))
        
    trajectory = build_trajectory(len(frame_data), keyframes)
    rng = np.random.default_rng(42)
    
    os.makedirs(out_seq_dir, exist_ok=True)
    
    for i, (filename, gray, low_freq, bg_level, library) in enumerate(frame_data):
        p = float(trajectory[i])
        left_margin = int(round(margin_small + p * (extra - 2 * margin_small)))
        left_margin = max(0, min(extra, left_margin))
        right_margin = extra - left_margin
        
        canvas = outpaint_frame(
            gray, left_margin, right_margin, library, low_freq, bg_level,
            PATCH_SIZE, PATCH_OVERLAP, BAND_COUNT, feather_px, rng
        )
        
        img = Image.fromarray(canvas, mode="L")
        
        # Crop and resize
        danger_width = int(target_w * 4.5 / 27)
        crop_left = max(0, danger_width - 25)
        crop_right = min(target_w, target_w - danger_width + 15)
        cropped_img = img.crop((crop_left, 0, crop_right, orig_h))
        resized_img = cropped_img.resize((target_w, orig_h), Image.Resampling.LANCZOS)
        
        resized_img.save(os.path.join(out_seq_dir, filename))

def main():
    # 1. Gather all base sequences (1 per sequence ID)
    seq_dict = {}
    for root, dirs, files in os.walk(INPUT_DATASET_ROOT):
        parts = os.path.normpath(root).split(os.sep)
        if "IR" in parts:
            if glob.glob(os.path.join(root, "*.png")):
                ir_idx = parts.index("IR")
                seq_num = parts[ir_idx - 1]
                if seq_num not in seq_dict:
                    seq_dict[seq_num] = root
                    
    base_dirs = list(seq_dict.values())
    base_dirs.sort() # Sort for determinism before shuffling
    
    print(f"Found {len(base_dirs)} total unique base sequences across dataset.")
    
    # 2. Build the exact 200 sequence mapping
    random.seed(42)
    shuffled_all = list(base_dirs)
    random.shuffle(shuffled_all)
    
    full_pool = list(shuffled_all)
    while len(full_pool) < 200:
        extras = list(base_dirs)
        random.shuffle(extras)
        needed = 200 - len(full_pool)
        full_pool.extend(extras[:needed])
        
    random.shuffle(full_pool)
    
    # 3. Process
    motion_keys = list(MOTIONS.keys())
    
    count = 0
    for i, motion_name in enumerate(motion_keys):
        # Assign 20 sequences to this motion
        group = full_pool[i*20 : (i+1)*20]
        
        for j, input_dir in enumerate(group):
            # Sequence name usually derived from parent, like "00001"
            # input_dir is like .../test1/00001/IR/uncover
            parts = os.path.normpath(input_dir).split(os.sep)
            seq_num = parts[-3]
            
            # e.g. new_dataset_IR/seq_00001_1_safe_to_left_instance_0/IR/uncover
            # added instance_j so if the same seq_num is used twice for the same motion, it doesn't overwrite
            out_seq_dir = os.path.join(OUTPUT_DATASET, f"{seq_num}_{motion_name}_inst{j}", "IR", "uncover")
            
            print(f"Processing {count+1}/200: Motion '{motion_name}', Base '{seq_num}'")
            process_sequence(input_dir, out_seq_dir, motion_name, MOTIONS[motion_name])
            count += 1
            
    print(f"\nSuccessfully generated {count} full sequences for new dataset!")

if __name__ == "__main__":
    main()
