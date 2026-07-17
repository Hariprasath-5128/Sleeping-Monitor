import os
import glob
import numpy as np
from PIL import Image

# Import the core logic from animate_slp (the user's patch quilting outpainter)
from animate_slp import (
    load_grayscale, estimate_bg_stats, build_patch_library, outpaint_frame, 
    build_trajectory, WIDTH_SCALE, MARGIN_SMALL_FRAC, PATCH_SIZE, PATCH_OVERLAP, 
    PERSON_MARGIN_COL_FRAC, BAND_COUNT, LOW_FREQ_SIGMA
)

def main():
    input_dir = r"C:\Projects\sleeping-monitor\dataset_IR\train\train\00001\IR\uncover"
    output_dir = r"C:\Projects\sleeping-monitor\batch_animations"
    os.makedirs(output_dir, exist_ok=True)
    
    paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No PNG frames found in {input_dir}")
    n_frames = len(paths)
    
    # Use the first frame to size the canvas / derive scaled parameters
    first_gray, (orig_w, orig_h) = load_grayscale(paths[0])
    target_w = int(orig_w * WIDTH_SCALE)
    extra = target_w - orig_w
    margin_small = max(4, int(orig_w * MARGIN_SMALL_FRAC))
    person_margin_col = max(PATCH_SIZE + PATCH_OVERLAP, int(orig_w * PERSON_MARGIN_COL_FRAC))
    feather_px = max(4, int(PATCH_SIZE * 1.5))
    
    # Calculate the crop bounds (removing danger zones)
    danger_width = int(target_w * 4.5 / 27)
    crop_left = danger_width
    
    # Extend the right side cropping line slightly outwards by 10 pixels
    crop_right = target_w - danger_width + 10

    # The 10 motion trajectories requested
    motions = {
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
    
    # Precompute statistics and background libraries for all frames to save massive amounts of time
    print(f"Precomputing background features for {n_frames} frames...")
    frame_data = []
    for path in paths:
        gray, (w, h) = load_grayscale(path)
        if (w, h) != (orig_w, orig_h):
            continue
        low_freq, bg_level = estimate_bg_stats(gray, LOW_FREQ_SIGMA)
        library = build_patch_library(gray, low_freq, person_margin_col, PATCH_SIZE, PATCH_OVERLAP, BAND_COUNT)
        frame_data.append((gray, low_freq, bg_level, library))

    for motion_name, keyframes in motions.items():
        print(f"\nGenerating motion: {motion_name}")
        trajectory = build_trajectory(len(frame_data), keyframes)
        
        gif_frames = []
        rng = np.random.default_rng(42) # Consistent noise seed per sequence
        
        for i, (gray, low_freq, bg_level, library) in enumerate(frame_data):
            p = float(trajectory[i])
            
            left_margin = int(round(margin_small + p * (extra - 2 * margin_small)))
            left_margin = max(0, min(extra, left_margin))
            right_margin = extra - left_margin
            
            # 1. Outpaint the full frame
            canvas = outpaint_frame(
                gray, left_margin, right_margin, library, low_freq, bg_level,
                PATCH_SIZE, PATCH_OVERLAP, BAND_COUNT, feather_px, rng
            )
            
            # Convert to PIL Image for cropping
            img = Image.fromarray(canvas, mode="L")
            
            # 2. Crop to the safe zone (danger reference lines)
            cropped_img = img.crop((crop_left, 0, crop_right, orig_h))
            
            # 3. Resize back to full dimensions
            resized_img = cropped_img.resize((target_w, orig_h), Image.Resampling.LANCZOS)
            
            gif_frames.append(resized_img)
            
        # Save as a slower GIF (250ms per frame)
        gif_path = os.path.join(output_dir, f"{motion_name}.gif")
        gif_frames[0].save(
            gif_path, 
            save_all=True, 
            append_images=gif_frames[1:], 
            duration=250, 
            loop=0
        )
        print(f"Saved {gif_path}")
        
    print("\nAll 10 motion batch sequences generated successfully!")

if __name__ == "__main__":
    main()
