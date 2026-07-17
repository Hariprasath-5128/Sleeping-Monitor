import os
import glob
import numpy as np
from PIL import Image

# Import the compositing logic from the user's provided output_ir.py
from output_ir import load_grayscale, estimate_background_stats, compose, to_rgba, TARGET_WIDTH

def main():
    input_dir = r"C:\Projects\sleeping-monitor\dataset_IR\train\train\00001\IR\uncover"
    output_dir = r"C:\Projects\sleeping-monitor\preview_animation\00001\IR\uncover"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Get all images, sorted
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    num_frames = len(image_paths)
    if num_frames == 0:
        print("No images found in", input_dir)
        return
        
    print(f"Found {num_frames} frames.")
    
    # Get original image dimensions from the first frame
    gray_first, h, w = load_grayscale(image_paths[0])
    extra = TARGET_WIDTH - w
    
    # Define physical keyframes
    # Physical bed: 27cm width, 6cm danger zones (outer ~22% of canvas)
    # Center safe zone is ~15cm in the middle
    center_pos = extra // 2
    left_danger_pos = 40  # Person near the left edge
    
    # We'll use a fixed random seed so the synthetic noise pattern doesn't completely 
    # randomize every frame, preventing unnatural "strobing" backgrounds in the animation.
    rng = np.random.default_rng(42) 
    
    for i, path in enumerate(image_paths):
        progress = i / max(1, num_frames - 1)
        
        # Design a smooth, meaningful trajectory:
        # 1. Stay in safe center for first 20% of the sequence
        # 2. Move smoothly into the left danger zone between 20% and 80%
        # 3. Stay in left danger zone for the final 20%
        if progress < 0.2:
            current_left = center_pos
        elif progress > 0.8:
            current_left = left_danger_pos
        else:
            # Smoothstep interpolation for realistic momentum
            t = (progress - 0.2) / 0.6
            smooth_t = t * t * (3 - 2 * t) 
            current_left = int(center_pos + (left_danger_pos - center_pos) * smooth_t)
            
        current_right = extra - current_left
        
        # Load frame
        gray, _, _ = load_grayscale(path)
        low_freq, _, _, _ = estimate_background_stats(gray)
        
        # Compose using the custom logic
        canvas = compose(gray, low_freq, current_left, current_right, rng)
        out_img = to_rgba(canvas)
        
        # Save frame in original dataset format
        filename = os.path.basename(path)
        out_path = os.path.join(output_dir, filename)
        out_img.save(out_path)
        
        print(f"Processed frame {i+1}/{num_frames}: pos={current_left}")
        
    print(f"\nAnimation sequence saved to: {output_dir}")

if __name__ == "__main__":
    main()
