import os
import glob
from PIL import Image

input_dir = r"C:\Projects\sleeping-monitor\preview_animation\00001\IR\uncover"
output_dir = r"C:\Projects\sleeping-monitor\preview_animation_cropped\00001\IR\uncover"
os.makedirs(output_dir, exist_ok=True)

images = sorted(glob.glob(os.path.join(input_dir, "*.png")))

if images:
    # 1. Create before_cropping.gif
    frames_before = [Image.open(img) for img in images]
    
    # 2. Crop and resize individual images
    frames_after = []
    for img_path in images:
        img = Image.open(img_path)
        w, h = img.size
        
        danger_width = int(w * 4.5 / 27)
        left_x = danger_width
        right_x = w - danger_width
        
        # Crop the image to just the safe zone
        cropped_img = img.crop((left_x, 0, right_x, h))
        
        # Resize the cropped image back to the original full dimensions!
        resized_img = cropped_img.resize((w, h), Image.Resampling.LANCZOS)
        
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, filename)
        resized_img.save(out_path)
        
        frames_after.append(resized_img)
        
    # 3. Create after_cropping.gif with resized frames
    frames_after[0].save(r"C:\Projects\sleeping-monitor\after_cropping_resized.gif", save_all=True, append_images=frames_after[1:], duration=250, loop=0)
    
    print(f"Done generating {len(images)} cropped & resized images.")
