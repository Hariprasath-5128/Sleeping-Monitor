import glob
from PIL import Image

input_dir = r"C:\Projects\sleeping-monitor\preview_animation_lines\00001\IR\uncover"
images = sorted(glob.glob(f"{input_dir}\\*.png"))
if images:
    frames = [Image.open(img) for img in images]
    # Slower animation: changed duration from 100ms to 250ms
    frames[0].save(r"C:\Projects\sleeping-monitor\preview_lines.gif", save_all=True, append_images=frames[1:], duration=250, loop=0)
