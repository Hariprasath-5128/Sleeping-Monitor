import os
import glob
import cv2

input_dir = r"C:\Projects\sleeping-monitor\preview_animation\00001\IR\uncover"
output_dir = r"C:\Projects\sleeping-monitor\preview_animation_lines\00001\IR\uncover"
os.makedirs(output_dir, exist_ok=True)

image_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))

for path in image_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    h, w = img_color.shape[:2]
    
    # Extended the lines slightly further outwards (smaller danger zone margin)
    # Using 4.5 instead of 6 to push them wider.
    danger_width = int(w * 4.5 / 27)
    
    left_line_x = danger_width
    right_line_x = w - danger_width
    
    cv2.line(img_color, (left_line_x, 0), (left_line_x, h), (255, 0, 0), 2)
    cv2.line(img_color, (right_line_x, 0), (right_line_x, h), (255, 0, 0), 2)
    
    filename = os.path.basename(path)
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, img_color)

print(f"Drew extended lines on {len(image_paths)} images and saved to {output_dir}")
