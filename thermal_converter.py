import sys
import cv2
import numpy as np

def main():
    input_path = r"C:\Users\HP\.gemini\antigravity\brain\54d8efe9-ba98-4b14-abde-a375b6f2a4e0\.user_uploaded\media_1787256347283.jpg"
    output_path = r"C:\Users\HP\.gemini\antigravity\brain\54d8efe9-ba98-4b14-abde-a375b6f2a4e0\thermal_output.png"
    
    print("Reading image...")
    img = cv2.imread(input_path)
    if img is None:
        print(f"Failed to read image at {input_path}")
        sys.exit(1)
        
    print("Segmenting human from background using GrabCut...")
    h, w = img.shape[:2]
    
    # Initialize masks and models for GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    
    # Define a rectangle that encompasses the person
    # (Leaving a small margin to identify the background)
    rect = (int(w*0.1), int(h*0.05), int(w*0.8), int(h*0.95))
    
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
    # Modify mask so that sure and likely background (0 and 2) are 0, else 1
    float_mask = np.where((mask==2)|(mask==0), 0.0, 1.0).astype(np.float32)
    
    # Smooth the mask slightly for better blending
    float_mask = cv2.GaussianBlur(float_mask, (21, 21), 0)
    
    print("Applying thermal effects...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    normalized_gray = gray.astype(float) / 255.0
    
    # Map human to warm colors (0.6 to 1.0) and bg to cool colors (0.0 to 0.3)
    human_intensity = 0.55 + (0.45 * normalized_gray)
    bg_intensity = 0.0 + (0.35 * normalized_gray)
    
    intensity_map = (float_mask * human_intensity) + ((1.0 - float_mask) * bg_intensity)
    intensity_uint8 = (intensity_map * 255).astype(np.uint8)
    
    # Add a blur to simulate heat diffusion typical of thermal cameras
    intensity_uint8 = cv2.GaussianBlur(intensity_uint8, (15, 15), 0)
    
    # Apply JET colormap for classic thermal look (low=blue, high=red)
    thermal_img = cv2.applyColorMap(intensity_uint8, cv2.COLORMAP_JET)
    
    # Pixelate to match the reference style
    pixel_size = 12
    small_img = cv2.resize(thermal_img, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_LINEAR)
    final_img = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_NEAREST)
    
    cv2.imwrite(output_path, final_img)
    print("SUCCESS: Image saved to", output_path)
    
if __name__ == "__main__":
    main()
