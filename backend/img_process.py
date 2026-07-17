import cv2
import numpy as np
import time
import threading
import os
import sys
import joblib

# Ensure we can import thermal_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thermal_utils import segment_person, extract_features, feature_vector

# -----------------------------
# CONFIG
# -----------------------------
IP = "ENTER_WIFI_IP_HERE"
PORT = 8080
URL = f"http://{IP}/capture"
RF_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "zone_model_rf.joblib")

# -----------------------------
# GLOBALS
# -----------------------------
import requests
import re
import base64

class ESP32HTTPCapture:
    """Threaded capture that polls an ESP32 HTTP endpoint for base64 images."""
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.ret = False
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            try:
                response = requests.get(self.url, timeout=2.0)
                if response.status_code == 200:
                    html = response.text
                    img_matches = re.findall(r"data:image/jpeg;base64,([A-Za-z0-9+/=\r\n]+)", html)
                    if img_matches:
                        # Take the first matched image
                        b64_data = img_matches[0]
                        img_bytes = base64.b64decode(b64_data)
                        np_arr = np.frombuffer(img_bytes, np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            with self.lock:
                                self.frame = frame
                                self.ret = True
                            continue # Successfully grabbed frame, loop immediately
            except Exception as e:
                pass # Connection failed or timeout, just retry
            
            # If we reached here, we failed to get a frame. Wait before retrying.
            with self.lock:
                self.ret = False
            time.sleep(0.1)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)

    def isOpened(self):
        return True # Always pretend opened so main loop keeps trying to connect

def draw_overlay(frame, mask, feats, probas, labels):
    """Draws segmentation mask, bounding box, probabilities and warnings."""
    display = frame.copy()
    h, w = frame.shape[:2]

    # Draw Mask (Tinted Green)
    if mask is not None:
        green_overlay = np.zeros_like(display)
        green_overlay[mask > 0] = (0, 255, 0)
        cv2.addWeighted(display, 1.0, green_overlay, 0.4, 0, display)

    # If person detected, draw BBox and Warning
    if feats is not None and probas is not None:
        x1, y1, x2, y2 = feats["bbox"]
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Center marker
        cx = int(feats["centroid_x_frac"] * w)
        cy = int(feats["centroid_y_frac"] * h)
        cv2.drawMarker(display, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        # Probabilities block
        y_offset = 30
        highest_prob = 0
        predicted_class = "UNKNOWN"
        for i, label in enumerate(labels):
            p = probas[i] * 100
            text = f"{label}: {p:.1f}%"
            cv2.putText(display, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
            
            if p > highest_prob:
                highest_prob = p
                predicted_class = label

        # Danger Warnings
        if predicted_class == "LEFT":
            cv2.putText(display, "<-- LEFT EDGE WARNING", (10, h - 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 3)
        elif predicted_class == "RIGHT":
            text_size = cv2.getTextSize("RIGHT EDGE WARNING -->", cv2.FONT_HERSHEY_DUPLEX, 1.0, 3)[0]
            cv2.putText(display, "RIGHT EDGE WARNING -->", (w - text_size[0] - 10, h - 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 3)
        else:
            cv2.putText(display, "SAFE (CENTER)", (w // 2 - 100, h - 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)

    return display

def main():
    print("=" * 52)
    print("  Thermal Bed Monitor (RF Model)")
    print(f"  Attempting to connect to: {URL}")
    print("  Waiting for video stream...")
    print("=" * 52)
    
    stream = ESP32HTTPCapture(URL)
    
    # Grab first valid frame
    while True:
        ret, frame = stream.read()
        if ret and frame is not None:
            break
        time.sleep(0.1)
        
    print("  Video stream connected!")
    
    print(f"  Loading RF Model from: {RF_MODEL_PATH}")
    if not os.path.exists(RF_MODEL_PATH):
        print(f"ERROR: Could not find model at {RF_MODEL_PATH}")
        sys.exit(1)
        
    bundle = joblib.load(RF_MODEL_PATH)
    clf = bundle["model"]
    labels = bundle["labels"]
    
    cv2.namedWindow("Thermal Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Thermal Monitor", 800, 600)
    
    fps = 0.0
    _fps_t0 = time.time()
    _fps_count = 0
    
    while True:
        ret, frame = stream.read()
        if not ret or frame is None:
            continue
            
        _fps_count += 1
        if time.time() - _fps_t0 >= 1.0:
            fps = _fps_count / (time.time() - _fps_t0)
            _fps_count = 0
            _fps_t0 = time.time()
            
        # 1. Convert to Grayscale for Thermal processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Segment the person (math-based thresholding, ignores background)
        mask, thresh, plausible = segment_person(gray)
        
        feats = None
        probas = None
        
        if plausible and mask is not None:
            # 3. Extract Features
            feats = extract_features(mask, gray.shape)
            if feats is not None:
                # 4. RF Model Inference
                vec = feature_vector(feats).reshape(1, -1)
                probas = clf.predict_proba(vec)[0]
                
        # 5. Draw overlay
        display = draw_overlay(frame, mask if plausible else None, feats, probas, labels)
        
        # FPS Counter
        cv2.putText(display, f"FPS: {fps:.1f}", (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Thermal Monitor", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
