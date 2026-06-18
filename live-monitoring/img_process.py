import cv2
import numpy as np
import time
import threading
import os
import sys

# Add the sleeping-monitor folder to the path so yolo_tracker imports cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_tracker import WhitenerTracker

# -----------------------------
# CONFIG
# -----------------------------
IP = "192.168.1.3"
PORT = 8080
URL = f"http://{IP}:{PORT}/video"

WARP_SIZE = 400   # internal processing resolution (not displayed)

# -----------------------------
# GLOBALS
# -----------------------------
class FreshFrameCapture:
    """Threaded capture to avoid buffering lag in OpenCV MJPEG streams."""
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self.ret = False
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
            if not ret:
                time.sleep(0.1)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()



def order_points(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]    # top-left
    rect[2] = pts[np.argmax(s)]    # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    return rect


def nothing(x):
    pass


# -----------------------------
# WARP DESTINATION (fixed grid)
# -----------------------------
DST = np.array([
    [0,          0],
    [WARP_SIZE-1, 0],
    [WARP_SIZE-1, WARP_SIZE-1],
    [0,          WARP_SIZE-1]
], dtype="float32")


def build_transforms(corners):
    M     = cv2.getPerspectiveTransform(corners, DST)
    M_inv = cv2.getPerspectiveTransform(DST, corners)
    return M, M_inv

def find_box_corners(frame):
    """
    Find the cardboard box using HSV color segmentation (orange/brown).
    Erodes mask after gap-filling so the final rectangle hugs the actual
    cardboard edges tightly instead of inflating outward.
    Falls back to Canny if color mask gives too little coverage.
    """
    h, w = frame.shape[:2]
    frame_area = h * w

    # --- Stage 1: HSV color mask for brown/orange cardboard ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, np.array([5,  60,  40]),  np.array([22, 255, 220]))  # brown/tan
    mask2 = cv2.inRange(hsv, np.array([10, 80, 100]),  np.array([30, 255, 255]))  # orange
    mask  = cv2.bitwise_or(mask1, mask2)

    # Close gaps (fills dark logos/text inside the box)
    close_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  close_kern, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kern, iterations=5)
    # Erode back to original extents — removes the inflation caused by closing
    erode_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.erode(mask, erode_kern, iterations=3)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Stage 2: Canny fallback ---
    if not cnts or max(cv2.contourArea(c) for c in cnts) < 0.01 * frame_area:
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges   = cv2.Canny(blurred, 25, 90)
        k2      = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k2, iterations=2)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return None

    # Largest contour within 5 %–90 % of frame area
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    best = None
    for c in cnts:
        ratio = cv2.contourArea(c) / frame_area
        if 0.05 <= ratio <= 0.90:
            best = c
            break

    if best is None:
        return None

    hull = cv2.convexHull(best)
    rect = cv2.minAreaRect(hull)
    box  = cv2.boxPoints(rect).astype("float32")
    return box


def warp_pts_to_frame(pts_warp, M_inv):
    """Transform an array of points from warp-space to original frame-space."""
    pts = np.array(pts_warp, dtype="float32").reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, M_inv).reshape(-1, 2).astype(np.int32)


def draw_safety_zones_on_frame(canvas, left_z, right_z, M_inv):
    """Back-project the safety zone polygons onto the original frame canvas."""
    overlay = canvas.copy()
    h = WARP_SIZE - 1

    # Left danger zone
    l_warp = np.array([[0, 0], [left_z, 0], [left_z, h], [0, h]], dtype="float32")
    l_frame = warp_pts_to_frame(l_warp, M_inv)
    cv2.fillPoly(overlay, [l_frame], (0, 0, 180))

    # Right danger zone
    r_warp = np.array([[right_z, 0], [h, 0], [h, h], [right_z, h]], dtype="float32")
    r_frame = warp_pts_to_frame(r_warp, M_inv)
    cv2.fillPoly(overlay, [r_frame], (0, 0, 180))

    # Safe centre zone
    s_warp = np.array([[left_z, 0], [right_z, 0], [right_z, h], [left_z, h]], dtype="float32")
    s_frame = warp_pts_to_frame(s_warp, M_inv)
    cv2.fillPoly(overlay, [s_frame], (0, 100, 0))

    cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)

    # Boundary lines
    for xw in (left_z, right_z):
        pt_top = warp_pts_to_frame([[xw, 0]], M_inv)[0]
        pt_bot = warp_pts_to_frame([[xw, h]], M_inv)[0]
        cv2.line(canvas, tuple(pt_top), tuple(pt_bot), (0, 160, 255), 2)

    # Box outline
    box_corners_warp = np.array([[0, 0], [WARP_SIZE-1, 0],
                                 [WARP_SIZE-1, WARP_SIZE-1], [0, WARP_SIZE-1]], dtype="float32")
    box_frame = warp_pts_to_frame(box_corners_warp, M_inv)
    cv2.polylines(canvas, [box_frame.reshape(-1, 1, 2)], True, (0, 220, 255), 2)


def draw_blob_on_frame(canvas, x, y, w, h, is_warn, M_inv):
    """Back-project a motion blob bounding box from warp-space to the frame."""
    pts_warp = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype="float32")
    pts_frame = warp_pts_to_frame(pts_warp, M_inv)
    color = (0, 0, 255) if is_warn else (0, 255, 100)
    cv2.polylines(canvas, [pts_frame.reshape(-1, 1, 2)], True, color, 2)
    cx, cy = pts_frame.mean(axis=0).astype(int)
    cv2.drawMarker(canvas, tuple([cx, cy]), color, cv2.MARKER_CROSS, 18, 2)
    cv2.circle(canvas, tuple([cx, cy]), 5, color, -1)


# -----------------------------
# MAIN
# -----------------------------
stream = FreshFrameCapture(URL)

print("=" * 52)
print("  Hospital Bed / Box Monitor")
print(f"  Attempting to connect to: {URL}")
print("  Waiting for video stream...")
print("=" * 52)

# Grab first valid frame
while True:
    ret, frame = stream.read()
    if ret and frame is not None:
        break
    time.sleep(0.1)

# ---- YOLO Whitener Tracker ----
print("  Loading YOLO model (first run downloads ~6 MB)...")
LOG_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "position_log.csv")
tracker = WhitenerTracker(csv_path=LOG_CSV)
print(f"  Position log: {LOG_CSV}")

# ---- Controls window ----
cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Controls", 420, 130)
cv2.createTrackbar("LEFT  margin", "Controls",  60, WARP_SIZE, nothing)
cv2.createTrackbar("RIGHT margin", "Controls", 340, WARP_SIZE, nothing)
cv2.createTrackbar("MIN AREA",     "Controls", 600, 5000,      nothing)

prev_gray_warp = None
fps = 0.0
_fps_t0 = time.time()
_fps_count = 0
_warn_flash = False
_warn_flash_t = 0.0
FLASH_INTERVAL = 0.4

# Sticky warning state — persists until motion is detected back in safe zone
sticky_warn = False
sticky_msg  = ""

# Cached trackbar values (used if Controls window is closed)
_tb_left  = 60
_tb_right = 340
_tb_area  = 600

# -----------------------------
# Manual Box Selection
# -----------------------------
manual_corners = []

def mouse_callback(event, x, y, flags, param):
    global manual_corners
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(manual_corners) < 4:
            manual_corners.append([x, y])
            print(f"  Corner added: ({x}, {y})")

def ensure_controls():
    """Recreate the Controls window & trackbars if the user closed it."""
    global _tb_left, _tb_right, _tb_area
    try:
        l = cv2.getTrackbarPos("LEFT  margin", "Controls")
        r = cv2.getTrackbarPos("RIGHT margin", "Controls")
        a = cv2.getTrackbarPos("MIN AREA",     "Controls")
        if l < 0 or r < 0 or a < 0:
            raise Exception("dead")
        _tb_left, _tb_right, _tb_area = l, r, a
    except Exception:
        cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Controls", 420, 130)
        cv2.createTrackbar("LEFT  margin", "Controls", _tb_left,  WARP_SIZE, nothing)
        cv2.createTrackbar("RIGHT margin", "Controls", _tb_right, WARP_SIZE, nothing)
        cv2.createTrackbar("MIN AREA",     "Controls", _tb_area,  5000,      nothing)
    return _tb_left, _tb_right, _tb_area

# Corner smoothing parameters
smoothed_corners = None
ALPHA = 0.06          # Low = very stable / slow to follow big moves
DEAD_ZONE_PX = 20    # Only update if new detection moved more than this many pixels

# ---- Phase 2: Monitoring loop ----
cv2.namedWindow("Bed Monitor", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Bed Monitor", mouse_callback)
while True:
    ret, frame = stream.read()
    if not ret or frame is None:
        continue

    # FPS
    _fps_count += 1
    if time.time() - _fps_t0 >= 1.0:
        fps = _fps_count / (time.time() - _fps_t0)
        _fps_count = 0
        _fps_t0 = time.time()

    # Read trackbars (safe — recreates window if closed)
    LEFT_Z, RIGHT_Z, MIN_AREA = ensure_controls()
    if LEFT_Z >= RIGHT_Z - 20:
        LEFT_Z = max(0, RIGHT_Z - 20)

    key = cv2.waitKey(1) & 0xFF

    # ---- Reset tracking ----
    if key == ord('r'):
        smoothed_corners = None
        prev_gray_warp = None
        manual_corners.clear()
        print("  Tracking reset.")
        continue

    if key == ord('q'):
        break

    # ---- Manual Box Tracking ----
    if len(manual_corners) == 4:
        raw_corners = np.array(manual_corners, dtype="float32")
    else:
        raw_corners = None

    if raw_corners is not None:
        pts = order_points(raw_corners)
        if smoothed_corners is None:
            smoothed_corners = pts
            prev_gray_warp = None
        else:
            # Dead-zone: only pull toward new detection if it moved enough
            drift = np.linalg.norm(pts - smoothed_corners, axis=1).mean()
            if drift > DEAD_ZONE_PX:
                smoothed_corners = smoothed_corners * (1 - ALPHA) + pts * ALPHA
            # else: corners barely changed — hold current position (ignore MJPEG noise)
            
    if smoothed_corners is None:
        display = frame.copy()
        for pt in manual_corners:
            cv2.circle(display, tuple(pt), 6, (0, 255, 0), -1)
            cv2.circle(display, tuple(pt), 2, (0, 0, 0), -1)
        if len(manual_corners) < 4:
            cv2.putText(display, f"Click 4 corners. Selected: {len(manual_corners)}/4", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Bed Monitor", display)
        continue
        
    M, M_inv = build_transforms(smoothed_corners)

    # ---- Internal: warp to bird's-eye for motion detection ----
    warped = cv2.warpPerspective(frame, M, (WARP_SIZE, WARP_SIZE))
    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    if prev_gray_warp is None:
        prev_gray_warp = gray_w
        continue

    diff = cv2.absdiff(prev_gray_warp, gray_w)
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    kern = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kern)
    thresh = cv2.dilate(thresh, kern, iterations=2)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ---- Draw on FULL ORIGINAL FRAME ----
    display = frame.copy()
    draw_safety_zones_on_frame(display, LEFT_Z, RIGHT_Z, M_inv)

    motion_detected = False
    motion_in_safe  = False
    this_warn       = False
    this_msg        = ""
    blob_positions  = []  # for on-screen debug indicator

    for cnt in cnts:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        is_warn = cx < LEFT_Z or cx > RIGHT_Z
        motion_detected = True
        blob_positions.append((cx, is_warn))
        if is_warn:
            this_warn = True
            this_msg  = "<-- LEFT EDGE WARNING" if cx < LEFT_Z else "RIGHT EDGE WARNING -->"
        else:
            motion_in_safe = True
        draw_blob_on_frame(display, x, y, w, h, is_warn, M_inv)

    # ---- YOLO: detect whitener & update tracker ----
    # Run on every 3rd frame to save CPU (tracker smooths internally)
    box_mask = None
    if smoothed_corners is not None:
        box_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(box_mask, [smoothed_corners.astype(np.int32)], 255)
        # Inflate mask slightly so we track it if it falls just off the edge
        box_mask = cv2.dilate(box_mask, np.ones((50, 50), np.uint8))

    if _fps_count % 3 == 0:
        yolo_bbox = tracker.detect(frame, box_mask=box_mask)
    else:
        yolo_bbox = tracker.last_bbox_frame

    if yolo_bbox is not None and smoothed_corners is not None:
        tracker.update_warp(yolo_bbox, M, WARP_SIZE, LEFT_Z, RIGHT_Z)

    # ---- Warning logic: YOLO-primary, motion-blob fallback ----
    # If YOLO detected the whitener recently, use its warp-space position directly.
    # This clears the warning INSTANTLY when object returns to safe zone.
    yolo_in_danger = False
    yolo_msg       = ""
    yolo_is_fresh  = (len(tracker.history) > 0 and
                      time.time() - tracker.history[-1]["t"] < 2.0)

    if yolo_is_fresh:
        last_wx = tracker.history[-1]["wx"]
        if last_wx < LEFT_Z:
            yolo_in_danger = True
            yolo_msg = "<-- LEFT EDGE WARNING"
        elif last_wx > RIGHT_Z:
            yolo_in_danger = True
            yolo_msg = "RIGHT EDGE WARNING -->"
        # YOLO is active and object is in safe zone — clear any old sticky state
        if not yolo_in_danger:
            sticky_warn = False
            sticky_msg  = ""
    else:
        # YOLO hasn't detected recently — fall back to motion blobs
        if this_warn:
            sticky_warn = True
            sticky_msg  = this_msg
        elif motion_in_safe:
            sticky_warn = False
            sticky_msg  = ""

    any_warn = yolo_in_danger or sticky_warn
    warn_msg = yolo_msg or sticky_msg

    # ---- Warning flash (continuous blink while warning is active) ----
    now = time.time()
    if any_warn:
        if now - _warn_flash_t >= FLASH_INTERVAL:
            _warn_flash = not _warn_flash
            _warn_flash_t = now
    else:
        _warn_flash = False

    fh, fw = display.shape[:2]

    # Show blob position hint: helps user tune LEFT/RIGHT margins
    for i, (bx, bwarn) in enumerate(blob_positions):
        pct  = int(bx / WARP_SIZE * 100)
        col  = (60, 60, 255) if bwarn else (60, 220, 60)
        hint = f"Motion @ {pct}% {'[WARN]' if bwarn else '[SAFE]'}"
        cv2.putText(display, hint, (12, fh - 60 - i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

    prev_gray_warp = gray_w

    # ---- YOLO tracker overlay (trail, predicted point, risk bar) ----
    if smoothed_corners is not None:
        tracker.draw_on_frame(display, M_inv, WARP_SIZE)

    # Solid red banner always visible when warning is active; blinks the text
    if any_warn:
        ov = display.copy()
        cv2.rectangle(ov, (0, 0), (fw, 65), (0, 0, 200), -1)
        cv2.addWeighted(ov, 0.75, display, 0.25, 0, display)
        if _warn_flash:   # only the text blinks, background stays solid
            tsz = cv2.getTextSize(warn_msg, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)[0]
            cv2.putText(display, warn_msg, ((fw - tsz[0]) // 2, 45),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

    # Status strip
    s_col  = (60, 220, 60) if not any_warn else (60, 60, 255)
    s_text = "PATIENT SAFE" if not any_warn else "PATIENT AT RISK"
    cv2.rectangle(display, (0, fh - 42), (fw, fh), (20, 20, 20), -1)
    cv2.putText(display, s_text,
                (12, fh - 14), cv2.FONT_HERSHEY_DUPLEX, 0.8, s_col, 2)
    cv2.putText(display, f"FPS {fps:.1f}  |  [r] reset tracking  [q] quit",
                (fw - 325, fh - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    cv2.imshow("Bed Monitor", display)

tracker.close()
stream.release()
cv2.destroyAllWindows()
