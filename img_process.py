import os as _os
# libjpeg prints "Corrupt JPEG data: premature end of data segment" straight to
# stderr whenever a streamed frame arrives truncated. That is expected on a live
# MJPEG feed and the frame is simply skipped, so keep it out of the log.
_os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
import numpy as np
import time
import csv
import json
import os
import socket
import sys
import threading
from collections import deque
from datetime import datetime

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════
#  CAMERA SOURCE — pick one
# ═════════════════════════════════════════════════════════
# "ask"    : (default) prompt in the terminal, showing which cameras respond
# "phone"  : the IP Webcam Android app (much lower latency, better resolution)
# "esp32"  : the ESP32-CAM running esp/live_monitor/live_monitor.ino
# "auto"   : no prompt — phone if it answers, else the ESP32-CAM
#
# Override per run without editing this file:
#     set CAM_SOURCE=phone
#
# Currently pinned to the phone: it goes straight to IP Webcam with no probing
# and no prompt. To bring the camera picker back, swap the two lines below.
CAM_SOURCE = os.environ.get("CAM_SOURCE", "phone").strip().lower()
# CAM_SOURCE = os.environ.get("CAM_SOURCE", "ask").strip().lower()

# Without a real terminal (piped output, a service) there is nobody to answer
# the prompt, so fall back to automatic selection instead of hanging on input.
if CAM_SOURCE == "ask" and not sys.stdin.isatty():
    CAM_SOURCE = "auto"

# ── ESP32-CAM live stream (esp/live_monitor/live_monitor.ino) ──
# Set CAM_IP to the address the ESP32-CAM prints on its serial monitor.
ESP_IP        = os.environ.get("CAM_IP", "192.168.137.66")
ESP_PORT      = int(os.environ.get("CAM_PORT", "81"))
ESP_STREAM    = os.environ.get("STREAM_URL", f"http://{ESP_IP}:{ESP_PORT}/stream")
# Single-JPEG endpoint on the ESP32-CAM's control server (port 80). Used when
# the MJPEG stream on :81 will not open — that server is the fragile half.
ESP_CAPTURE   = os.environ.get("CAPTURE_URL", f"http://{ESP_IP}/capture")

# ── Phone camera: the "IP Webcam" app (Android) ──
# Start the app, tap "Start server", and it shows an address like
# http://10.184.140.140:8080 — put that host and port here.
#   /video  = MJPEG stream        /shot.jpg = single frame
PHONE_IP      = os.environ.get("PHONE_IP", "172.23.226.131")
PHONE_PORT    = int(os.environ.get("PHONE_PORT", "8080"))
PHONE_STREAM  = os.environ.get("PHONE_STREAM", f"http://{PHONE_IP}:{PHONE_PORT}/video")
PHONE_CAPTURE = os.environ.get("PHONE_CAPTURE", f"http://{PHONE_IP}:{PHONE_PORT}/shot.jpg")

# Resolved below by pick_camera(); the rest of the pipeline uses these.
CAM_IP      = ESP_IP
CAM_PORT    = ESP_PORT
STREAM_URL  = ESP_STREAM
CAPTURE_URL = ESP_CAPTURE

# Seconds to wait for the MJPEG stream before giving up on it.
STREAM_OPEN_TIMEOUT = float(os.environ.get("STREAM_OPEN_TIMEOUT", "12"))

# Fall back to a locally attached webcam when the ESP32-CAM is unreachable.
# Default None: this is a bed monitor, so silently switching to the laptop
# webcam would show the wrong scene while still reporting zones to the ESP32.
# Better to keep retrying the real camera. Set to 0 only for offline testing.
FALLBACK_SOURCE = None if os.environ.get("NO_WEBCAM_FALLBACK", "1") == "1" else 0

# ── Local status server (app.py) ──
# img_process.py writes the zone here; app.py serves it to the motor ESP32.
STATUS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "status.json")
STATUS_TXT  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "status.txt")   # legacy plain-text mirror
STATUS_EVERY_N = 2      # write the status file every N frames

# Live view — img_process.py drops its annotated frame here and app.py serves
# it at /video, so the bed can be watched (and corners set) from the browser.
LIVE_FRAME  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "live_frame.jpg")
LIVE_EVERY_N = 2        # publish every N frames
LIVE_QUALITY = 70       # JPEG quality for the web view
# Where app.py is listening; frames are POSTed here for /video.
LIVE_POST_URL = os.environ.get("LIVE_POST_URL", "http://127.0.0.1:5000/frame")
# Where to read the bed's real tilt (reported by the motor ESP32).
TILT_URL = os.environ.get("TILT_URL", "http://127.0.0.1:5000/tilt")
TILT_POLL_EVERY_N = 10        # frames between tilt refreshes

# How often to rebuild the capture connection while frames are not arriving.
REOPEN_EVERY_S = 15.0

# ── Capture-path latency tuning ──
# The ESP32-CAM answers /capture in 0.02-2.7 s depending on the link. Several
# overlapping workers keep a fresh frame in hand so the monitor loop never
# waits on a slow individual request.
CAPTURE_WORKERS   = int(os.environ.get("CAPTURE_WORKERS", "4"))
# Cap a single request below the 2 s budget: a request already slower than
# this will never yield a "live" frame, so abandon it and start a fresh one
# rather than letting every worker pile up behind one stalled connection.
CAPTURE_TIMEOUT   = float(os.environ.get("CAPTURE_TIMEOUT", "6.0"))
# Frames older than this are considered stale (the 2 s end-to-end budget).
CAPTURE_MAX_AGE   = float(os.environ.get("CAPTURE_MAX_AGE", "2.0"))
# How long read() waits for a fresh frame before handing back what it has.
CAPTURE_READ_WAIT = float(os.environ.get("CAPTURE_READ_WAIT", "1.5"))
# Skip the MJPEG stream and go straight to threaded /capture polling.
PREFER_CAPTURE    = os.environ.get("PREFER_CAPTURE", "0") == "1"
# A struggling camera can take several seconds to answer, so probe patiently
# and retry before writing the stream off.
PROBE_TIMEOUT     = float(os.environ.get("PROBE_TIMEOUT", "6.0"))
PROBE_ATTEMPTS    = int(os.environ.get("PROBE_ATTEMPTS", "3"))

WARP_SIZE      = 640    # internal bird's-eye canvas — also optimal YOLO input size
TRAIL_LEN      = 80     # centroid history for the trail
LOG_EVERY_N    = 5      # CSV rows written every N frames
LOG_FILE       = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "position_log.csv")
FLASH_INTERVAL = 0.4    # warning blink period (s)
LOST_TIMEOUT   = 60     # frames without a detection before re-acquiring
NOT_FOUND_TIMEOUT = 30  # frames lost before declaring NOT FOUND

# Zone smoothing — only commit to a new zone if it appears in majority of last N frames
ZONE_SMOOTH_N  = 8      # rolling window size

# Fraction of the body that must be past a boundary to count as a WARNING.
# A patient lying across the bed has a wide box that clips the danger strip
# while their centre of mass is safe, so a small overlap is not a warning.
WARN_OVERLAP_FRAC = 0.15

# ...and this much of the body over the line is a fall already in progress,
# even if the centroid has not crossed yet (a wide box keeps the centroid
# inside long after the patient is really going over).
DANGER_OVERLAP_FRAC = 0.38

_HERE = os.path.dirname(os.path.abspath(__file__))

# Dual-Model System — both models come from commit 8e887afd
# ("Final working software part").
#
# 1. Visual tracking model (draws the box): yolov8s.pt, committed at the repo
#    root so tracking works with no download.
YOLO_MODEL     = os.environ.get("YOLO_MODEL", os.path.join(_HERE, "yolov8s.pt"))
YOLO_CONF      = 0.20
YOLO_IOU       = 0.45

# On a 320x240 ESP32-CAM feed the same object is re-detected as a different
# class from frame to frame (bottle / vase / tennis racket...), each time with a
# new track ID. So identity is re-established by position: a detection within
# this many warp-space pixels of the last known centroid is treated as the same
# object, whatever YOLO decided to call it this frame.
REACQUIRE_RADIUS = 140

# Reject detections that are really the background. Pointed at a bed (or a
# board on a table) YOLO confidently labels the whole surface as one big
# object — a 47%-of-frame "refrigerator" at 0.91 beats the actual object
# every frame, so the tracker locks onto the furniture and never lets go.
# Anything covering more than this fraction of the warped view is scenery.
MAX_DET_AREA_FRAC = 0.25

# ...and anything smaller than this is noise.
MIN_DET_AREA_FRAC = 0.0002

# 2. The brain: the trained RandomForest risk classifier
#    (backend/models/risk_classifier.pkl, 86.71% test accuracy).
#    It reads motion features — position, gaps to each edge, velocity,
#    acceleration — rather than raw pixels, so it is camera-agnostic and
#    transfers straight from the IP-webcam recordings to the ESP32-CAM feed.
STATE_MODEL    = os.environ.get(
    "STATE_MODEL", os.path.join(_HERE, "backend", "models", "risk_classifier.pkl"))

# The classifier speaks clinical risk; the ESP32 servo logic speaks zones.
# Side (left vs right) comes from the tracked centroid, not the model.
RISK_ALIASES   = {
    "STABLE":        "SAFE",
    "DRIFT WARNING": "WARNING",   # side filled in from geometry below
    "FALL IMMINENT": "DANGER",     # resolved to DANGER_LEFT / DANGER_RIGHT
}

# Ignore a call the model is not reasonably sure about; the previous smoothed
# zone carries over instead of flapping the servos.
STATE_MIN_CONF = 0.50

# ─────────────────────────────────────────────────────────
# CORNER-SELECTION GLOBALS
# ─────────────────────────────────────────────────────────
clicked_pts    = []
selection_done = False


def mouse_callback(event, x, y, flags, param):
    global clicked_pts, selection_done
    if selection_done:
        return
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_pts) < 4:
        clicked_pts.append((x, y))
        if len(clicked_pts) == 4:
            selection_done = True


def order_points(pts):
    pts  = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s         = pts.sum(axis=1)
    rect[0]   = pts[np.argmin(s)]
    rect[2]   = pts[np.argmax(s)]
    diff      = np.diff(pts, axis=1)
    rect[1]   = pts[np.argmin(diff)]
    rect[3]   = pts[np.argmax(diff)]
    return rect


def nothing(_):
    pass


# Fraction of the screen the preview windows are allowed to fill.
DISPLAY_FILL = 0.85


def _screen_size(default=(1536, 864)):
    """Best-effort desktop resolution, falling back to a sane default."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        w, h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return default


SCREEN_W, SCREEN_H = _screen_size()


def display_scale(src_w, src_h):
    """How much to enlarge a camera frame for comfortable on-screen viewing.

    The ESP32-CAM sends 320x240, which is tiny on a modern display. Scaling the
    image up here (rather than letting the window stretch it) keeps the drawn
    overlay — text, corner dots — at a sensible size instead of magnifying it.
    """
    fit = min((SCREEN_W * DISPLAY_FILL) / max(1, src_w),
              (SCREEN_H * DISPLAY_FILL) / max(1, src_h))
    # Cap at 8x: at QQVGA (160x120) a 4x limit left the window at only
    # 640x480, which is hard to click corners on.
    return max(1.0, min(fit, 8.0))


# ─────────────────────────────────────────────────────────
# PERSPECTIVE HELPERS
# ─────────────────────────────────────────────────────────
DST = np.array([
    [0,           0          ],
    [WARP_SIZE-1, 0          ],
    [WARP_SIZE-1, WARP_SIZE-1],
    [0,           WARP_SIZE-1],
], dtype="float32")


def build_transforms(corners):
    M     = cv2.getPerspectiveTransform(corners, DST)
    M_inv = cv2.getPerspectiveTransform(DST, corners)
    return M, M_inv


def warp_to_frame(pts_w, M_inv):
    """Back-project warp-space points → original frame space."""
    arr = np.array(pts_w, dtype="float32").reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, M_inv).reshape(-1, 2).astype(np.int32)


# ─────────────────────────────────────────────────────────
# CORNER SELECTION UI
# ─────────────────────────────────────────────────────────
def run_corner_selection(first_frame, cap_ref):
    global clicked_pts, selection_done
    clicked_pts    = []
    selection_done = False
    WIN    = "Select 4 Corners"
    labels = ["TL", "TR", "BR", "BL"]

    src_h, src_w = first_frame.shape[:2]
    scale = display_scale(src_w, src_h)
    disp_w, disp_h = int(src_w * scale), int(src_h * scale)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)
    cv2.setMouseCallback(WIN, mouse_callback)

    while not selection_done:
        ret, live = cap_ref.read()
        frame_src = live if ret else first_frame

        # Enlarge first, then annotate. Drawing on the small camera frame and
        # letting the window stretch it magnifies every dot and glyph too.
        disp = cv2.resize(frame_src, (disp_w, disp_h),
                          interpolation=cv2.INTER_LINEAR)
        fh, fw = disp.shape[:2]

        idx = min(len(clicked_pts), 3)
        title = f"Click corner {len(clicked_pts)+1}/4  ({labels[idx]})"
        hint  = "Order: TL -> TR -> BR -> BL   |  [r] reset   [q] quit"
        # A 1 px offset shadow reads cleanly at this size; a thick outline
        # closes up thin glyphs and makes the text look smudged.
        cv2.putText(disp, title, (13, 31), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                    (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(disp, title, (12, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                    (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(disp, hint, (13, fh - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(disp, hint, (12, fh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)

        # clicked_pts are window coordinates, so they are already display-space.
        for i, pt in enumerate(clicked_pts):
            cv2.circle(disp, pt, 4, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(disp, pt, 6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(disp, labels[i], (pt[0] + 10, pt[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(disp, labels[i], (pt[0] + 9, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                        cv2.LINE_AA)
            if i > 0:
                cv2.line(disp, clicked_pts[i-1], pt, (0, 255, 0), 1, cv2.LINE_AA)
        if len(clicked_pts) >= 3:
            cv2.line(disp, clicked_pts[-1], clicked_pts[0], (0, 200, 0), 1,
                     cv2.LINE_AA)

        # Mirror the corner picker to the browser as well.
        publish_frame(disp)
        cv2.imshow(WIN, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            clicked_pts    = []
            selection_done = False
        elif key == ord('q'):
            cv2.destroyAllWindows()
            cap_ref.release()
            exit()

    cv2.destroyWindow(WIN)
    # Clicks were made on the enlarged preview; the warp works in camera
    # coordinates, so scale them back down before returning.
    src_pts = [(p[0] / scale, p[1] / scale) for p in clicked_pts]
    return order_points(src_pts)


# ─────────────────────────────────────────────────────────
# ZONE OVERLAY (Visual Only now)
# ─────────────────────────────────────────────────────────
def draw_zones(canvas, lz, rz, M_inv):
    """Draw strong, clearly visible safety zone overlays on the full frame."""
    W = H = WARP_SIZE - 1

    # ── Zone fills (stronger alpha so zones are clearly visible) ──
    overlay = canvas.copy()
    polys = [
        (np.array([[0,0],[lz,0],[lz,H],[0,H]],   dtype="float32"), (0,   0, 200)),  # left danger
        (np.array([[rz,0],[W,0],[W,H],[rz,H]],    dtype="float32"), (0,   0, 200)),  # right danger
        (np.array([[lz,0],[rz,0],[rz,H],[lz,H]],  dtype="float32"), (0, 130,   0)),  # safe centre
    ]
    for pts_w, col in polys:
        cv2.fillPoly(overlay, [warp_to_frame(pts_w, M_inv)], col)
    cv2.addWeighted(overlay, 0.38, canvas, 0.62, 0, canvas)

    # ── Boundary lines — glow effect (thick dark shadow + thin bright core) ──
    for xw in (lz, rz):
        t = warp_to_frame([[xw, 0]], M_inv)[0]
        b = warp_to_frame([[xw, H]], M_inv)[0]
        # Outer glow / shadow
        cv2.line(canvas, tuple(t), tuple(b), (0, 80, 180), 10, cv2.LINE_AA)
        # Bright core
        cv2.line(canvas, tuple(t), tuple(b), (0, 220, 255), 3, cv2.LINE_AA)

    # ── Left / Right Text Markers ──
    l_mid = warp_to_frame([[lz // 2, H // 2]], M_inv)[0]
    r_mid = warp_to_frame([[(rz + W) // 2, H // 2]], M_inv)[0]
    
    cv2.putText(canvas, "<-- LEFT", (l_mid[0]-40, l_mid[1]), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(canvas, "<-- LEFT", (l_mid[0]-40, l_mid[1]), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "RIGHT -->", (r_mid[0]-40, r_mid[1]), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(canvas, "RIGHT -->", (r_mid[0]-40, r_mid[1]), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

    # ── Bed / box outline — bright cyan, thick ──
    box_w = np.array([[0,0],[W,0],[W,H],[0,H]], dtype="float32")
    box_f = warp_to_frame(box_w, M_inv)
    # Shadow
    cv2.polylines(canvas, [box_f.reshape(-1,1,2)], True, (0, 100, 150), 7)
    # Bright outline
    cv2.polylines(canvas, [box_f.reshape(-1,1,2)], True, (0, 230, 255), 3)



# ─────────────────────────────────────────────────────────
# PERSON DRAWING (bounding box + targeting reticle + trail)
# ─────────────────────────────────────────────────────────
def draw_person(canvas, bx, by, bw, bh, cx_w, cy_w,
                zone, trail, M_inv, is_searching=False, conf=None, label="PERSON"):
    if is_searching:
        box_col  = (0, 200, 255)
        text_col = (0, 200, 255)
        status   = "SEARCHING..."
    elif zone == "NOT_FOUND":
        box_col  = (0, 100, 255)
        text_col = (0, 100, 255)
        status   = "NOT FOUND"
    elif zone == "SAFE":
        box_col  = (0, 255, 80)
        text_col = (0, 220, 80)
        status   = "SAFE"
    elif zone.startswith("WARNING"):
        box_col  = (0, 220, 255)
        text_col = (0, 200, 255)
        status   = "WARNING"
    elif zone == "EMPTY":
        box_col  = (150, 150, 150)
        text_col = (150, 150, 150)
        status   = "EMPTY"
    else:
        box_col  = (0, 60, 255)
        text_col = (60, 60, 255)
        status   = zone

    # Back-project bounding box
    box_pts_w = np.array([[bx, by], [bx+bw, by], [bx+bw, by+bh], [bx, by+bh]], dtype="float32")
    box_pts_f = warp_to_frame(box_pts_w, M_inv)

    # Transparent fill
    ov = canvas.copy()
    cv2.fillPoly(ov, [box_pts_f], box_col)
    cv2.addWeighted(ov, 0.12, canvas, 0.88, 0, canvas)

    # Box outline
    cv2.polylines(canvas, [box_pts_f.reshape(-1,1,2)], True, box_col, 3, cv2.LINE_AA)

    # Corner bracket reticles
    tl = tuple(box_pts_f[0]); tr = tuple(box_pts_f[1])
    br = tuple(box_pts_f[2]); bl = tuple(box_pts_f[3])
    span = max(1.0, float(np.linalg.norm(np.array(tr) - np.array(tl))))
    t    = min(18.0 / span, 0.35)

    def lerp(a, b, t):
        return (int(a[0] + (b[0]-a[0])*t), int(a[1] + (b[1]-a[1])*t))

    for corner, nb1, nb2 in [(tl, tr, bl), (tr, tl, br), (br, tr, bl), (bl, tl, br)]:
        cv2.line(canvas, corner, lerp(corner, nb1, t), box_col, 4, cv2.LINE_AA)
        cv2.line(canvas, corner, lerp(corner, nb2, t), box_col, 4, cv2.LINE_AA)

    # Centroid in frame space
    cx_f, cy_f = warp_to_frame([[cx_w, cy_w]], M_inv)[0]

    # Double ring targeting reticle
    cv2.circle(canvas, (cx_f, cy_f), 20, box_col, 1, cv2.LINE_AA)
    cv2.circle(canvas, (cx_f, cy_f), 24, box_col, 1, cv2.LINE_AA)
    cv2.drawMarker(canvas, (cx_f, cy_f), box_col, cv2.MARKER_CROSS, 30, 2, cv2.LINE_AA)
    cv2.circle(canvas, (cx_f, cy_f), 5, box_col, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx_f, cy_f), 5, (255,255,255), 1, cv2.LINE_AA)

    # Zone badge next to box
    cv2.putText(canvas, f"[ {status} ]",
                (box_pts_f[1][0] + 6, box_pts_f[1][1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_col, 2, cv2.LINE_AA)

    # Movement trail — fading coloured polyline
    if len(trail) > 1:
        trail_pts = [warp_to_frame([[p[0], p[1]]], M_inv)[0] for p in trail]
        n = len(trail_pts)
        for i in range(1, n):
            alpha     = i / n
            t_col     = tuple(int(c * alpha) for c in box_col)
            thickness = max(1, int(3 * alpha))
            cv2.line(canvas, tuple(trail_pts[i-1]), tuple(trail_pts[i]),
                     t_col, thickness, cv2.LINE_AA)
        for i, pt in enumerate(trail_pts):
            if i % 5 == 0:
                a = (i / n) * 0.55
                cv2.circle(canvas, tuple(pt), max(1, int(3*a)),
                           tuple(int(c*a) for c in box_col), -1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────
# CONTROLS WINDOW
# ─────────────────────────────────────────────────────────
# conf 12%: the ESP32-CAM feed detects this object between ~0.20 and 0.50,
# so a 20% gate cuts out the dips and breaks the lock mid-track.
_tb = {"left": 150, "right": 490, "conf": 12}


def make_controls():
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 460, 130)
    cv2.createTrackbar("LEFT  margin", "Controls", _tb["left"],  WARP_SIZE, nothing)
    cv2.createTrackbar("RIGHT margin", "Controls", _tb["right"], WARP_SIZE, nothing)
    cv2.createTrackbar("CONF  %",      "Controls", _tb["conf"],  100,       nothing)


def ensure_controls():
    global _tb
    try:
        l = cv2.getTrackbarPos("LEFT  margin", "Controls")
        r = cv2.getTrackbarPos("RIGHT margin", "Controls")
        c = cv2.getTrackbarPos("CONF  %",      "Controls")
        if l < 0 or r < 0 or c < 0:
            raise RuntimeError
        _tb["left"], _tb["right"], _tb["conf"] = l, r, c
    except Exception:
        make_controls()
    return _tb["left"], _tb["right"], max(1, _tb["conf"]) / 100.0


# ─────────────────────────────────────────────────────────
# CSV LOG
# ─────────────────────────────────────────────────────────
def open_log(path):
    is_new = not os.path.isfile(path) or os.path.getsize(path) == 0
    fh     = open(path, "a", newline="")
    writer = csv.writer(fh)
    if is_new:
        writer.writerow(["timestamp", "frame_no",
                         "track_id", "yolo_class", "conf",
                         "warp_cx", "warp_cy",
                         "frame_cx", "frame_cy",
                         "zone", "warning"])
        fh.flush()
    return fh, writer


def log_row(writer, fh, frame_no, tid, cls_name, conf,
            cx_w, cy_w, cx_f, cy_f, zone, warning):
    writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        frame_no, tid, cls_name, f"{conf:.3f}",
        int(cx_w), int(cy_w), int(cx_f), int(cy_f),
        zone, int(warning),
    ])
    fh.flush()


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

print("=" * 58)
print("  Hospital Bed / Patient Monitor  — Dual Model AI")
print("=" * 58)
print("  Loading AI models …")

from ultralytics import YOLO
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Running on: {device.upper()}")

print(f"  Loading Visual Tracker: {os.path.basename(YOLO_MODEL)}")
model = YOLO(YOLO_MODEL)
model.to(device)


# ── The brain: trained RandomForest risk classifier (commit 8e887afd) ──
# Bundle layout written by backend/ml_trainer.py:
#   {"clf": RandomForestClassifier, "le": LabelEncoder, "features": [...]}
state_model = None
risk_le = None
risk_features = []

if os.path.isfile(STATE_MODEL):
    print(f"  Loading Risk Classifier (The Brain): {os.path.basename(STATE_MODEL)}")
    try:
        import pickle
        with open(STATE_MODEL, "rb") as _f:
            _bundle = pickle.load(_f)
        state_model   = _bundle["clf"]
        risk_le       = _bundle["le"]
        risk_features = list(_bundle["features"])
        print(f"  Classes: {list(risk_le.classes_)}")
    except Exception as exc:
        print(f"  Could not load risk classifier ({exc}).")
        print("  Falling back to geometric zoning.")
        state_model = None
else:
    print(f"  Risk classifier not found at: {STATE_MODEL}")
    print("  Falling back to geometric zoning (centroid vs LEFT/RIGHT margins).")


def classify_state(cx_w, cy_w, bw, bh, vx, vy, ax, lz, rz):
    """Score the tracked centroid with the trained RandomForest.

    Feature engineering mirrors backend/ml_trainer.build_features() so the
    vector matches what the model was fitted on. Returns (zone, confidence)
    in the pipeline's zone vocabulary; FALL IMMINENT is resolved to a side
    using which boundary the centroid is closer to.
    """
    safe_w = max(1, rz - lz)
    row = {
        "warp_x":       float(cx_w),
        "warp_y":       float(cy_w),
        "aspect_ratio": float(bw) / float(bh) if bh else 1.0,
        "norm_x":       (cx_w - lz) / safe_w,
        "norm_y":       cy_w / 400.0,          # WARP_SIZE at training time
        "left_gap":     float(cx_w - lz),
        "right_gap":    float(rz - cx_w),
        "in_left":      int(cx_w < lz),
        "in_right":     int(cx_w > rz),
        "vx":           float(vx),
        "vy":           float(vy),
        "speed":        float(np.hypot(vx, vy)),
        "ax":           float(ax),
    }
    X = np.array([[row.get(f, 0.0) for f in risk_features]], dtype=float)

    probs = state_model.predict_proba(X)[0]
    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])
    risk  = str(risk_le.classes_[idx])

    zone = RISK_ALIASES.get(risk, risk)
    if zone == "DANGER":
        # The model says a fall is imminent; the tracker says which way.
        zone = "DANGER_LEFT" if row["left_gap"] <= row["right_gap"] else "DANGER_RIGHT"
    return zone, conf

# ── Camera — ESP32-CAM ──
class MjpegReader:
    """Reads the ESP32-CAM MJPEG stream directly, bypassing FFmpeg.

    OpenCV hands the URL to FFmpeg, which handles this camera's chunked
    multipart response badly: the socket delivers ~12 fps while
    cv2.VideoCapture manages 0.4. Parsing the JPEG frames ourselves off a
    plain socket keeps the full frame rate, and a background thread means
    read() always returns the newest frame without blocking.
    """

    def __init__(self, url):
        from urllib.parse import urlparse
        u = urlparse(url)
        self.host = u.hostname
        self.port = u.port or 80
        self.path = u.path or "/stream"

        self._latest = None
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

        deadline = time.time() + STREAM_OPEN_TIMEOUT
        while time.time() < deadline:
            with self._lock:
                if self._latest is not None:
                    return
            time.sleep(0.05)

    def _pump(self):
        SOI = b"\xff\xd8"          # JPEG start of image
        EOI = b"\xff\xd9"          # JPEG end of image
        while not self._stop.is_set():
            sk = None
            try:
                sk = socket.create_connection((self.host, self.port), timeout=4)
                # Short read timeout: a healthy stream delivers ~12 fps, so
                # 2 s of silence means it has stalled. Reconnecting quickly
                # beats waiting on a dead socket.
                sk.settimeout(2.0)
                sk.sendall(("GET %s HTTP/1.1\r\nHost: %s\r\n"
                            "Connection: keep-alive\r\n\r\n"
                            % (self.path, self.host)).encode())
                buf = b""
                while not self._stop.is_set():
                    chunk = sk.recv(65536)
                    if not chunk:
                        break
                    buf += chunk

                    # Pull out every complete JPEG the buffer now holds.
                    while True:
                        i = buf.find(SOI)
                        if i < 0:
                            break
                        j = buf.find(EOI, i + 2)
                        if j < 0:
                            break
                        jpg = buf[i:j + 2]
                        buf = buf[j + 2:]
                        # A truncated frame decodes to None (or prints
                        # "Corrupt JPEG data"); either way just skip it — the
                        # next frame is milliseconds away.
                        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8),
                                             cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self._lock:
                                self._latest = frame
                                self._stamp = time.time()

                    # Never let a desynced buffer grow without bound.
                    if len(buf) > 1_000_000:
                        buf = buf[-100_000:]
            except OSError:
                pass
            finally:
                if sk is not None:
                    try:
                        sk.close()
                    except OSError:
                        pass
            if not self._stop.is_set():
                time.sleep(0.1)     # reconnect promptly

    def read(self):
        """Newest decoded frame, without waiting.

        Blocking here to wait for a "fresher" frame adds latency rather than
        removing it: the pump thread is already storing frames the instant
        they arrive, so whatever is in hand IS the newest. Only wait when
        nothing has arrived at all yet.
        """
        with self._lock:
            if self._latest is not None:
                return True, self._latest

        deadline = time.time() + CAPTURE_READ_WAIT
        while time.time() < deadline:
            time.sleep(0.005)
            with self._lock:
                if self._latest is not None:
                    return True, self._latest
        return False, None

    def set(self, *_a, **_k):
        return False

    def release(self):
        self._stop.set()


class CapturePoller:
    """Reads single JPEGs from /capture, quacking like cv2.VideoCapture.

    The ESP32-CAM's MJPEG server on :81 is the fragile half; /capture on :80
    stays up. Fetching happens on background threads that keep the newest
    frame in hand, so a slow request never stalls the monitor loop: an
    individual grab can take 2 s on a bad link, but read() returns the most
    recent frame immediately.
    """

    def __init__(self, url, workers=CAPTURE_WORKERS):
        self.url = url
        self._latest = None          # newest decoded frame
        self._stamp = 0.0            # when it arrived
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = []

        try:
            import requests
            self._requests = requests
        except ImportError:
            self._requests = None

        # Several workers overlap their requests, so one slow response does
        # not gate the next: whichever returns first refreshes the frame.
        for _ in range(max(1, workers)):
            t = threading.Thread(target=self._pump, daemon=True)
            t.start()
            self._threads.append(t)

        # Give the first fetch a moment to land.
        deadline = time.time() + 8.0
        while time.time() < deadline:
            with self._lock:
                if self._latest is not None:
                    break
            time.sleep(0.05)

    def _fetch(self):
        if self._requests is not None:
            sess = self._requests.Session()
            while not self._stop.is_set():
                try:
                    r = sess.get(self.url, timeout=CAPTURE_TIMEOUT)
                    if r.status_code == 200 and r.content:
                        return r.content
                except Exception:
                    pass
                if self._stop.wait(0.05):
                    return None
            return None
        from urllib.request import urlopen
        try:
            with urlopen(self.url, timeout=CAPTURE_TIMEOUT) as resp:
                return resp.read()
        except Exception:
            return None

    def _pump(self):
        while not self._stop.is_set():
            data = self._fetch()
            if not data:
                continue
            try:
                frame = cv2.imdecode(np.frombuffer(data, np.uint8),
                                     cv2.IMREAD_COLOR)
            except Exception:
                continue
            if frame is None:
                continue
            with self._lock:
                self._latest = frame
                self._stamp = time.time()

    def read(self):
        """Newest frame, without waiting. See MjpegReader.read()."""
        with self._lock:
            if self._latest is not None:
                return True, self._latest

        deadline = time.time() + CAPTURE_READ_WAIT
        while time.time() < deadline:
            time.sleep(0.005)
            with self._lock:
                if self._latest is not None:
                    return True, self._latest
        return False, None

    def set(self, *_a, **_k):
        return False

    def release(self):
        self._stop.set()


def _mjpeg_responds(host, port, path="/stream", timeout=PROBE_TIMEOUT):
    """True if the stream port actually returns HTTP headers.

    Worth doing before opening the stream: it turns a long stall on a dead
    port into a quick check. A struggling camera often misses the first
    attempt and answers the second, so retry before writing it off.
    """
    for attempt in range(PROBE_ATTEMPTS):
        try:
            with socket.create_connection((host, port), timeout=timeout) as sk:
                sk.settimeout(timeout)
                sk.sendall(("GET %s HTTP/1.1\r\nHost: %s\r\n\r\n"
                            % (path, host)).encode())
                if sk.recv(64).startswith(b"HTTP/"):
                    return True
        except OSError:
            pass
        if attempt + 1 < PROBE_ATTEMPTS:
            time.sleep(0.5)
    return False


def _stream_path():
    """URL path of the selected camera's MJPEG stream."""
    from urllib.parse import urlparse
    return urlparse(STREAM_URL).path or "/stream"


def pick_camera():
    """Decide which camera to use and point the globals at it.

    Returns a short label for logging. "auto" prefers the phone because IP
    Webcam gives far lower latency and a much larger frame than the
    ESP32-CAM's QQVGA, but the ESP32-CAM stays fully supported.
    """
    global CAM_IP, CAM_PORT, STREAM_URL, CAPTURE_URL

    def use_phone():
        global CAM_IP, CAM_PORT, STREAM_URL, CAPTURE_URL
        CAM_IP, CAM_PORT = PHONE_IP, PHONE_PORT
        STREAM_URL, CAPTURE_URL = PHONE_STREAM, PHONE_CAPTURE

    def use_esp():
        global CAM_IP, CAM_PORT, STREAM_URL, CAPTURE_URL
        CAM_IP, CAM_PORT = ESP_IP, ESP_PORT
        STREAM_URL, CAPTURE_URL = ESP_STREAM, ESP_CAPTURE

    if CAM_SOURCE == "phone":
        use_phone()
        print(f"  Camera source: PHONE (IP Webcam) at {PHONE_IP}:{PHONE_PORT}")
        return "phone"

    if CAM_SOURCE == "esp32":
        use_esp()
        print(f"  Camera source: ESP32-CAM at {ESP_IP}")
        return "esp32"

    # "ask" (the default): probe both, then let the operator choose. Showing
    # which cameras actually responded avoids picking one that is not on.
    if CAM_SOURCE == "ask":
        print()
        print("  Checking which cameras are available...")
        phone_up = _mjpeg_responds(PHONE_IP, PHONE_PORT, "/video")
        esp_up   = _mjpeg_responds(ESP_IP, ESP_PORT, "/stream")

        p_mark = "ONLINE " if phone_up else "offline"
        e_mark = "ONLINE " if esp_up else "offline"
        # Plain ASCII: the Windows console is cp1252 and box-drawing
        # characters raise UnicodeEncodeError there.
        print()
        print("  " + "=" * 60)
        print("   SELECT CAMERA")
        print("  " + "-" * 60)
        print("   1) Phone - IP Webcam   %s:%d   [%s]"
              % (PHONE_IP, PHONE_PORT, p_mark))
        print("        1920x1080, ~30 fps, lowest latency")
        print("   2) ESP32-CAM           %s   [%s]" % (ESP_IP, e_mark))
        print("        160x120, ~12 fps")
        print("  " + "=" * 60)

        # Default to whichever is up (phone first) so Enter does the sane thing.
        default = "1" if phone_up else ("2" if esp_up else "1")
        while True:
            try:
                choice = input(f"  Choice [1/2] (Enter = {default}): ").strip()
            except (EOFError, KeyboardInterrupt):
                choice = ""
            choice = choice or default
            if choice in ("1", "2"):
                break
            print("  Please type 1 or 2.")

        if choice == "1":
            use_phone()
            if not phone_up:
                print(f"  NOTE: phone did not answer — is IP Webcam started?")
            print(f"  Camera source: PHONE (IP Webcam) at {PHONE_IP}:{PHONE_PORT}")
            return "phone"

        use_esp()
        if not esp_up:
            print("  NOTE: ESP32-CAM stream did not answer; will try /capture.")
        print(f"  Camera source: ESP32-CAM at {ESP_IP}")
        return "esp32"

    # auto: whichever answers, phone first. No prompt.
    if _mjpeg_responds(PHONE_IP, PHONE_PORT, "/video"):
        use_phone()
        print(f"  Camera source: PHONE (IP Webcam) at {PHONE_IP}:{PHONE_PORT}")
        return "phone"

    use_esp()
    print(f"  Phone camera not answering at {PHONE_IP}:{PHONE_PORT}")
    print(f"  Camera source: ESP32-CAM at {ESP_IP}")
    return "esp32"


def open_stream():
    """Open the selected camera: MJPEG first, then single-frame polling."""
    pick_camera()

    # PREFER_CAPTURE skips the MJPEG attempt entirely. Worth setting when the
    # camera's stream server is unreliable: the threaded poller gives steadier
    # latency than a stream that keeps collapsing and reconnecting.
    if PREFER_CAPTURE:
        print(f"  PREFER_CAPTURE set — using {CAPTURE_URL}")
    elif _mjpeg_responds(CAM_IP, CAM_PORT, _stream_path()):
        print(f"  Opening stream: {STREAM_URL}")
        # Direct MJPEG parsing, not cv2.VideoCapture: FFmpeg only manages
        # ~0.4 fps on this camera's chunked response, against ~12 fps read
        # straight off the socket.
        c = MjpegReader(STREAM_URL)
        # The constructor waits for the first frame, but give read() a couple
        # of tries: the very first decode can land just after that window.
        ok, frm = False, None
        for _ in range(3):
            ok, frm = c.read()
            if ok and frm is not None:
                break
        if ok and frm is not None:
            print("  Stream is live.")
            return c, frm, True

        c.release()
        print("  MJPEG stream opened but sent no frames.")
    else:
        print(f"  No MJPEG server on {CAM_IP}:{CAM_PORT} — skipping it.")

    # Fall back to polling single frames on the control server.
    print(f"  Trying single-frame capture: {CAPTURE_URL}")
    poller = CapturePoller(CAPTURE_URL)
    t0 = time.time()
    while time.time() - t0 < 10.0:
        ok, frm = poller.read()
        if ok:
            print("  Single-frame polling is live.")
            print("  (Lower frame rate than MJPEG, but the pipeline is identical.)")
            return poller, frm, True
        time.sleep(0.3)
    poller.release()

    if FALLBACK_SOURCE is None:
        raise SystemExit(f"  Could not reach the camera at {CAM_IP}:{CAM_PORT}.")

    print(f"  ESP32-CAM unreachable — falling back to local camera {FALLBACK_SOURCE}.")
    c = cv2.VideoCapture(FALLBACK_SOURCE)
    try:
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    t0 = time.time()
    while time.time() - t0 < 10.0:
        ok, frm = c.read()
        if ok and frm is not None:
            return c, frm, False
        time.sleep(0.2)

    raise SystemExit("  No video source available (ESP32-CAM and fallback both failed).")


cap, first_frame, stream_ok = open_stream()


# ── Status publishing — read by app.py, served to the motor ESP32 ──
def publish_status(zone, conf, cx, cy, tracking, frame_no):
    """Atomically write the current zone so app.py never reads a half file."""
    payload = {
        "status": zone,
        "confidence": float(conf),
        "cx": int(cx) if cx is not None else -1,
        "cy": int(cy) if cy is not None else -1,
        "tracking": bool(tracking),
        "stream_ok": bool(stream_ok),
        "frame": int(frame_no),
        "ts": time.time(),
    }
    tmp = STATUS_JSON + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, STATUS_JSON)
    except OSError:
        pass
    # Keep the legacy plain-text file in sync for anything still reading it.
    try:
        with open(STATUS_TXT, "w", encoding="utf-8") as f:
            f.write(zone)
    except OSError:
        pass


_frame_session = None
_frame_post_ok = True


_frame_q = None
_frame_worker = None


def _frame_sender():
    """Background sender for the browser view.

    Encoding and POSTing used to happen inline in the monitor loop, so every
    published frame cost a JPEG encode plus a blocking HTTP request (up to a
    second). That delay landed straight on the live view AND on the zone the
    ESP32 reads. Now the loop just drops the newest frame in a 1-slot queue.
    """
    global _frame_post_ok
    sess = None
    try:
        import requests
        sess = requests.Session()
    except ImportError:
        _frame_post_ok = False

    while True:
        img = _frame_q.get()
        if img is None:
            return
        try:
            ok, buf = cv2.imencode(".jpg", img,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), LIVE_QUALITY])
            if not ok:
                continue
            jpg = buf.tobytes()
        except cv2.error:
            continue

        if sess is not None:
            try:
                sess.post(LIVE_POST_URL, data=jpg, timeout=2.0,
                          headers={"Content-Type": "image/jpeg"})
                continue
            except Exception:
                pass   # server down; fall through to the file

        try:
            tmp = LIVE_FRAME + ".tmp"
            with open(tmp, "wb") as f:
                f.write(jpg)
            for _ in range(3):
                try:
                    os.replace(tmp, LIVE_FRAME)
                    break
                except PermissionError:
                    time.sleep(0.005)
        except OSError:
            pass


def publish_frame(img):
    """Hand the newest frame to the background sender. Never blocks."""
    global _frame_q, _frame_worker
    if _frame_q is None:
        import queue
        _frame_q = queue.Queue(maxsize=1)
        _frame_worker = threading.Thread(target=_frame_sender, daemon=True)
        _frame_worker.start()

    # Keep only the newest frame: if the sender is still busy, the older one
    # is already stale and worth dropping.
    try:
        _frame_q.put_nowait(img)
    except Exception:
        try:
            _frame_q.get_nowait()
            _frame_q.put_nowait(img)
        except Exception:
            pass


# Latest tilt reported by the motor ESP32, fetched from app.py.
tilt_state = {"left": 0.0, "right": 0.0, "stale": True}
_tilt_session = None


def fetch_tilt():
    """Read the bed's real tilt from app.py. Never blocks for long."""
    global _tilt_session
    try:
        if _tilt_session is None:
            import requests
            _tilt_session = requests.Session()
        r = _tilt_session.get(TILT_URL, timeout=0.4)
        if r.status_code == 200:
            d = r.json()
            tilt_state["left"]  = float(d.get("left", 0.0))
            tilt_state["right"] = float(d.get("right", 0.0))
            tilt_state["stale"] = bool(d.get("stale", True))
            return
    except Exception:
        pass
    tilt_state["stale"] = True


def zone_from_geometry(cx_w, lz, rz, tracking, bx=None, bw=None):
    """Derive the zone from how much of the body has crossed a boundary.

    Judged on the SHARE of the body past the line, not on its outermost pixel
    and not on the centroid alone:

      * A patient lying across the bed has a wide box whose tip clips the
        danger strip while their mass is safe - that must not raise anything.
      * A patient who is genuinely half over the edge is falling even though
        their centroid has not crossed the line yet, because the box is wide.
    """
    if not tracking or cx_w is None:
        return "EMPTY"

    # Centroid fully across is unambiguous: always DANGER.
    if cx_w < lz:
        return "DANGER_LEFT"
    if cx_w > rz:
        return "DANGER_RIGHT"

    # Without a box the centroid is all we have, and it is inside.
    if bx is None or bw is None or bw <= 0:
        return "SAFE"

    left, right = bx, bx + bw
    over_left  = max(0, lz - left)  / float(bw)
    over_right = max(0, right - rz) / float(bw)

    # A large share of the body past the line is a fall already happening,
    # even while the centroid is still (just) inside.
    if over_left >= DANGER_OVERLAP_FRAC:
        return "DANGER_LEFT"
    if over_right >= DANGER_OVERLAP_FRAC:
        return "DANGER_RIGHT"

    # A smaller share is a lean worth bracing for — and the side matters, so
    # that only the threatened edge is raised.
    if over_left >= WARN_OVERLAP_FRAC and over_left >= over_right:
        return "WARNING_LEFT"
    if over_right >= WARN_OVERLAP_FRAC:
        return "WARNING_RIGHT"

    return "SAFE"


# ── Corner selection ──
print("  Click the 4 corners of the bed/box surface.")
corners  = run_corner_selection(first_frame, cap)
M, M_inv = build_transforms(corners)
print("  Surface defined!  Monitoring started.")
print("  Keys: [r] reselect corners  |  [t] reset tracker  |  [q] quit")
print(f"  Logging to: {LOG_FILE}")

make_controls()
log_fh, log_writer = open_log(LOG_FILE)

# ── Tracking state ──
locked_id    = None    # YOLO track ID we are following
last_bbox_w  = None    # last known warp-space bbox (x,y,w,h)
last_cx_w    = None
last_cy_w    = None
last_label   = "PERSON"
last_conf    = 0.0
trail        = deque(maxlen=TRAIL_LEN)
lost_frames  = 0

sticky_warn  = False
sticky_msg   = ""
sticky_zone  = "SAFE"
current_zone = "SAFE"   # seeded so frame 1 has a value to fall back to

# Motion features for the risk classifier (per-frame deltas of the centroid).
vx = vy = ax = 0.0
prev_vx = 0.0

# Zone smoothing buffer
from collections import Counter
zone_history = []

fps          = 0.0
_fps_t0      = time.time()
_fps_n       = 0
flash_on     = False
flash_t      = 0.0
tracking_ok  = False

frame_no     = 0
log_counter  = 0
status_counter = 0
live_counter = 0
tilt_counter = 0
read_fail    = 0
last_fail_report = 0.0
last_reopen      = time.time()

# Size the monitor window to the screen rather than to the tiny camera frame.
monitor_scale = display_scale(first_frame.shape[1], first_frame.shape[0])
# Multiplying M_inv by this maps warp-space points onto the enlarged canvas.
SCALE_MAT = np.array([[monitor_scale, 0.0, 0.0],
                      [0.0, monitor_scale, 0.0],
                      [0.0, 0.0, 1.0]], dtype="float32")
cv2.namedWindow("Bed Monitor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Bed Monitor",
                 int(first_frame.shape[1] * monitor_scale),
                 int(first_frame.shape[0] * monitor_scale))

# Announce an initial SAFE so the ESP32 has something to read immediately.
publish_status("SAFE", 0.0, None, None, False, 0)

# ─────────────────────────────────────────────────────────
# MONITORING LOOP
# ─────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        # The ESP32-CAM dropped the connection (reboot, WiFi blip). Tell the
        # ESP32 the feed is gone, then try to pick the stream back up.
        read_fail += 1

        # A single-frame poller fails one read at a time, so the old
        # "reconnect after 30 consecutive misses" rule never fired for it and
        # the loop just spun printing the same line. Reconnect on a timer
        # instead, and only report occasionally.
        now_fail = time.time()
        if read_fail == 1 or now_fail - last_fail_report >= 5.0:
            last_fail_report = now_fail
            print(f"  [STREAM] no frame ({read_fail} misses) — retrying…")
            publish_status("NOT_FOUND", 0.0, last_cx_w, last_cy_w, False, frame_no)

        if now_fail - last_reopen >= REOPEN_EVERY_S:
            last_reopen = now_fail
            read_fail = 0
            # Reopen the same way we opened originally, so a dropped MJPEG
            # stream can come back as /capture polling rather than dying.
            try:
                cap.release()
            except Exception:
                pass
            time.sleep(1.0)
            try:
                cap, _frm, stream_ok = open_stream()
                last_reopen = time.time()
            except SystemExit:
                # No webcam fallback configured, which is what we want: keep
                # the old handle and try the real camera again next cycle
                # rather than quietly monitoring the wrong scene.
                print("  [STREAM] ESP32-CAM unreachable — will keep retrying.")
                last_reopen = time.time()
        else:
            time.sleep(0.3)   # do not hammer a struggling camera
        continue

    read_fail = 0

    # Only a real cv2.VideoCapture buffers frames and needs draining. The
    # threaded readers already hand back the newest frame, so re-reading them
    # twice per iteration just burned time and could block on a slow link.
    if isinstance(cap, cv2.VideoCapture):
        for _ in range(2):
            ret2, f2 = cap.read()
            if ret2:
                frame = f2

    frame_no += 1

    # FPS counter
    _fps_n += 1
    now = time.time()
    if now - _fps_t0 >= 1.0:
        fps    = _fps_n / (now - _fps_t0)
        _fps_n = 0
        _fps_t0 = now

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # ── Reselect corners ──
    if key == ord('r'):
        corners  = run_corner_selection(frame, cap)
        M, M_inv = build_transforms(corners)
        locked_id   = None
        last_bbox_w = last_cx_w = last_cy_w = None
        trail.clear(); lost_frames = 0; tracking_ok = False
        sticky_warn = False; sticky_msg = ""; sticky_zone = "SAFE"
        print("  Corners reselected.")
        continue

    # ── Reset tracker ──
    if key == ord('t'):
        locked_id   = None
        last_bbox_w = last_cx_w = last_cy_w = None
        trail.clear(); lost_frames = 0; tracking_ok = False
        sticky_warn = False; sticky_msg = ""; sticky_zone = "SAFE"
        print("  Tracker reset.")
        continue

    # Read controls
    LZ, RZ, conf_thresh = ensure_controls()
    if LZ >= RZ - 30:
        LZ = max(0, RZ - 30)

    # ── Warp to bird's-eye ──
    warped = cv2.warpPerspective(frame, M, (WARP_SIZE, WARP_SIZE))

    # ── 1. YOLOv8 Visual Tracking ──
    results = model.track(warped,
                          persist=True,
                          conf=conf_thresh,
                          iou=YOLO_IOU,
                          verbose=False,
                          device=device)

    # Enlarge the frame BEFORE annotating. Drawing on the small camera image
    # and upscaling afterwards magnifies every glyph and marker with it, which
    # is why the banners used to swamp the view.
    if monitor_scale > 1.0:
        display = cv2.resize(frame, None, fx=monitor_scale, fy=monitor_scale,
                             interpolation=cv2.INTER_LINEAR)
    else:
        display = frame.copy()

    # Overlay geometry is projected through M_inv into camera space, so scale
    # the projection to match the enlarged canvas.
    M_inv_disp = SCALE_MAT @ M_inv if monitor_scale > 1.0 else M_inv

    draw_zones(display, LZ, RZ, M_inv_disp)

    # ── Parse YOLO tracking results ──
    detections = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        if len(boxes):
            xywhs = boxes.xywh.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss  = boxes.cls.cpu().numpy().astype(int)
            # ByteTrack only issues an ID once a detection is confirmed across
            # frames, so a real object often appears with id == None for the
            # first few frames. Dropping those frames is what made the tracker
            # sit in SEARCHING while the object was plainly visible — keep them
            # and mark the untracked ones with id -1.
            if boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
            else:
                ids = np.full(len(xywhs), -1, dtype=int)

            frame_area = float(WARP_SIZE * WARP_SIZE)
            for i in range(len(xywhs)):
                x1 = int(xywhs[i,0] - xywhs[i,2]/2)
                y1 = int(xywhs[i,1] - xywhs[i,3]/2)
                bw = int(xywhs[i,2])
                bh = int(xywhs[i,3])

                # Drop the scenery. A box covering half the view is the bed,
                # the tray or the wall — never the patient we are tracking.
                area_frac = (bw * bh) / frame_area
                if area_frac > MAX_DET_AREA_FRAC or area_frac < MIN_DET_AREA_FRAC:
                    continue

                detections.append({
                    "id":    int(ids[i]),
                    "bbox":  (x1, y1, bw, bh),
                    "cx":    int(xywhs[i,0]),
                    "cy":    int(xywhs[i,1]),
                    "conf":  float(confs[i]),
                    "class": model.names[clss[i]],
                })

    chosen = None
    if locked_id is not None and locked_id >= 0:
        for d in detections:
            if d["id"] == locked_id:
                chosen = d
                break

    # The ID often changes on this feed — the same bottle gets re-detected as a
    # different class with a fresh track ID. If the locked ID is gone, re-acquire
    # by proximity to where we last saw it rather than declaring the object lost.
    if chosen is None and detections and last_cx_w is not None:
        near = min(detections,
                   key=lambda d: (d["cx"] - last_cx_w) ** 2
                                 + (d["cy"] - last_cy_w) ** 2)
        dist = ((near["cx"] - last_cx_w) ** 2
                + (near["cy"] - last_cy_w) ** 2) ** 0.5
        if dist <= REACQUIRE_RADIUS:
            chosen = near
            if near["id"] != locked_id and near["id"] >= 0:
                locked_id = near["id"]
            lost_frames = 0

    if chosen is None and detections:
        chosen      = max(detections, key=lambda d: d["conf"])
        lost_frames = 0
        # id -1 means the tracker has not confirmed this detection yet. Storing
        # it as the lock would be meaningless (every unconfirmed box shares it),
        # so stay unlocked and let position re-acquisition carry identity until
        # a real ID appears.
        if chosen["id"] >= 0:
            locked_id = chosen["id"]
            print(f"  Locked onto track ID={locked_id} "
                  f"({chosen['class']} @ {chosen['conf']:.0%})")
        else:
            locked_id = None

    # ── 2. Classification Brain (or geometric fallback) ──
    geo_zone = zone_from_geometry(
        chosen["cx"] if chosen is not None else last_cx_w,
        LZ, RZ,
        chosen is not None,
        # Box edges, so a body fully inside the safe zone reads SAFE.
        bx=chosen["bbox"][0] if chosen is not None else None,
        bw=chosen["bbox"][2] if chosen is not None else None,
    )

    if state_model is not None and chosen is not None:
        _bx, _by, _bw, _bh = chosen["bbox"]
        raw_zone, state_conf = classify_state(
            chosen["cx"], chosen["cy"], _bw, _bh, vx, vy, ax, LZ, RZ)
        # An unconfident DANGER call is worse than no call — a false sweep is
        # disruptive. Fall back to geometry, which is unambiguous about where
        # the centroid actually is, rather than freezing on the old zone.
        if state_conf < STATE_MIN_CONF:
            raw_zone = geo_zone
        # The classifier reports risk, not proximity. Geometry supplies the
        # near-edge WARNING that earns a gentle nudge before a full sweep.
        elif raw_zone == "SAFE" and geo_zone.startswith("WARNING"):
            raw_zone = geo_zone
        elif raw_zone == "WARNING":
            # The classifier has no notion of side. Prefer geometry's verdict;
            # otherwise decide from which boundary the body is nearer. The old
            # code defaulted to WARNING_LEFT, which braced the wrong motor for
            # anything happening on the right.
            if geo_zone.startswith("WARNING"):
                raw_zone = geo_zone
            else:
                mid = (LZ + RZ) / 2.0
                raw_zone = ("WARNING_LEFT" if chosen["cx"] < mid
                            else "WARNING_RIGHT")

        # GEOMETRY HAS THE FINAL SAY ON DANGER.
        #
        # The classifier was trained on a different bed, camera and object, so
        # it happily calls DANGER while the body is plainly inside the green
        # zone. Where the object physically is, is not a matter of opinion:
        # if the whole body is inside the boundaries it is not falling, no
        # matter how confident the model is.
        if raw_zone.startswith("DANGER") and geo_zone in ("SAFE", "EMPTY"):
            raw_zone = geo_zone
        # ...and a DANGER call must at least agree on which side.
        elif raw_zone == "DANGER_LEFT" and geo_zone == "DANGER_RIGHT":
            raw_zone = geo_zone
        elif raw_zone == "DANGER_RIGHT" and geo_zone == "DANGER_LEFT":
            raw_zone = geo_zone
    else:
        # No detection this frame (or no model): geometry is all we have.
        raw_zone, state_conf = geo_zone, 0.0

    # Nothing is on the bed: trust the tracker over the classifier, which was
    # trained on occupied frames and will still shout LEFT/RIGHT at an empty one.
    if chosen is None and lost_frames >= NOT_FOUND_TIMEOUT:
        raw_zone = "NOT_FOUND"

    # Majority-vote smoothing: only commit if dominant in last N frames
    zone_history.append(raw_zone)
    if len(zone_history) > ZONE_SMOOTH_N:
        zone_history.pop(0)
    zone_counts = Counter(zone_history)
    current_zone = zone_counts.most_common(1)[0][0]

    if chosen is not None:
        bx, by, bw, bh = chosen["bbox"]
        cx_w  = chosen["cx"]
        cy_w  = chosen["cy"]

        bx = max(0, min(bx, WARP_SIZE - 1))
        by = max(0, min(by, WARP_SIZE - 1))
        bw = min(bw, WARP_SIZE - bx)
        bh = min(bh, WARP_SIZE - by)
        cx_w = max(0, min(cx_w, WARP_SIZE - 1))
        cy_w = max(0, min(cy_w, WARP_SIZE - 1))

        if last_cx_w is not None:
            cx_w = int(0.8 * cx_w + 0.2 * last_cx_w)
            cy_w = int(0.8 * cy_w + 0.2 * last_cy_w)

        # Per-frame motion, used by the risk classifier on the next pass.
        if last_cx_w is not None:
            vx, vy  = float(cx_w - last_cx_w), float(cy_w - last_cy_w)
            ax      = vx - prev_vx
            prev_vx = vx
        else:
            vx = vy = ax = 0.0
            prev_vx = 0.0

        last_cx_w, last_cy_w = cx_w, cy_w
        last_bbox_w  = (bx, by, bw, bh)
        last_label   = chosen["class"]
        last_conf    = chosen["conf"]
        tracking_ok  = True
        lost_frames  = 0

        trail.append((cx_w, cy_w))

        draw_person(display, bx, by, bw, bh, cx_w, cy_w,
                    current_zone, trail, M_inv_disp,
                    conf=last_conf, label=last_label.upper())

        # CSV log
        cx_f, cy_f = warp_to_frame([[cx_w, cy_w]], M_inv)[0]
        log_counter += 1
        if log_counter >= LOG_EVERY_N:
            log_counter = 0
            log_row(log_writer, log_fh, frame_no,
                    chosen["id"], last_label, last_conf,
                    cx_w, cy_w, cx_f, cy_f,
                    current_zone, sticky_warn)

    else:
        lost_frames += 1
        tracking_ok  = False
        if lost_frames >= LOST_TIMEOUT:
            locked_id = None
            lost_frames = 0

        # After NOT_FOUND_TIMEOUT frames, override zone to NOT_FOUND
        if lost_frames >= NOT_FOUND_TIMEOUT:
            current_zone = "NOT_FOUND"
            zone_history.clear()   # flush smoothing buffer so it reacts fast on re-detection

        if last_bbox_w is not None:
            bx, by, bw, bh = last_bbox_w
            draw_person(display, bx, by, bw, bh,
                        last_cx_w, last_cy_w,
                        "SEARCHING" if lost_frames < NOT_FOUND_TIMEOUT else "NOT_FOUND",
                        trail, M_inv_disp, is_searching=True,
                        label=last_label.upper())

    # Sticky warning logic from classification
    # NOT_FOUND is deliberately NOT an alarm state. YOLO drops the object for
    # a few frames all the time; treating that as "may have fallen" produced
    # constant false alerts while the object was plainly on the bed.
    if (current_zone.startswith("DANGER") or current_zone.startswith("WARNING")):
        sticky_warn = True
        if current_zone == "DANGER_LEFT":
            sticky_msg = "<-- ABOUT TO FALL LEFT"
        elif current_zone == "DANGER_RIGHT":
            sticky_msg = "ABOUT TO FALL RIGHT -->"
        elif current_zone == "WARNING_LEFT":
            sticky_msg = "WARNING: NEAR LEFT EDGE"
        elif current_zone == "WARNING_RIGHT":
            sticky_msg = "WARNING: NEAR RIGHT EDGE"
        else:
            sticky_msg = "WARNING: NEAR EDGE"
        sticky_zone = current_zone
    else:
        sticky_warn = False
        sticky_msg  = ""
        sticky_zone = current_zone

    # ── Flash timer ──
    if sticky_warn:
        if now - flash_t >= FLASH_INTERVAL:
            flash_on = not flash_on
            flash_t  = now
    else:
        flash_on = False

    fh_d, fw_d = display.shape[:2]

    # ── Warning banner ──
    if sticky_warn:
        ov = display.copy()
        cv2.rectangle(ov, (0, 0), (fw_d, 72), (0, 0, 200), -1)
        cv2.addWeighted(ov, 0.82, display, 0.18, 0, display)
        for sx in (18, fw_d - 46):
            cv2.putText(display, "!", (sx, 56),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 200, 0), 3)
        if flash_on:
            tsz = cv2.getTextSize(sticky_msg, cv2.FONT_HERSHEY_DUPLEX, 0.95, 2)[0]
            cv2.putText(display, sticky_msg,
                        ((fw_d - tsz[0]) // 2, 52),
                        cv2.FONT_HERSHEY_DUPLEX, 0.95, (255,255,255), 2, cv2.LINE_AA)

    # ── Zone badge (top-right) ──
    if sticky_zone == "SAFE":
        badge_text = "STABLE"
        badge_col  = (0, 210, 80)
    elif sticky_zone == "EMPTY":
        badge_text = "EMPTY BED"
        badge_col  = (150, 150, 150)
    elif sticky_zone.startswith("WARNING"):
        badge_text = sticky_zone.replace("_", " ")
        badge_col  = (0, 200, 255)
    elif sticky_zone == "NOT_FOUND":
        badge_text = "NOT FOUND"
        badge_col  = (0, 100, 255)   # orange-red
    else:
        badge_text = f"DANGER ({sticky_zone})"
        badge_col  = (60, 60, 255)

    bsz = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    bx0 = fw_d - bsz[0] - 20; by0 = 12
    cv2.rectangle(display, (bx0 - 6, by0),
                  (fw_d - 8, by0 + bsz[1] + 10), (30,30,30), -1)
    cv2.putText(display, badge_text, (bx0, by0 + bsz[1] + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, badge_col, 2, cv2.LINE_AA)

    # ── Tracker status (top-left) ──
    if tracking_ok:
        ts_txt = f"TRACKING  ID:{locked_id}"
        ts_col = (0, 220, 0)
    elif last_bbox_w is not None:
        ts_txt = f"SEARCHING  (lost {lost_frames} fr)"
        ts_col = (0, 180, 255)
    else:
        ts_txt = "ACQUIRING …"
        ts_col = (0, 180, 255)
    cv2.putText(display, ts_txt, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, ts_col, 2, cv2.LINE_AA)

    # ── Bed tilt, as reported by the motor ESP32 ──
    tilt_counter += 1
    if tilt_counter >= TILT_POLL_EVERY_N:
        tilt_counter = 0
        fetch_tilt()

    if tilt_state["stale"]:
        tilt_txt, tilt_col = "TILT  --  /  --", (140, 140, 140)
    else:
        tilt_txt = "TILT  L %.0f%s  R %.0f%s" % (
            tilt_state["left"], chr(176), tilt_state["right"], chr(176))
        raised = max(tilt_state["left"], tilt_state["right"]) > 5.0
        tilt_col = (0, 200, 255) if raised else (0, 220, 0)
    cv2.putText(display, tilt_txt, (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, tilt_col, 2, cv2.LINE_AA)

    # ── Status bar ──
    if not sticky_warn:
        # Searching for the object is not an alarm; say so plainly.
        if sticky_zone == "NOT_FOUND":
            s_text = "SEARCHING FOR PATIENT..."
            s_col  = (180, 180, 180)
        else:
            s_text = "PATIENT SAFE"
            s_col  = (50, 220, 50)
    else:
        s_text = "!! PATIENT AT RISK — ABOUT TO FALL !!"
        s_col  = (60, 60, 255)
    cv2.rectangle(display, (0, fh_d - 44), (fw_d, fh_d), (20,20,20), -1)
    cv2.putText(display, s_text, (12, fh_d - 14),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, s_col, 2, cv2.LINE_AA)
    cv2.putText(display,
                f"FPS {fps:.1f}  |  [t] reset  [r] reselect  [q] quit",
                (fw_d - 370, fh_d - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150,150,150), 1)

    # ── Publish the zone for app.py -> motor ESP32 ──
    status_counter += 1
    if status_counter >= STATUS_EVERY_N:
        status_counter = 0
        publish_status(sticky_zone, last_conf, last_cx_w, last_cy_w,
                       tracking_ok, frame_no)

    # Mirror the annotated view to the browser.
    live_counter += 1
    if live_counter >= LIVE_EVERY_N:
        live_counter = 0
        publish_frame(display)

    # display is already at window size — the overlay was drawn on the
    # enlarged canvas, so nothing more to scale here.
    cv2.imshow("Bed Monitor", display)

# ── Cleanup ──
publish_status("EMPTY", 0.0, None, None, False, frame_no)
log_fh.close()
cap.release()
cv2.destroyAllWindows()
print("  Session ended.  Log saved to:", LOG_FILE)
