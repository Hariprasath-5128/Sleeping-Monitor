import cv2
import numpy as np
import time
import csv
import json
import os
from collections import deque
from datetime import datetime

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
# ── ESP32-CAM live stream (esp/live_monitor/live_monitor.ino) ──
# Set CAM_IP to the address the ESP32-CAM prints on its serial monitor.
# Override without editing this file:  set CAM_IP=192.168.1.50
CAM_IP     = os.environ.get("CAM_IP", "192.168.137.99")
CAM_PORT   = int(os.environ.get("CAM_PORT", "81"))
STREAM_URL = os.environ.get("STREAM_URL", f"http://{CAM_IP}:{CAM_PORT}/stream")

# Single-JPEG endpoint on the control server (port 80). Used automatically when
# the MJPEG stream on :81 will not open — the ESP32-CAM's stream server is the
# fragile half, while /capture keeps working.
CAPTURE_URL = os.environ.get("CAPTURE_URL", f"http://{CAM_IP}/capture")

# Seconds to wait for the MJPEG stream before giving up on it.
STREAM_OPEN_TIMEOUT = float(os.environ.get("STREAM_OPEN_TIMEOUT", "12"))

# Fall back to a locally attached webcam when the ESP32-CAM is unreachable.
# Set to None to disable the fallback and fail loudly instead.
FALLBACK_SOURCE = 0

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
    "DRIFT WARNING": "WARNING",
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
    return max(1.0, min(fit, 4.0))


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
    elif zone == "WARNING":
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
class CapturePoller:
    """Reads single JPEGs from /capture, quacking like cv2.VideoCapture.

    The ESP32-CAM's MJPEG server on :81 is the part that tends to fall over;
    /capture on :80 stays up. Polling it is slower but keeps the monitor
    running on exactly the same code path.
    """

    def __init__(self, url):
        self.url = url
        self._session = None
        try:
            import requests
            self._session = requests.Session()
        except ImportError:
            pass

    def read(self):
        try:
            if self._session is not None:
                r = self._session.get(self.url, timeout=5)
                if r.status_code != 200:
                    return False, None
                buf = np.frombuffer(r.content, dtype=np.uint8)
            else:
                from urllib.request import urlopen
                with urlopen(self.url, timeout=5) as resp:
                    buf = np.frombuffer(resp.read(), dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return frame is not None, frame
        except Exception:
            return False, None

    def set(self, *_a, **_k):
        return False

    def release(self):
        if self._session is not None:
            self._session.close()


def _mjpeg_responds(host, port, path="/stream", timeout=3.0):
    """True if the stream port actually returns HTTP headers.

    Worth doing before cv2.VideoCapture: FFmpeg enforces its own ~30 s timeout
    and ignores ours, so probing first turns a 30 s stall into a 3 s check.
    """
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout) as sk:
            sk.settimeout(timeout)
            sk.sendall(f"GET {path} HTTP/1.1\r\nHost: {host}\r\n\r\n".encode())
            head = sk.recv(64)
            return head.startswith(b"HTTP/")
    except OSError:
        return False


def open_stream():
    """Open the ESP32-CAM: MJPEG first, then /capture, then a local webcam."""
    if _mjpeg_responds(CAM_IP, CAM_PORT):
        print(f"  Opening ESP32-CAM stream: {STREAM_URL}")
        c = cv2.VideoCapture(STREAM_URL)
        try:
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        t0 = time.time()
        while time.time() - t0 < STREAM_OPEN_TIMEOUT:
            ok, frm = c.read()
            if ok and frm is not None:
                print("  ESP32-CAM stream is live.")
                return c, frm, True
            time.sleep(0.2)

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
            print("  ESP32-CAM /capture is live — polling single frames.")
            print("  (Lower frame rate than MJPEG, but the pipeline is identical.)")
            return poller, frm, True
        time.sleep(0.3)
    poller.release()

    if FALLBACK_SOURCE is None:
        raise SystemExit(f"  Could not reach the ESP32-CAM at {CAM_IP}.")

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


def publish_frame(img):
    """Send the annotated frame to app.py for its /video endpoint.

    Posting into the server's memory rather than swapping a file on disk:
    Windows timestamps are too coarse to distinguish frames written a few
    milliseconds apart, so a file-based handoff silently drops most of them.
    """
    global _frame_session, _frame_post_ok

    try:
        ok, buf = cv2.imencode(".jpg", img,
                               [int(cv2.IMWRITE_JPEG_QUALITY), LIVE_QUALITY])
        if not ok:
            return
        jpg = buf.tobytes()
    except cv2.error:
        return

    if _frame_post_ok:
        try:
            if _frame_session is None:
                import requests
                _frame_session = requests.Session()
            _frame_session.post(LIVE_POST_URL, data=jpg, timeout=1.0,
                                headers={"Content-Type": "image/jpeg"})
            return
        except ImportError:
            _frame_post_ok = False
        except Exception:
            # Server not up (yet). Fall through to the file so the view still
            # works, and keep trying the socket on later frames.
            pass

    # Fallback: leave the frame on disk for app.py to pick up.
    try:
        tmp = LIVE_FRAME + ".tmp"
        with open(tmp, "wb") as f:
            f.write(jpg)
        for _ in range(3):
            try:
                os.replace(tmp, LIVE_FRAME)
                return
            except PermissionError:
                time.sleep(0.005)
    except OSError:
        pass


def zone_from_geometry(cx_w, lz, rz, tracking):
    """Derive the zone from the centroid when no classifier is loaded."""
    if not tracking or cx_w is None:
        return "EMPTY"

    band = max(20, int((rz - lz) * 0.15))   # WARNING band inside each boundary
    if cx_w < lz:
        return "DANGER_LEFT"
    if cx_w > rz:
        return "DANGER_RIGHT"
    if cx_w < lz + band or cx_w > rz - band:
        return "WARNING"
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
read_fail    = 0

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
        if read_fail == 1 or read_fail % 30 == 0:
            print(f"  [STREAM] read failed ({read_fail}) — reconnecting…")
            publish_status("NOT_FOUND", 0.0, last_cx_w, last_cy_w, False, frame_no)
        if read_fail >= 30:
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
            except SystemExit:
                print("  [STREAM] camera still unreachable — retrying…")
        else:
            time.sleep(0.05)
        continue

    read_fail = 0

    # Drain buffer for freshest frame (minimize stream latency)
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

            for i in range(len(xywhs)):
                x1 = int(xywhs[i,0] - xywhs[i,2]/2)
                y1 = int(xywhs[i,1] - xywhs[i,3]/2)
                bw = int(xywhs[i,2])
                bh = int(xywhs[i,3])
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
        chosen     = max(detections, key=lambda d: d["conf"])
        locked_id  = chosen["id"]
        lost_frames = 0
        print(f"  Locked onto track ID={locked_id} ({chosen['class']} @ {chosen['conf']:.0%})")

    # ── 2. Classification Brain (or geometric fallback) ──
    geo_zone = zone_from_geometry(
        chosen["cx"] if chosen is not None else last_cx_w,
        LZ, RZ,
        chosen is not None,
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
        elif raw_zone == "SAFE" and geo_zone == "WARNING":
            raw_zone = "WARNING"
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
    if current_zone in ("DANGER_LEFT", "DANGER_RIGHT", "WARNING", "NOT_FOUND"):
        sticky_warn = True
        if current_zone == "DANGER_LEFT":
            sticky_msg = "<-- ABOUT TO FALL LEFT"
        elif current_zone == "DANGER_RIGHT":
            sticky_msg = "ABOUT TO FALL RIGHT -->"
        elif current_zone == "NOT_FOUND":
            sticky_msg = "!! OBJECT NOT FOUND — MAY HAVE FALLEN !!"
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
    elif sticky_zone == "WARNING":
        badge_text = "WARNING"
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

    # ── Status bar ──
    if not sticky_warn:
        s_text = "PATIENT SAFE"
        s_col  = (50, 220, 50)
    elif sticky_zone == "NOT_FOUND":
        s_text = "OBJECT NOT FOUND — MAY HAVE FALLEN OFF BED"
        s_col  = (0, 100, 255)
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
