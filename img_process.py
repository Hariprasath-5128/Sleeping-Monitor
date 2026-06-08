import cv2
import numpy as np
import time
import csv
import os
from collections import deque
from datetime import datetime

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
IP   = "192.168.1.2"
PORT = 8080
URL  = f"http://{IP}:{PORT}/video"

WARP_SIZE      = 640    # internal bird's-eye canvas — also optimal YOLO input size
TRAIL_LEN      = 80     # centroid history for the trail
LOG_EVERY_N    = 5      # CSV rows written every N frames
LOG_FILE       = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "position_log.csv")
FLASH_INTERVAL = 0.4    # warning blink period (s)
LOST_TIMEOUT   = 60     # frames without a detection before re-acquiring

# YOLOv8 — yolov8s gives the best speed/accuracy tradeoff on an RTX 2050
YOLO_MODEL     = "yolov8s.pt"
YOLO_CONF      = 0.20   # low threshold — whitener may score lower than 'person'
YOLO_IOU       = 0.45   # NMS IoU threshold

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

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, mouse_callback)

    while not selection_done:
        ret, live = cap_ref.read()
        disp = live.copy() if ret else first_frame.copy()
        fh, fw = disp.shape[:2]

        idx = min(len(clicked_pts), 3)
        cv2.putText(disp,
                    f"Click corner {len(clicked_pts)+1}/4  ({labels[idx]})",
                    (10, 38), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 220, 255), 2)
        cv2.putText(disp,
                    "Order: TL -> TR -> BR -> BL   |  [r] reset   [q] quit",
                    (10, fh - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        for i, pt in enumerate(clicked_pts):
            cv2.circle(disp, pt, 9, (0, 255, 0), -1)
            cv2.circle(disp, pt, 11, (255, 255, 255), 1)
            cv2.putText(disp, labels[i], (pt[0]+12, pt[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            if i > 0:
                cv2.line(disp, clicked_pts[i-1], pt, (0, 255, 0), 2)
        if len(clicked_pts) >= 3:
            cv2.line(disp, clicked_pts[-1], clicked_pts[0], (0, 200, 0), 1)

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
    return order_points(clicked_pts)


# ─────────────────────────────────────────────────────────
# ZONE OVERLAY
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

    # ── Zone labels — large, bold, with dark background box ──
    for xw_mid, label, col in [
        (lz // 2,        "DANGER", (80, 80, 255)),
        ((rz + W) // 2,  "DANGER", (80, 80, 255)),
        ((lz + rz) // 2, "SAFE",   (80, 255, 120)),
    ]:
        pt = warp_to_frame([[xw_mid, H // 2]], M_inv)[0]
        font  = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.75
        thick = 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        lx = pt[0] - tw // 2
        ly = pt[1] + th // 2
        # Dark backing rectangle
        cv2.rectangle(canvas, (lx - 6, ly - th - 6), (lx + tw + 6, ly + 6),
                      (15, 15, 15), -1)
        cv2.putText(canvas, label, (lx, ly), font, scale, col, thick, cv2.LINE_AA)

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
    elif zone == "SAFE":
        box_col  = (0, 255, 80)
        text_col = (0, 220, 80)
        status   = "SAFE"
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

    # (YOLO class label hidden — only boundary box is shown)

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
_tb = {"left": 150, "right": 490, "conf": 20}


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

# --- Load YOLOv8 (runs on GPU automatically if CUDA available) ---
print("=" * 58)
print("  Hospital Bed / Patient Monitor  — YOLOv8 tracking")
print("=" * 58)
print("  Loading YOLOv8s model …")

from ultralytics import YOLO  # import here so startup message prints first
model = YOLO(YOLO_MODEL)

# Force GPU if available
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Running on: {device.upper()}")
model.to(device)

# ── Camera ──
cap = cv2.VideoCapture(URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("  Waiting for video stream …")
while True:
    ret, first_frame = cap.read()
    if ret:
        break

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

fps          = 0.0
_fps_t0      = time.time()
_fps_n       = 0
flash_on     = False
flash_t      = 0.0
tracking_ok  = False

frame_no     = 0
log_counter  = 0

# ─────────────────────────────────────────────────────────
# MONITORING LOOP
# ─────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        continue

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

    # ── Warp to bird's-eye (internal — YOLO runs here) ──
    # Running YOLO on the warped frame means it ONLY detects objects
    # inside the defined boundary — the "no detection outside boundary" feature.
    warped = cv2.warpPerspective(frame, M, (WARP_SIZE, WARP_SIZE))

    # ── YOLOv8 tracking ──
    # persist=True keeps track IDs stable across frames
    # verbose=False suppresses per-frame console output
    results = model.track(warped,
                          persist=True,
                          conf=conf_thresh,
                          iou=YOLO_IOU,
                          verbose=False,
                          device=device)

    display = frame.copy()
    draw_zones(display, LZ, RZ, M_inv)

    # ── Parse YOLO results ──
    # Collect all detections with valid track IDs
    detections = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        if boxes.id is not None:
            ids   = boxes.id.cpu().numpy().astype(int)
            xywhs = boxes.xywh.cpu().numpy()          # cx, cy, w, h (warp-space pixels)
            confs = boxes.conf.cpu().numpy()
            clss  = boxes.cls.cpu().numpy().astype(int)
            for i in range(len(ids)):
                x1 = int(xywhs[i,0] - xywhs[i,2]/2)
                y1 = int(xywhs[i,1] - xywhs[i,3]/2)
                bw = int(xywhs[i,2])
                bh = int(xywhs[i,3])
                detections.append({
                    "id":    ids[i],
                    "bbox":  (x1, y1, bw, bh),
                    "cx":    int(xywhs[i,0]),
                    "cy":    int(xywhs[i,1]),
                    "conf":  float(confs[i]),
                    "class": model.names[clss[i]],
                })

    # ── ID Locking ──
    # If we have a locked target, prefer that track ID.
    # If locked ID is gone, pick the highest-confidence detection to re-lock.
    # This ensures the whitener box never jumps to another object.
    chosen = None
    if locked_id is not None:
        for d in detections:
            if d["id"] == locked_id:
                chosen = d
                break

    if chosen is None and detections:
        # Acquire / re-acquire: pick highest confidence detection
        chosen     = max(detections, key=lambda d: d["conf"])
        locked_id  = chosen["id"]
        lost_frames = 0
        print(f"  Locked onto track ID={locked_id} "
              f"({chosen['class']} @ {chosen['conf']:.0%})")

    # ── Update state from chosen detection ──
    current_zone = sticky_zone

    if chosen is not None:
        bx, by, bw, bh = chosen["bbox"]
        cx_w  = chosen["cx"]
        cy_w  = chosen["cy"]

        # Clamp to warp canvas
        bx = max(0, min(bx, WARP_SIZE - 1))
        by = max(0, min(by, WARP_SIZE - 1))
        bw = min(bw, WARP_SIZE - bx)
        bh = min(bh, WARP_SIZE - by)
        cx_w = max(0, min(cx_w, WARP_SIZE - 1))
        cy_w = max(0, min(cy_w, WARP_SIZE - 1))

        # Light EMA smoothing
        if last_cx_w is not None:
            cx_w = int(0.8 * cx_w + 0.2 * last_cx_w)
            cy_w = int(0.8 * cy_w + 0.2 * last_cy_w)

        last_cx_w, last_cy_w = cx_w, cy_w
        last_bbox_w  = (bx, by, bw, bh)
        last_label   = chosen["class"]
        last_conf    = chosen["conf"]
        tracking_ok  = True
        lost_frames  = 0

        trail.append((cx_w, cy_w))

        # ── Zone classification: overlap percentage threshold ──
        # Only trigger DANGER when ≥25% of the box width is inside a danger zone.
        # This prevents a minor/accidental edge overlap from firing a false alarm.
        DANGER_OVERLAP_THRESH = 0.35   # 35 % of box width must be inside danger zone

        left_overlap  = max(0, LZ - bx)          # pixels of box inside LEFT danger zone
        right_overlap = max(0, (bx + bw) - RZ)   # pixels of box inside RIGHT danger zone
        box_width     = max(bw, 1)

        if left_overlap / box_width >= DANGER_OVERLAP_THRESH:
            current_zone = "LEFT"
        elif right_overlap / box_width >= DANGER_OVERLAP_THRESH:
            current_zone = "RIGHT"
        else:
            current_zone = "SAFE"

        draw_person(display, bx, by, bw, bh, cx_w, cy_w,
                    current_zone, trail, M_inv,
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

        # Sticky warning
        if current_zone in ("LEFT", "RIGHT"):
            sticky_warn = True
            sticky_msg  = ("<-- ABOUT TO FALL LEFT"
                           if current_zone == "LEFT"
                           else "ABOUT TO FALL RIGHT -->")
            sticky_zone = current_zone
        else:
            sticky_warn = False
            sticky_msg  = ""
            sticky_zone = "SAFE"

    else:
        # No matching detection — show last known position
        lost_frames += 1
        tracking_ok  = False

        if lost_frames >= LOST_TIMEOUT:
            locked_id = None          # fully re-acquire next time
            lost_frames = 0
            print("  Target lost — re-acquiring …")

        if last_bbox_w is not None:
            bx, by, bw, bh = last_bbox_w
            draw_person(display, bx, by, bw, bh,
                        last_cx_w, last_cy_w, sticky_zone,
                        trail, M_inv, is_searching=True,
                        label=last_label.upper())

    # ── Flash timer ──
    now = time.time()
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
    badge_text = ("STABLE" if sticky_zone == "SAFE"
                  else f"DANGER  ({sticky_zone} EDGE)")
    badge_col  = (0, 210, 80) if sticky_zone == "SAFE" else (60, 60, 255)
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
    s_text = ("PATIENT SAFE" if not sticky_warn
              else "!! PATIENT AT RISK — ABOUT TO FALL !!")
    s_col  = (50, 220, 50) if not sticky_warn else (60, 60, 255)
    cv2.rectangle(display, (0, fh_d - 44), (fw_d, fh_d), (20,20,20), -1)
    cv2.putText(display, s_text, (12, fh_d - 14),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, s_col, 2, cv2.LINE_AA)
    cv2.putText(display,
                f"FPS {fps:.1f}  |  [t] reset  [r] reselect  [q] quit",
                (fw_d - 370, fh_d - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150,150,150), 1)

    cv2.imshow("Bed Monitor", display)

# ── Cleanup ──
log_fh.close()
cap.release()
cv2.destroyAllWindows()
print("  Session ended.  Log saved to:", LOG_FILE)
