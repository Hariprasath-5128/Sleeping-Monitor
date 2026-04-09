import cv2
import numpy as np
import time

# -----------------------------
# CONFIG
# -----------------------------
IP = "10.164.151.177"
PORT = 8080
URL = f"http://{IP}:{PORT}/video"

WARP_SIZE = 400   # internal processing resolution (not displayed)

# -----------------------------
# GLOBALS
# -----------------------------
clicked_pts = []
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


def run_corner_selection(frame, window_name="Select Box Corners"):
    """Show frame, let user click 4 corner points, return ordered corners."""
    global clicked_pts, selection_done
    clicked_pts = []
    selection_done = False

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    labels = ["TL", "TR", "BR", "BL"]

    while not selection_done:
        disp = frame.copy()
        h, w = disp.shape[:2]

        # Instructions
        remaining = 4 - len(clicked_pts)
        cv2.putText(disp,
                    f"Click corner {len(clicked_pts)+1}/4  "
                    f"({labels[len(clicked_pts)]})",
                    (10, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 220, 255), 2)
        cv2.putText(disp, "Order: TL -> TR -> BR -> BL   |  [r] reset  [q] quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Draw placed points and connecting lines
        for i, pt in enumerate(clicked_pts):
            cv2.circle(disp, pt, 9, (0, 255, 0), -1)
            cv2.circle(disp, pt, 9, (255, 255, 255), 1)
            cv2.putText(disp, labels[i], (pt[0]+12, pt[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            if i > 0:
                cv2.line(disp, clicked_pts[i-1], pt, (0, 255, 0), 2)
        if len(clicked_pts) >= 3:
            cv2.line(disp, clicked_pts[-1], clicked_pts[0], (0, 255, 0), 1)

        cv2.imshow(window_name, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            clicked_pts = []
            selection_done = False
        elif key == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            exit()

    cv2.destroyWindow(window_name)
    return order_points(clicked_pts)


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
cap = cv2.VideoCapture(URL)

print("=" * 52)
print("  Hospital Bed / Box Monitor")
print("  Waiting for video stream...")
print("=" * 52)

# Grab first valid frame
while True:
    ret, first_frame = cap.read()
    if ret:
        break

# ---- Phase 1: Manual corner selection ----
print("  Click 4 corners of the box/bed surface.")
corners = run_corner_selection(first_frame)
M, M_inv = build_transforms(corners)
print("  Box defined! Monitoring started.")
print("  Press [r] to reselect corners | [q] to quit")

# ---- Controls window ----
cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Controls", 420, 130)
cv2.createTrackbar("LEFT  margin", "Controls", 100, WARP_SIZE, nothing)
cv2.createTrackbar("RIGHT margin", "Controls", 300, WARP_SIZE, nothing)
cv2.createTrackbar("MIN AREA",     "Controls", 600, 5000,      nothing)

prev_gray_warp = None
fps = 0.0
_fps_t0 = time.time()
_fps_count = 0
_warn_flash = False
_warn_flash_t = 0.0
FLASH_INTERVAL = 0.4

# ---- Phase 2: Monitoring loop ----
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # FPS
    _fps_count += 1
    if time.time() - _fps_t0 >= 1.0:
        fps = _fps_count / (time.time() - _fps_t0)
        _fps_count = 0
        _fps_t0 = time.time()

    key = cv2.waitKey(1) & 0xFF

    # ---- Reselect corners ----
    if key == ord('r'):
        corners = run_corner_selection(frame)
        M, M_inv = build_transforms(corners)
        prev_gray_warp = None
        print("  Corners reselected.")
        continue

    if key == ord('q'):
        break

    # Read trackbars
    LEFT_Z  = cv2.getTrackbarPos("LEFT  margin", "Controls")
    RIGHT_Z = cv2.getTrackbarPos("RIGHT margin", "Controls")
    MIN_AREA = cv2.getTrackbarPos("MIN AREA",    "Controls")
    if LEFT_Z >= RIGHT_Z - 20:
        LEFT_Z = max(0, RIGHT_Z - 20)

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

    any_warn = False
    warn_msg = ""

    for cnt in cnts:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        is_warn = cx < LEFT_Z or cx > RIGHT_Z
        if is_warn:
            any_warn = True
            warn_msg = "<-- LEFT EDGE WARNING" if cx < LEFT_Z else "RIGHT EDGE WARNING -->"
        draw_blob_on_frame(display, x, y, w, h, is_warn, M_inv)

    prev_gray_warp = gray_w

    # ---- Warning flash ----
    now = time.time()
    if any_warn:
        if now - _warn_flash_t >= FLASH_INTERVAL:
            _warn_flash = not _warn_flash
            _warn_flash_t = now
    else:
        _warn_flash = False

    fh, fw = display.shape[:2]

    # Blinking banner
    if any_warn and _warn_flash:
        ov = display.copy()
        cv2.rectangle(ov, (0, 0), (fw, 65), (0, 0, 200), -1)
        cv2.addWeighted(ov, 0.75, display, 0.25, 0, display)
        tsz = cv2.getTextSize(warn_msg, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)[0]
        cv2.putText(display, warn_msg, ((fw - tsz[0]) // 2, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

    # Status strip
    s_col  = (60, 220, 60) if not any_warn else (60, 60, 255)
    s_text = "PATIENT SAFE" if not any_warn else "PATIENT AT RISK"
    cv2.rectangle(display, (0, fh - 42), (fw, fh), (20, 20, 20), -1)
    cv2.putText(display, s_text,
                (12, fh - 14), cv2.FONT_HERSHEY_DUPLEX, 0.8, s_col, 2)
    cv2.putText(display, f"FPS {fps:.1f}  |  [r] reselect corners  [q] quit",
                (fw - 310, fh - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    cv2.imshow("Bed Monitor", display)

cap.release()
cv2.destroyAllWindows()