"""
yolo_tracker.py
--------------
Detects the whitener (correction pen) using YOLOv8 and tracks its
position over time to predict whether it is drifting toward the edge
of the monitored surface (box / bed).

Fall Risk Levels
----------------
  NO ONE DETECTED -- No tracked object is visible in the monitored surface
  STABLE         -- Object is stationary in the safe zone
  DRIFT WARNING  -- Object is slowly moving toward an edge
  FALL IMMINENT  -- Fast movement or predicted position is in danger zone
"""

import csv
import os
import time
from collections import deque

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# YOLO class IDs that may represent the whitener in COCO-80 dataset
# bottle=39, scissors=76, toothbrush=79, remote=65, cell phone=67
# We search all of them and pick the biggest inside the box.
CANDIDATE_CLASSES = None # Track any object inside the manual borders

# ── Detection sensitivity ─────────────────────────────────────────────────
# Lower conf threshold catches partially-occluded / edge-lit whiteners
DETECT_CONF       = 0.05   # reduced to heavily catch the pen

# Extra buffer on the LEFT boundary: we treat LEFT_Z - LEFT_EDGE_BUFFER
# as the effective danger line so the tracker fires BEFORE the object
# fully exits the safe zone (the most common miss scenario at left edge).
LEFT_EDGE_BUFFER  = 35    # px in warp-space

MODEL_PATH  = "yolov8n.pt"   # auto-downloaded on first use (~6 MB)
HISTORY_LEN = 45             # frames kept for trajectory analysis (≈1.5 s)
PREDICT_AHEAD = 30           # frames ahead for fall prediction

# Timestamp-based movement thresholds in warp pixels per second.
MIN_HISTORY_FOR_RISK = 3
RECENT_WINDOW_SEC = 1.5
PREDICT_AHEAD_SEC = 1.2
SLOW_DRIFT_PX_PER_SEC = 8.0
FAST_DRIFT_PX_PER_SEC = 35.0
MIN_DRIFT_DISPLACEMENT_PX = 4.0
NEAR_EDGE_BUFFER_PX = 90.0

RISK_STABLE   = "STABLE"
RISK_DRIFT    = "DRIFT WARNING"
RISK_IMMINENT = "FALL IMMINENT"
RISK_NO_DETECTION = "NO ONE DETECTED"

RISK_COLORS = {
    RISK_STABLE:   (60, 220, 60),
    RISK_DRIFT:    (0, 200, 255),
    RISK_IMMINENT: (0, 0, 255),
    RISK_NO_DETECTION: (180, 180, 180),
}
# ──────────────────────────────────────────────────────────────────────


class WhitenerTracker:
    """YOLOv8-powered whitener tracker with trajectory-based fall prediction."""

    def __init__(self, csv_path: str = "position_log.csv"):
        from ultralytics import YOLO  # import here so startup is fast
        self.model = YOLO(MODEL_PATH)
        self.model.fuse()            # speed optimisation

        self.history: deque = deque(maxlen=HISTORY_LEN)  # (time, cx_warp, cy_warp, ar)
        self.last_bbox_frame = None  # (x1, y1, x2, y2) in original frame coords
        self.risk_level = RISK_NO_DETECTION
        self.predicted_warp_pt = None
        self._last_no_detection_log_t = 0.0

        # CSV setup
        self.csv_path = csv_path
        self._init_csv()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect(self, frame, box_mask=None):
        """
        Run YOLO on the frame and return (x1,y1,x2,y2) of the best
        whitener candidate, or None.

        Parameters
        ----------
        frame    : original BGR frame
        box_mask : optional binary mask limiting detection to the box region
        """
        results = self.model(frame, verbose=False, conf=DETECT_CONF)[0]
        best_box  = None
        best_conf = 0.0

        for det in results.boxes:
            cls_id = int(det.cls[0])
            conf   = float(det.conf[0])
            x1, y1, x2, y2 = map(int, det.xyxy[0])

            # Accept any class if we have no candidate restriction,
            # otherwise prefer known candidates
            if CANDIDATE_CLASSES and cls_id not in CANDIDATE_CLASSES:
                continue

            # Optionally restrict to region inside the box
            if box_mask is not None:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                if 0 <= cy < box_mask.shape[0] and 0 <= cx < box_mask.shape[1]:
                    if box_mask[cy, cx] == 0:
                        continue

            if conf > best_conf:
                best_conf = conf
                best_box  = (x1, y1, x2, y2)

        self.last_bbox_frame = best_box
        return best_box

    # ------------------------------------------------------------------
    # Update history with warp-space centroid
    # ------------------------------------------------------------------
    def update_warp(self, bbox_frame, M, warp_size, left_z, right_z):
        """
        Map detected bounding box from frame-space to warp-space, record
        history, and compute the current risk level.

        Parameters
        ----------
        bbox_frame : (x1,y1,x2,y2) in frame coords, or None
        M          : perspective transform matrix (frame → warp)
        warp_size  : WARP_SIZE constant
        left_z     : LEFT danger boundary in warp-x
        right_z    : RIGHT danger boundary in warp-x
        """
        if bbox_frame is None:
            return

        x1, y1, x2, y2 = bbox_frame
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        aspect_ratio = (y2 - y1) / max(x2 - x1, 1)

        # Project centroid into warp space
        pt     = np.array([[[cx, cy]]], dtype="float32")
        warp_pt = cv2.perspectiveTransform(pt, M)[0][0]
        wx, wy = warp_pt

        record = {
            "t":  time.time(),
            "wx": float(wx),
            "wy": float(wy),
            "ar": float(aspect_ratio),
            "lz": left_z,
            "rz": right_z,
        }
        self.history.append(record)
        self._compute_risk(warp_size, left_z, right_z)
        self._log_csv(record)

    def mark_no_detection(self, left_z, right_z, log_interval=1.0):
        """Set and periodically log the explicit no-object state."""
        self.last_bbox_frame = None
        self.predicted_warp_pt = None
        self.risk_level = RISK_NO_DETECTION
        self.history.clear()

        now = time.time()
        if now - self._last_no_detection_log_t < log_interval:
            return

        self._last_no_detection_log_t = now
        self._log_csv({
            "wx": -1.0,
            "wy": -1.0,
            "ar": 0.0,
            "lz": left_z,
            "rz": right_z,
        })

    # ------------------------------------------------------------------
    # Fall / drift risk computation
    # ------------------------------------------------------------------
    def _compute_risk_legacy_unused(self, warp_size, left_z, right_z):
        if len(self.history) < MIN_HISTORY_FOR_RISK:
            self.risk_level = RISK_STABLE
            self.predicted_warp_pt = None
            return

        records = list(self.history)
        latest = records[-1]
        cur_x = latest["wx"]
        cur_y = latest["wy"]
        cur_t = latest["t"]

        window_start = latest
        for rec in reversed(records[:-1]):
            window_start = rec
            if cur_t - rec["t"] >= RECENT_WINDOW_SEC:
                break

        dt_window = max(cur_t - window_start["t"], 1e-3)
        dx_window = cur_x - window_start["wx"]
        dy_window = cur_y - window_start["wy"]
        vx = dx_window / dt_window
        vy = dy_window / dt_window
        speed = float(np.hypot(vx, vy))
        displacement = float(np.hypot(dx_window, dy_window))

        prev = records[-2]
        dt_inst = max(cur_t - prev["t"], 1e-3)
        inst_dx = cur_x - prev["wx"]
        inst_dy = cur_y - prev["wy"]
        inst_speed = float(np.hypot(inst_dx / dt_inst, inst_dy / dt_inst))
        speed = max(speed, inst_speed * 0.6)

        # --- Velocity (avg over last 10 frames) ---
        recent = []
        if len(recent) >= 2:
            vel = (recent[-1] - recent[0]) / (len(recent) - 1)   # px/frame
        else:
            vel = np.zeros(2)

        vel_x = vel[0]   # positive → moving right

        # Predicted position in PREDICT_AHEAD frames
        pred = np.array([cur_x + vx * PREDICT_AHEAD_SEC,
                         cur_y + vy * PREDICT_AHEAD_SEC])
        self.predicted_warp_pt = tuple(pred.astype(int))

        # Aspect ratio change rate
        ars = [r["ar"] for r in records]
        time_span = max(records[-1]["t"] - records[0]["t"], 1e-3)
        ar_rate = abs(ars[-1] - ars[0]) / time_span

        # --- Risk classification ---
        # 1. Already in danger zone (note: uses eff_left_z on the left)
        in_danger = cur_x < eff_left_z or cur_x > right_z
        # 2. Prediction crosses into danger
        pred_danger = pred[0] < eff_left_z or pred[0] > right_z
        # 3. Fast drift (> 3 px/frame toward edge)
        drift_toward_left  = vel_x < -3 and cur_x < eff_left_z + 60
        drift_toward_right = vel_x >  3 and cur_x > right_z - 60
        fast_drift = drift_toward_left or drift_toward_right
        # 4. Tilting rapidly
        tilting = ar_rate > 0.08

        if in_danger or (pred_danger and abs(vel_x) > 2) or tilting:
            self.risk_level = RISK_IMMINENT
        elif pred_danger or fast_drift or (abs(vel_x) > 1.5):
            self.risk_level = RISK_DRIFT
        else:
            self.risk_level = RISK_STABLE

    def _compute_risk(self, warp_size, left_z, right_z):
        """Classify risk from recent timestamped positions in warp-space."""
        if len(self.history) < MIN_HISTORY_FOR_RISK:
            self.risk_level = RISK_STABLE
            self.predicted_warp_pt = None
            return

        records = list(self.history)
        latest = records[-1]
        cur_x = latest["wx"]
        cur_y = latest["wy"]
        cur_t = latest["t"]

        window_start = latest
        for rec in reversed(records[:-1]):
            window_start = rec
            if cur_t - rec["t"] >= RECENT_WINDOW_SEC:
                break

        dt_window = max(cur_t - window_start["t"], 1e-3)
        dx_window = cur_x - window_start["wx"]
        dy_window = cur_y - window_start["wy"]
        vx = dx_window / dt_window
        vy = dy_window / dt_window
        avg_speed = float(np.hypot(vx, vy))
        displacement = float(np.hypot(dx_window, dy_window))

        prev = records[-2]
        dt_inst = max(cur_t - prev["t"], 1e-3)
        inst_vx = (cur_x - prev["wx"]) / dt_inst
        inst_vy = (cur_y - prev["wy"]) / dt_inst
        inst_speed = float(np.hypot(inst_vx, inst_vy))
        speed = max(avg_speed, inst_speed * 0.6)

        pred = np.array([cur_x + vx * PREDICT_AHEAD_SEC,
                         cur_y + vy * PREDICT_AHEAD_SEC])
        self.predicted_warp_pt = tuple(pred.astype(int))

        ars = [r["ar"] for r in records]
        time_span = max(records[-1]["t"] - records[0]["t"], 1e-3)
        ar_rate = abs(ars[-1] - ars[0]) / time_span

        center_x = (left_z + right_z) / 2.0
        nearest_edge_dist = (cur_x - left_z) if cur_x < center_x else (right_z - cur_x)
        in_danger = cur_x < left_z or cur_x > right_z
        pred_danger = pred[0] < left_z or pred[0] > right_z
        near_edge = nearest_edge_dist <= NEAR_EDGE_BUFFER_PX

        moving_left = vx < -SLOW_DRIFT_PX_PER_SEC
        moving_right = vx > SLOW_DRIFT_PX_PER_SEC
        moving_toward_left_edge = cur_x < center_x and moving_left
        moving_toward_right_edge = cur_x >= center_x and moving_right
        moving_toward_edge = moving_toward_left_edge or moving_toward_right_edge

        slow_drift = (
            speed >= SLOW_DRIFT_PX_PER_SEC and
            displacement >= MIN_DRIFT_DISPLACEMENT_PX
        )
        fast_drift = speed >= FAST_DRIFT_PX_PER_SEC
        fast_edge_drift = fast_drift and moving_toward_edge and near_edge
        tilting = ar_rate > 0.20

        if in_danger or fast_edge_drift or (pred_danger and moving_toward_edge) or tilting:
            self.risk_level = RISK_IMMINENT
        elif pred_danger or fast_drift or slow_drift:
            self.risk_level = RISK_DRIFT
        else:
            self.risk_level = RISK_STABLE

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def draw_on_frame(self, canvas, M_inv, warp_size):
        """
        Draw detection, trail, predicted position, and risk indicator
        on the original frame canvas (in-place).
        """
        color = RISK_COLORS.get(self.risk_level, (255, 255, 255))

        # Detection box
        if self.last_bbox_frame is not None:
            x1, y1, x2, y2 = self.last_bbox_frame
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, "WHITENER", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Trail (project warp history back to frame)
        if len(self.history) >= 2:
            trail_pts = []
            for r in self.history:
                wp = np.array([[[r["wx"], r["wy"]]]], dtype="float32")
                fp = cv2.perspectiveTransform(wp, M_inv)[0][0].astype(int)
                trail_pts.append(tuple(fp))

            for i in range(1, len(trail_pts)):
                alpha = i / len(trail_pts)
                t_col = tuple(int(c * alpha) for c in color)
                cv2.line(canvas, trail_pts[i-1], trail_pts[i], t_col, 2)
            cv2.circle(canvas, trail_pts[-1], 5, color, -1)

        # Predicted position marker
        if self.predicted_warp_pt is not None and len(self.history) > 5:
            pred_wp = np.array([[[float(self.predicted_warp_pt[0]),
                                   float(self.predicted_warp_pt[1])]]], dtype="float32")
            pred_fp = cv2.perspectiveTransform(pred_wp, M_inv)[0][0].astype(int)
            cv2.drawMarker(canvas, tuple(pred_fp), (255, 165, 0),
                           cv2.MARKER_DIAMOND, 20, 2)
            cv2.putText(canvas, "PRED", (pred_fp[0]+8, pred_fp[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1)

        # Risk label (top-right)
        fh, fw = canvas.shape[:2]
        label  = f"Object: {self.risk_level}"
        tsz    = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0]
        tx     = fw - tsz[0] - 12
        ty     = 35
        cv2.rectangle(canvas, (tx - 6, ty - tsz[1] - 6), (fw - 6, ty + 6),
                      (20, 20, 20), -1)
        cv2.putText(canvas, label, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 2)

        # Risk bar (right edge, vertical)
        bar_h    = 150
        bar_x    = fw - 18
        bar_top  = 55
        bar_bot  = bar_top + bar_h
        risk_map = {RISK_NO_DETECTION: 0, RISK_STABLE: 0, RISK_DRIFT: 0.55, RISK_IMMINENT: 1.0}
        fill_frac = risk_map.get(self.risk_level, 0)
        fill_y   = int(bar_bot - fill_frac * bar_h)
        cv2.rectangle(canvas, (bar_x, bar_top), (bar_x + 12, bar_bot), (50, 50, 50), -1)
        cv2.rectangle(canvas, (bar_x, fill_y),  (bar_x + 12, bar_bot), color, -1)
        cv2.rectangle(canvas, (bar_x, bar_top), (bar_x + 12, bar_bot), (180, 180, 180), 1)

    # ------------------------------------------------------------------
    # CSV logging
    # ------------------------------------------------------------------
    def _init_csv(self):
        write_header = not os.path.exists(self.csv_path)
        self._csv_file = open(self.csv_path, "a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if write_header:
            self._csv_writer.writerow(["timestamp", "warp_x", "warp_y",
                                       "aspect_ratio", "left_z", "right_z",
                                       "risk"])

    def _log_csv(self, record):
        self._csv_writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            f"{record['wx']:.1f}",
            f"{record['wy']:.1f}",
            f"{record['ar']:.3f}",
            record["lz"],
            record["rz"],
            self.risk_level,
        ])
        self._csv_file.flush()

    def close(self):
        self._csv_file.close()
