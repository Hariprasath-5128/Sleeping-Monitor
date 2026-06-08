"""
monitor_engine.py
-----------------
Adaptive health monitoring and prediction engine.

Reads position_log.csv in real-time, runs LNN-style online learning,
and generates predictions + a web-ready JSON payload.
Also saves a full report every 10 minutes.
"""

import csv
import json
import math
import os
import time
from collections import deque
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
WINDOW_SIZE  = 40          # rolling window
REPORT_DIR   = os.path.join(os.path.dirname(__file__), "reports")
REPORT_EVERY = 600         # seconds (10 minutes)

# LNN weights (updated online via exponential decay)
_LNN_ALPHA   = 0.18        # learning rate
_DRIFT_THRESH = 0.25       # drift detection threshold on variance ratio


# ─────────────────────────────────────────────
class AdaptiveMonitorEngine:
    """
    Online learning engine that produces health predictions
    from position_log.csv entries in real-time.
    """

    def __init__(self, csv_path: str):
        self.csv_path     = csv_path
        self.window: deque = deque(maxlen=WINDOW_SIZE)

        # LNN internal weights (scalar parameters that adapt online)
        self._w_vel   = 0.5   # weight on velocity signal
        self._w_acc   = 0.3   # weight on acceleration signal
        self._w_pos   = 0.2   # weight on absolute position danger

        # Baseline variance (used for drift detection)
        self._base_var    = None
        self._drift_count = 0

        # State
        self.last_prediction: dict = {}
        self._last_row_count  = 0
        self._last_report_t   = time.time()
        self._last_report_path = ""

        os.makedirs(REPORT_DIR, exist_ok=True)

    # ─────────────────────────────────────────
    # Main update call — tail the CSV for new rows
    # ─────────────────────────────────────────
    def update(self) -> dict | None:
        """
        Read any new rows from the CSV, update the window,
        run predictions, and return the latest JSON payload.
        Returns None if no new data.
        """
        rows = self._read_new_rows()
        if not rows:
            return None

        for row in rows:
            self.window.append(row)

        if len(self.window) < 3:
            return None

        payload = self._run_prediction()
        self.last_prediction = payload

        # Save report every 10 minutes
        if time.time() - self._last_report_t >= REPORT_EVERY:
            self._save_report(payload)
            self._last_report_t = time.time()

        return payload

    # ─────────────────────────────────────────
    # CSV reader (tail-mode)
    # ─────────────────────────────────────────
    def _read_new_rows(self) -> list:
        try:
            with open(self.csv_path, newline="") as f:
                reader = list(csv.DictReader(f))
        except (FileNotFoundError, StopIteration):
            return []

        new_rows = reader[self._last_row_count:]
        self._last_row_count = len(reader)

        parsed = []
        for r in new_rows:
            try:
                parsed.append({
                    "t":   r.get("timestamp", ""),
                    "wx":  float(r.get("warp_x", 0)),
                    "wy":  float(r.get("warp_y", 0)),
                    "ar":  float(r.get("aspect_ratio", 1)),
                    "lz":  float(r.get("left_z", 60)),
                    "rz":  float(r.get("right_z", 340)),
                    "risk": r.get("risk", "STABLE"),
                })
            except ValueError:
                continue
        return parsed

    # ─────────────────────────────────────────
    # Core prediction engine
    # ─────────────────────────────────────────
    def _run_prediction(self) -> dict:
        pts   = list(self.window)
        wx    = [p["wx"] for p in pts]
        wy    = [p["wy"] for p in pts]
        ar    = [p["ar"] for p in pts]
        lz    = pts[-1]["lz"]
        rz    = pts[-1]["rz"]
        safe_w = rz - lz          # width of safe zone in warp pixels

        # ── Derived features ────────────────────────────────────────────
        vel_x = wx[-1] - wx[-2]                        # velocity (px/step)
        vel_y = wy[-1] - wy[-2]
        acc_x = (wx[-1] - wx[-2]) - (wx[-2] - wx[-3]) if len(wx) >= 3 else 0
        ar_rate = abs(ar[-1] - ar[0]) / max(len(ar), 1)   # tilt rate

        # Anomaly: normalised z-score of last wx vs window mean
        mean_x = sum(wx) / len(wx)
        std_x  = math.sqrt(sum((v - mean_x) ** 2 for v in wx) / len(wx)) + 1e-6
        anomaly_score = min(abs(wx[-1] - mean_x) / std_x / 3.0, 1.0)

        # Variance of recent positions (for drift)
        var_x = std_x ** 2
        if self._base_var is None:
            self._base_var = var_x

        drift_intensity = abs(var_x - self._base_var) / (self._base_var + 1e-6)
        drift_detected  = drift_intensity > _DRIFT_THRESH

        # ── Online LNN weight update ─────────────────────────────────────
        # Give heavier weight to whichever signal last signalled correctly
        last_risk = pts[-1]["risk"]
        if last_risk == "FALL IMMINENT":
            self._w_vel = min(1.0, self._w_vel + _LNN_ALPHA * 0.1)
            self._w_pos = min(1.0, self._w_pos + _LNN_ALPHA * 0.05)
        elif last_risk == "STABLE":
            self._w_vel = max(0.05, self._w_vel - _LNN_ALPHA * 0.05)

        total_w = self._w_vel + self._w_acc + self._w_pos + 1e-9
        w_vel = self._w_vel / total_w
        w_acc = self._w_acc / total_w
        w_pos = self._w_pos / total_w

        if drift_detected:
            self._base_var = var_x * 0.7 + self._base_var * 0.3  # adapt baseline

        # ── Position danger ──────────────────────────────────────────────
        cur_x    = wx[-1]
        dist_lz  = max(0, lz - cur_x)           # how far into left danger zone
        dist_rz  = max(0, cur_x - rz)           # how far into right danger zone
        edge_dist = max(dist_lz, dist_rz)        # 0 = inside safe zone
        pos_danger = min(edge_dist / max(safe_w * 0.5, 1), 1.0)

        # ── Velocity danger ──────────────────────────────────────────────
        # Positive if moving toward nearest edge
        if cur_x < (lz + rz) / 2:
            vel_toward_edge = -vel_x               # moving left = more dangerous
        else:
            vel_toward_edge = vel_x                # moving right = more dangerous
        vel_danger = min(max(vel_toward_edge / 10.0, 0), 1.0)

        # ── Acceleration danger ──────────────────────────────────────────
        acc_danger = min(abs(acc_x) / 5.0, 1.0)

        # ── Combined risk score (LNN-weighted) ───────────────────────────
        risk_score = (
            w_pos * pos_danger +
            w_vel * vel_danger +
            w_acc * acc_danger +
            0.1   * anomaly_score +
            0.05  * ar_rate
        )
        risk_score = min(risk_score, 1.0)

        # ── Fall probability ─────────────────────────────────────────────
        fall_prob = risk_score ** 0.8       # slightly nonlinear

        # ── Time to fall (extrapolate at current velocity) ───────────────
        if vel_toward_edge > 0.2:
            dist_to_edge = safe_w / 2 - edge_dist + 1
            time_to_fall = max(dist_to_edge / (vel_toward_edge + 1e-6), 0)
        else:
            time_to_fall = None

        # ── Next state ───────────────────────────────────────────────────
        if risk_score > 0.70:
            next_state = "FALL IMMINENT"
        elif risk_score > 0.35:
            next_state = "DRIFT WARNING"
        else:
            next_state = "STABLE"

        # ── Sleep quality (inverse of movement / instability) ────────────
        avg_vel = sum(abs(wx[i] - wx[i-1]) for i in range(1, len(wx))) / max(len(wx)-1, 1)
        sleep_quality = max(0.0, 1.0 - (avg_vel / 10.0 + anomaly_score * 0.5))

        # ── Confidence ───────────────────────────────────────────────────
        confidence = min(len(self.window) / WINDOW_SIZE, 1.0)

        # ── Forecast: next 5 timesteps (simple linear extrapolation) ─────
        future_vitals = []
        for i in range(1, 6):
            fx = cur_x + vel_x * i + 0.5 * acc_x * i * i
            fy = wy[-1] + vel_y * i
            future_vitals.append([round(fx, 2), round(fy, 2)])

        # ── Current state ─────────────────────────────────────────────────
        if pos_danger > 0:
            current_state = "FALL IMMINENT" if pos_danger > 0.5 else "DRIFT WARNING"
        else:
            current_state = pts[-1]["risk"]

        color_map = {
            "STABLE": "green",
            "DRIFT WARNING": "yellow",
            "FALL IMMINENT": "red",
        }

        # ── Alert ─────────────────────────────────────────────────────────
        if next_state == "FALL IMMINENT":
            alert_level = "critical"
            alert_msg   = "⚠️ CRITICAL: Patient about to fall! Immediate intervention needed."
        elif risk_score > 0.7 or next_state == "DRIFT WARNING":
            alert_level = "warning"
            alert_msg   = "⚡ WARNING: Patient drifting toward the edge. Monitor closely."
        else:
            alert_level = "none"
            alert_msg   = "Patient is stable."

        # ── Assemble payload ─────────────────────────────────────────────
        payload = {
            "predictions": {
                "fall_probability":  round(fall_prob, 3),
                "time_to_fall_sec":  round(time_to_fall, 1) if time_to_fall else None,
                "next_state":        next_state,
                "risk_score":        round(risk_score, 3),
                "future_vitals":     future_vitals,
                "sleep_quality_score": round(sleep_quality, 3),
                "confidence":        round(confidence, 3),
            },
            "ui": {
                "cards": {
                    "fall_probability": f"{fall_prob * 100:.1f}%",
                    "time_to_fall":     f"{time_to_fall:.1f} sec" if time_to_fall else "N/A",
                    "risk_score":       f"{risk_score:.2f}",
                    "sleep_quality":    f"{sleep_quality * 100:.1f}%",
                },
                "status": {
                    "current_state": current_state,
                    "next_state":    next_state,
                    "color":         color_map.get(next_state, "green"),
                },
                "charts": {
                    "historical": [{"x": p["wx"], "y": p["wy"]} for p in pts[-20:]],
                    "forecast":   [{"x": f[0],    "y": f[1]}    for f in future_vitals],
                },
                "alerts": {
                    "level":   alert_level,
                    "message": alert_msg,
                },
                "model": {
                    "adaptive_learning": True,
                    "drift_detected":    drift_detected,
                    "drift_intensity":   round(drift_intensity, 3),
                    "weights": {
                        "velocity":     round(w_vel, 3),
                        "acceleration": round(w_acc, 3),
                        "position":     round(w_pos, 3),
                    },
                },
            },
            "_meta": {
                "timestamp":  datetime.now().isoformat(),
                "window_size": len(self.window),
                "total_rows":  self._last_row_count,
            },
        }
        return payload

    # ─────────────────────────────────────────
    # Report saving
    # ─────────────────────────────────────────
    def _save_report(self, payload: dict):
        fname = datetime.now().strftime("report_%Y%m%d_%H%M%S.json")
        path  = os.path.join(REPORT_DIR, fname)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        self._last_report_path = path
        print(f"  [Monitor] Report saved → {path}")

    def force_save_report(self):
        if self.last_prediction:
            self._save_report(self.last_prediction)
