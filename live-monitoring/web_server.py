"""
web_server.py
-------------
HTTP bridge for the live monitoring pipeline.

Run this beside img_process.py. It reads the latest position_log.csv and report
JSON files, then exposes:
  /api/status  - flattened dashboard payload for the React UI
  /result      - compact payload intended for ESP32 polling
  /stream      - Server-Sent Events heartbeat/file-change notifications
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from monitor_engine import AdaptiveMonitorEngine


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "position_log.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LIVE_PREDICTIONS_PATH = os.path.join(REPORTS_DIR, "live_predictions.json")
TRAINING_REPORT_PATH = os.path.join(REPORTS_DIR, "training_report.json")

HOST = "0.0.0.0"
PORT = 5050
HISTORY_LIMIT = 80

ENGINE = AdaptiveMonitorEngine(CSV_PATH)


def _read_json(path: str, fallback):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return fallback


def _read_recent_rows(limit: int = HISTORY_LIMIT) -> list[dict]:
    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except (FileNotFoundError, OSError):
        return []

    parsed = []
    start = max(0, len(rows) - limit)
    for idx, row in enumerate(rows[start:], start=start):
        try:
            parsed.append({
                "index": idx + 1,
                "timestamp": row.get("timestamp", ""),
                "warp_x": float(row.get("warp_x", 0) or 0),
                "warp_y": float(row.get("warp_y", 0) or 0),
                "aspect_ratio": float(row.get("aspect_ratio", 1) or 1),
                "left_z": float(row.get("left_z", 60) or 60),
                "right_z": float(row.get("right_z", 340) or 340),
                "risk": row.get("risk", "STABLE") or "STABLE",
            })
        except ValueError:
            continue
    return parsed


def _risk_color_value(risk: str) -> float:
    risk = (risk or "").upper()
    if "FALL" in risk:
        return 1.0
    if "DRIFT" in risk:
        return 0.55
    return 0.12


def _edge_info(row: dict) -> tuple[bool, str]:
    wx = row.get("warp_x", 0)
    left_z = row.get("left_z", 60)
    right_z = row.get("right_z", 340)
    if wx < left_z:
        return True, "LEFT"
    if wx > right_z:
        return True, "RIGHT"
    return False, ""


def build_status_payload() -> dict:
    engine_payload = ENGINE.update() or ENGINE.last_prediction or {}
    live = _read_json(LIVE_PREDICTIONS_PATH, {})
    training = _read_json(TRAINING_REPORT_PATH, {})
    rows = _read_recent_rows()

    latest = rows[-1] if rows else {
        "index": 0,
        "timestamp": "",
        "warp_x": 0.0,
        "warp_y": 0.0,
        "aspect_ratio": 1.0,
        "left_z": 60.0,
        "right_z": 340.0,
        "risk": "NO DATA",
    }

    predictions = engine_payload.get("predictions", {})
    ui = engine_payload.get("ui", {})
    model = ui.get("model", {})

    risk_pred = latest.get("risk", "NO DATA")
    risk_score = predictions.get("risk_score", _risk_color_value(risk_pred))
    fall_probability = predictions.get("fall_probability", risk_score) * 100
    sleep_quality = predictions.get("sleep_quality_score", 0) * 100
    time_to_fall = predictions.get("time_to_fall_sec")
    on_edge, edge_side = _edge_info(latest)

    forecast = [
        {"label": f"+{idx + 1}", "value": point[0] if isinstance(point, list) else point.get("x", 0)}
        for idx, point in enumerate(predictions.get("future_vitals", []))
    ]

    history = []
    for idx, row in enumerate(rows):
        history.append({
            "index": idx + 1,
            "timestamp": row["timestamp"],
            "warp_x": round(row["warp_x"], 2),
            "warp_y": round(row["warp_y"], 2),
            "risk": row["risk"],
            "risk_score": _risk_color_value(row["risk"]),
        })

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_csv": CSV_PATH,
        "latest_timestamp": latest.get("timestamp", ""),
        "warp_x": round(float(latest.get("warp_x", 0)), 2),
        "warp_y": round(float(latest.get("warp_y", 0)), 2),
        "aspect_ratio": round(float(latest.get("aspect_ratio", 1)), 3),
        "left_z": latest.get("left_z", 60),
        "right_z": latest.get("right_z", 340),
        "risk_pred": risk_pred,
        "next_pred": predictions.get("next_state", risk_pred),
        "confidence": round(float(predictions.get("confidence", 0)) * 100, 1),
        "fall_probability": round(float(fall_probability), 1),
        "risk_score": round(float(risk_score), 3),
        "sleep_quality": round(float(sleep_quality), 1),
        "time_to_fall": f"{time_to_fall:.1f} sec" if isinstance(time_to_fall, (int, float)) else "N/A",
        "on_edge": on_edge,
        "edge_side": edge_side,
        "alert": ui.get("alerts", {}),
        "weights": model.get("weights", {}),
        "drift_detected": model.get("drift_detected", False),
        "drift_intensity": model.get("drift_intensity", 0),
        "model_accuracy": live.get("model_accuracy", training.get("accuracy", 0)),
        "n_train": training.get("n_train"),
        "n_test": training.get("n_test"),
        "per_class_metrics": training.get("per_class_metrics", {}),
        "history": history,
        "forecast": forecast,
    }


def build_esp32_payload() -> dict:
    status = build_status_payload()
    return {
        "time": status["generated_at"],
        "risk": status["risk_pred"],
        "next": status["next_pred"],
        "fall_probability": status["fall_probability"],
        "risk_score": status["risk_score"],
        "warp_x": status["warp_x"],
        "warp_y": status["warp_y"],
        "on_edge": status["on_edge"],
        "edge_side": status["edge_side"],
        "alert": status.get("alert", {}).get("message", ""),
    }


class MonitorRequestHandler(BaseHTTPRequestHandler):
    server_version = "SleepingMonitorHTTP/1.0"

    def _send_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send_json(self, payload: dict, status=200):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._send_headers(status=status)
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/health"):
            self._send_json({"ok": True, "service": "sleeping-monitor", "port": PORT})
        elif path == "/api/status":
            self._send_json(build_status_payload())
        elif path == "/result":
            self._send_json(build_esp32_payload())
        elif path == "/stream":
            self._handle_stream()
        else:
            self._send_json({"error": "not found", "path": path}, status=404)

    def _handle_stream(self):
        self._send_headers(content_type="text/event-stream")
        last_mtime = 0.0
        while True:
            try:
                mtime = os.path.getmtime(CSV_PATH) if os.path.exists(CSV_PATH) else 0.0
                event = "change" if mtime != last_mtime else "heartbeat"
                last_mtime = mtime
                payload = json.dumps({"event": event, "time": time.time()})
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                time.sleep(2)
            except (BrokenPipeError, ConnectionResetError):
                break

    def log_message(self, fmt, *args):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {self.address_string()} {fmt % args}")


class MonitorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


def main():
    server = MonitorHTTPServer((HOST, PORT), MonitorRequestHandler)
    print("=" * 58)
    print("  Sleeping Monitor web server")
    print(f"  Local dashboard API: http://localhost:{PORT}/api/status")
    print(f"  ESP32 endpoint:      http://192.168.1.4:{PORT}/result")
    print("  Press Ctrl+C to stop")
    print("=" * 58)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
