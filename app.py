"""
app.py
------
Local status server that sits between img_process.py and the motor ESP32.

img_process.py writes the current zone to status.json every frame; the ESP32
running esp/motor_testing polls /status every 2 s and drives the servos from
whatever it reads here.

    ESP32-CAM  --MJPEG-->  img_process.py  --status.json-->  app.py
                                                                |
                                                        GET /status (2 s)
                                                                v
                                                          ESP32 + servos

Run:  python app.py       (listens on 0.0.0.0:5000)
"""

import json
import os
import threading
import time

from flask import Flask, Response, jsonify, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS_JSON = os.path.join(BASE_DIR, "status.json")
STATUS_TXT = os.path.join(BASE_DIR, "status.txt")   # legacy plain-text fallback
# Annotated frame published by img_process.py for the browser view.
LIVE_FRAME = os.path.join(BASE_DIR, "live_frame.jpg")
# JPEG end-of-image marker, used to detect a half-written frame.
JPEG_EOI = bytes((0xFF, 0xD9))

# If img_process.py has not refreshed the file within this many seconds we
# treat the pipeline as dead and report STALE, so the ESP32 parks the servos
# at their start positions instead of repeating an old DANGER command forever.
STALE_AFTER_S = 10.0

DEFAULT_PAYLOAD = {
    "status": "SAFE",
    "confidence": 0.0,
    "cx": -1,
    "cy": -1,
    "tracking": False,
    "stream_ok": False,
    "frame": 0,
}


def _read_status():
    """Return (payload_dict, age_seconds). Falls back to status.txt, then default."""
    if os.path.exists(STATUS_JSON):
        try:
            with open(STATUS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            age = time.time() - float(data.get("ts", 0.0))
            return data, age
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    # Legacy path: img_process.py used to write the bare zone name.
    if os.path.exists(STATUS_TXT):
        try:
            with open(STATUS_TXT, "r", encoding="utf-8") as f:
                zone = f.read().strip() or "SAFE"
            age = time.time() - os.path.getmtime(STATUS_TXT)
            payload = dict(DEFAULT_PAYLOAD, status=zone)
            return payload, age
        except OSError:
            pass

    return dict(DEFAULT_PAYLOAD), float("inf")


def _build_response():
    data, age = _read_status()

    zone = str(data.get("status", "SAFE")).strip().upper() or "SAFE"
    stale = age > STALE_AFTER_S
    if stale:
        # Pipeline is not running / has hung — command the ESP32 to idle.
        zone = "STALE"

    return {
        "status": zone,
        "confidence": round(float(data.get("confidence", 0.0)), 3),
        "cx": int(data.get("cx", -1)),
        "cy": int(data.get("cy", -1)),
        "tracking": bool(data.get("tracking", False)),
        "stream_ok": bool(data.get("stream_ok", False)),
        "frame": int(data.get("frame", 0)),
        "age_s": round(age, 2) if age != float("inf") else -1.0,
        "stale": stale,
    }


@app.route("/status")
def get_status():
    """Polled by the ESP32 motor controller every 2 s."""
    try:
        return jsonify(_build_response())
    except Exception as exc:            # never 500 at the ESP32
        app.logger.warning("status build failed: %s", exc)
        return jsonify({"status": "ERROR", "stale": True, "age_s": -1.0})


@app.route("/result")
def get_result():
    """Alias of /status — matches the endpoint name used by firmware/."""
    return get_status()


@app.route("/health")
def health():
    return jsonify({"ok": True, "ts": time.time()})


# Latest frame, held in memory. img_process.py POSTs to /frame; going through
# RAM rather than a file avoids losing frames to coarse filesystem timestamps
# and to Windows' locking during an atomic replace.
_frame_lock = threading.Lock()
_frame_cv = threading.Condition(_frame_lock)
_latest = {"jpg": None, "seq": 0}


def _store_frame(jpg):
    if not jpg or not jpg.endswith(JPEG_EOI):
        return False
    with _frame_cv:
        _latest["jpg"] = jpg
        _latest["seq"] += 1
        _frame_cv.notify_all()
    return True


def _load_frame_from_disk():
    """Fallback for when img_process.py only writes the file."""
    try:
        with open(LIVE_FRAME, "rb") as f:
            jpg = f.read()
        return jpg if jpg.endswith(JPEG_EOI) else None
    except OSError:
        return None


@app.route("/frame", methods=["POST"])
def post_frame():
    """img_process.py pushes each annotated frame here."""
    if _store_frame(request.get_data()):
        return jsonify({"ok": True, "seq": _latest["seq"]})
    return jsonify({"ok": False, "error": "not a complete JPEG"}), 400


def _mjpeg_frames():
    """Yield frames as a multipart MJPEG stream, waiting for each new one."""
    boundary = b"--frame"
    last_seq = -1
    while True:
        with _frame_cv:
            # Block until a genuinely new frame arrives; no polling, no misses.
            if _latest["seq"] == last_seq:
                _frame_cv.wait(timeout=1.0)
            jpg, seq = _latest["jpg"], _latest["seq"]

        if jpg is None or seq == last_seq:
            # Nothing pushed — fall back to whatever is on disk.
            jpg = _load_frame_from_disk()
            if jpg is None:
                continue
            seq = last_seq + 1

        last_seq = seq
        yield (boundary + b"\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
               + jpg + b"\r\n")


@app.route("/video")
def video():
    """Live annotated view from img_process.py (MJPEG)."""
    return Response(_mjpeg_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/frame.jpg")
def frame_jpg():
    """Single most recent frame — handy when MJPEG is awkward."""
    with _frame_lock:
        jpg = _latest["jpg"]
    if jpg is None:
        jpg = _load_frame_from_disk()
    if jpg is None:
        return jsonify({"error": "no frame yet"}), 404
    return Response(jpg, mimetype="image/jpeg")


@app.route("/live")
def live():
    """Full-page live view — watch the bed and set corners from a browser."""
    return (
        "<!doctype html><meta charset='utf-8'>"
        "<title>Sleeping Monitor — Live</title>"
        "<style>body{background:#111;color:#eee;font-family:system-ui,sans-serif;"
        "margin:0;padding:16px;text-align:center}"
        "img{max-width:100%;height:auto;border:2px solid #0cf;border-radius:8px}"
        "#z{font-size:1.5em;color:#0cf;font-weight:600}"
        "a{color:#0cf}</style>"
        "<h2>Sleeping Monitor &mdash; Live</h2>"
        "<p>Zone: <span id='z'>…</span></p>"
        "<img id='v' alt='live view'>"
        "<p style='opacity:.7;font-size:.9em'>Click the 4 bed corners in the "
        "<b>Select 4 Corners</b> window on the PC running img_process.py; "
        "this view mirrors it live.</p>"
        "<p id='fps' style='opacity:.5;font-size:.8em'></p>"
        "<script>"
        "const v=document.getElementById('v');let n=0,t0=Date.now();"
        "async function tick(){"
        " try{"
        "  const r=await fetch('/frame.jpg?t='+Date.now(),{cache:'no-store'});"
        "  if(r.ok){const b=await r.blob();"
        "   const u=URL.createObjectURL(b);"
        "   await new Promise(res=>{v.onload=v.onerror=res;v.src=u;});"
        "   URL.revokeObjectURL(u);n++;}"
        " }catch(e){}"
        " const el=(Date.now()-t0)/1000;"
        " if(el>=2){document.getElementById('fps').textContent="
        "   (n/el).toFixed(1)+' fps';n=0;t0=Date.now();}"
        " setTimeout(tick,0);"
        "}"
        "tick();"
        "setInterval(async()=>{try{"
        "const r=await fetch('/status');const j=await r.json();"
        "document.getElementById('z').textContent=j.status;"
        "}catch(e){}},1000);</script>"
    )


@app.route("/")
def index():
    r = _build_response()
    return (
        "<!doctype html><meta charset='utf-8'>"
        "<title>Sleeping Monitor Status</title>"
        "<style>body{background:#111;color:#eee;font-family:monospace;padding:24px}"
        "b{color:#0cf;font-size:2em}</style>"
        f"<h2>Sleeping Monitor</h2><p>Zone: <b>{r['status']}</b></p>"
        f"<pre>{json.dumps(r, indent=2)}</pre>"
        "<p>ESP32 endpoint: <code>/status</code> &nbsp;|&nbsp; Live view: <a href='/live'>/live</a></p>"
    )


if __name__ == "__main__":
    print("=" * 58)
    print("  Sleeping Monitor — Local Status Server")
    print("=" * 58)
    print("  Listening on http://0.0.0.0:5000")
    print("  ESP32 polls   http://<YOUR_PC_IP>:5000/status")
    print("  Live view     http://<YOUR_PC_IP>:5000/live")
    print(f"  Reading       {STATUS_JSON}")
    print("=" * 58)
    # 0.0.0.0 so the ESP32 on the same LAN can reach it.
    app.run(host="0.0.0.0", port=5000, threaded=True)
