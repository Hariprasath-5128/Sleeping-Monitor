# Sleeping Monitor

Bed-fall prevention prototype. A camera watches the bed, a vision pipeline
decides whether the patient is drifting towards an edge, and an ESP32 lifts the
side panel on that side to stop the roll.

```
  camera  ──video──▶  img_process.py  ──zone──▶  app.py  ──HTTP──▶  ESP32 + servos
 (phone /                (YOLO +                (:5000)              (one motor
  ESP32-CAM)          risk classifier)                                per side)
```

The camera only streams. `img_process.py` does the thinking. `app.py` holds the
answer and serves it. The motor ESP32 polls that answer and tilts the bed.

## Running it

Full step-by-step instructions, including wiring, flashing and the corner
selection, are in **[RUN_STEPS.txt](RUN_STEPS.txt)**.

The short version:

```bash
python app.py            # terminal 1 - status server on :5000
python img_process.py    # terminal 2 - vision pipeline
```

Then click the four bed corners (TL → TR → BR → BL) and open
`http://<PC_IP>:5000/live`.

## Layout

| Path | What it is |
|---|---|
| `img_process.py` | Vision pipeline: tracking, zone decision, live view |
| `app.py` | Status server the ESP32 polls (`/status`, `/live`, `/tilt`) |
| `esp/live_monitor/` | ESP32-CAM sketch — MJPEG streaming |
| `esp/motor_testing/` | ESP32 sketch — servo control, one motor per side |
| `backend/` | Risk classifier, training code, adaptive engine |
| `fl_server.py`, `fl_client.py` | Federated learning: beds share models, never data |
| `model/`, `tools/` | Zone-model training and dataset preparation |
| `yolov8s.pt` | Object detector used for tracking |

## Cameras

`img_process.py` works with either camera. Set `CAM_SOURCE`:

```bash
set CAM_SOURCE=phone     # IP Webcam app - lower latency, 1080p
set CAM_SOURCE=esp32     # ESP32-CAM
set CAM_SOURCE=ask       # prompt at startup, showing which respond
```

## Not in this repo

The IR training dataset (`data/`) and generated animations are excluded — they
are large and reproducible from the scripts in `tools/`. Runtime state
(`status.json`, `position_log.csv`, `fl_store/`) is regenerated on every run.
