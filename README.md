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

Full step-by-step instructions — wiring, flashing, corner selection — are in
**[docs/RUN_STEPS.txt](docs/RUN_STEPS.txt)**.

The short version:

```bash
python src\app.py            # terminal 1 - status server on :5000
python src\img_process.py    # terminal 2 - vision pipeline
```

Then click the four bed corners (TL → TR → BR → BL) and open
`http://<PC_IP>:5000/live`.

## Layout

```
src/          the running system
  img_process.py    vision pipeline: tracking, zone decision, live view
  app.py            status server the ESP32 polls (/status, /live, /tilt)
  fl_server.py      federated learning aggregator
  fl_client.py      federated learning client (one per bed)

firmware/     Arduino sketches
  live_monitor/     ESP32-CAM - MJPEG streaming
  motor_testing/    ESP32 - servo control, one motor per side

models/       weights used at runtime
  yolov8s.pt        object detector used for tracking
  yolov8n.pt        smaller detector, offline fallback
  zone_model_cnn.pt zone classifier (trained by tools/train_zone_cnn.py)

backend/      risk classifier and adaptive engine
  models/risk_classifier.pkl   the live "brain" (RandomForest)
  monitor_engine.py, ml_trainer.py, web_server.py

tools/        dataset preparation and model training (not needed to run)
docs/         RUN_STEPS.txt and analysis notes
libraries/    vendored ESP32Servo
```

## Cameras

`img_process.py` works with either camera. Set `CAM_SOURCE`:

```bash
set CAM_SOURCE=phone     # IP Webcam app - lower latency, 1080p
set CAM_SOURCE=esp32     # ESP32-CAM
set CAM_SOURCE=ask       # prompt at startup, showing which respond
```

## Not in this repo

The IR training dataset and generated animations are excluded — large and
reproducible from `tools/`. Runtime state (`status.json`, `position_log.csv`,
`fl_store/`) is written to the repository root and regenerated on every run.

The thermal/IR processing path was removed; the live pipeline works on RGB
video.
