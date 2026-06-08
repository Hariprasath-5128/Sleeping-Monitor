# Sleeping Monitor - Detailed Code Analysis Report

This document details the mechanics and purpose of each code file within the `sleeping-monitor` project, separated into the two core subsystems: the **Web Simulation UI** and the **Live Camera Monitoring Backend**.

---

## 1. Web UI & 3D Simulation Frontend

This sub-system provides a high-fidelity, interactive 3D clinical simulation of a patient in a hospital bed. It uses Three.js for 3D rendering and an HTML5 Canvas for the 2D bed overlay.

### `app.js`
**Purpose:** Core 3D engine and simulation logic.
**Details:**
- **Three.js Setup:** Initializes the 3D scene, camera, lights, and renderer.
- **Model Parsing (`loadModel`):** Loads `models/human.obj`. It contains advanced logic to procedurally extract and segment the single 3D mesh into separate body parts (Torso, Left/Right Arms) using geometric thresholds and bounding boxes. This allows the simulation to independently rotate limbs.
- **2D Canvas Overlay (`drawBase`):** Draws the clinical bed, grid background, and safety boundary margins (using the `simCanvas`).
- **Safety Logic (`checkStatus`):** Computes if the 3D model's torso or limbs intersect with the configurable "safe zone" margins. Updates the system state between SAFE, WARNING, DANGER, and FALLEN.
- **Interactivity:** Handles mouse raycasting to let the user drag the 3D patient around the bed and triggers status updates dynamically.

### `index.html`
**Purpose:** The main application shell for the 3D simulation.
**Details:**
- Sets up the DOM structure, including the `simCanvas` for 2D overlays and `threeOverlay` for the 3D renderer.
- Contains the HTML for the sidebar "Control Panel" which provides sliders for bed dimensions, safe zone margins, limb angles, and buttons to toggle simulation modes.
- Imports `app.js` as an ES module and loads the Three.js library via CDN import maps.

### `style.css`
**Purpose:** The stylesheet for the simulation dashboard.
**Details:**
- Implements a modern clinical aesthetic (dark themes, green/orange/red status colors).
- Styles the control panel, sliders, buttons, and the real-time event log.

### `models/human.obj`
**Purpose:** The 3D asset for the patient.
**Details:** A 3D geometry file loaded dynamically by `app.js` to visualize the patient.

### `monitor-ui/`
**Purpose:** Modern Frontend Application Setup.
**Details:** This folder is a scaffolding for a Vite/React (or Vanilla JS) application (contains `package.json`, `vite.config.js`, `src/`, `public/`). It seems to be a modernized or parallel frontend being built.

---

## 2. Live Camera Monitoring & ML Backend

This sub-system is the "real" computer vision backend. It captures live video, tracks a physical object (mimicking a patient) using YOLO, and uses Machine Learning to predict fall risks.

### `img_process.py`
**Purpose:** Live stream capture and visual orchestration.
**Details:**
- **Stream Capture:** Connects to an IP camera stream via OpenCV. Uses a dedicated thread (`FreshFrameCapture`) to prevent buffering lag.
- **Bed Detection:** Uses HSV color masking (looking for orange/brown) to automatically locate a physical cardboard box/bed in the video feed and calculates a perspective transform matrix (`M`) to warp it into a top-down "bird's-eye" view.
- **Tracking Integration:** Initializes the `WhitenerTracker` (from `yolo_tracker.py`) to detect the patient/object.
- **Overlay:** Back-projects the safety zones, YOLO bounding boxes, and warning indicators from the warped space onto the original video frame and displays it using `cv2.imshow`.

### `yolo_tracker.py`
**Purpose:** Object detection and physics tracking.
**Details:**
- **YOLOv8 Inference:** Uses the pre-trained `yolov8n.pt` model to detect a specific object (the "whitener").
- **Trajectory Analysis:** Maintains a rolling history of the object's centroid in the warped perspective space.
- **Risk Calculation:** Computes velocity, acceleration, and aspect ratio changes to determine if the object is stable, drifting toward the edge, or if a fall is imminent.
- **Logging:** Appends all real-time coordinates, bounding box shapes, and calculated risk levels to `position_log.csv`.

### `monitor_engine.py`
**Purpose:** Adaptive Machine Learning inference engine.
**Details:**
- Reads the continuously updating `position_log.csv` file in real-time.
- Extracts mathematical features (velocity, variance, drift intensity, anomaly scores) over a rolling window.
- Uses an online learning (LNN-style) mechanism to dynamically adjust weights for velocity, acceleration, and position depending on what signal is most accurately predicting danger.
- Generates a comprehensive JSON payload containing predictions, fall probability, and time-to-fall estimates, saving it as `reports/live_predictions.json`.

### `ml_trainer.py`
**Purpose:** Offline Machine Learning training.
**Details:**
- Reads the historical `position_log.csv` dataset.
- Engineers features (normalized X/Y, distances to edges, rolling velocity/acceleration).
- Trains a `RandomForestClassifier` to predict fall risks based on these physical traits.
- Saves the serialized model to `models/risk_classifier.pkl` and metadata to `models/model_meta.json`.
- Outputs a detailed accuracy and confusion matrix report to `reports/training_report.json`.

### Assets & Data Logs
- **`yolov8n.pt`**: Pre-trained weights for the YOLO model.
- **`position_log.csv`**: The primary data sink. Stores the historical motion vectors and risk labels generated by the camera.
- **`models/`**: Houses the generated `risk_classifier.pkl` (Random Forest model) and its metadata.
- **`reports/`**: Houses the output JSON files containing the real-time AI insights (`live_predictions.json`) and training statistics (`training_report.json`).
