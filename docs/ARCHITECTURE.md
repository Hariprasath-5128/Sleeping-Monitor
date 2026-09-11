# Sleeping Monitor — Workflow and System Architecture

*Vision-based bed-fall prevention with privacy-preserving federated learning*

**Document status:** derived from the running implementation (`src/`, `backend/`,
`firmware/`). Every figure quoted here is read from committed artifacts, not
estimated — sources are named inline so each can be re-verified.

---

## 1. Problem statement

Falls from hospital and care beds are among the most common in-facility patient
safety incidents, and they happen quickly: a patient who begins to roll toward
an edge is past the point of self-correction within a second or two. Bed-exit
alarms in current use are *reactive* — they announce that a fall has already
begun and depend on a staff member being close enough to intervene in time.

This project takes a different position: **the system should act, not just
alarm.** A camera watches the bed, a two-stage vision pipeline judges whether
the patient is drifting toward an edge, and a bedside microcontroller lifts the
side panel on the threatened side to arrest the roll before it completes. The
alarm still fires; it is simply no longer the only line of defence.

A second constraint shapes the whole design. Patient video is among the most
sensitive data a care facility holds, and a system that ships bedroom footage to
a cloud service is difficult to deploy for exactly that reason. The learning
layer here is therefore **federated**: each bed trains on its own recorded
motion and uploads only a trained model. No frame, and no position log, leaves
the bedside.

---

## 2. Design principles

Four commitments constrain the architecture, and each shows up as a concrete
mechanism later in this document.

**P1 — Physical intervention over notification.**
The output of the system is servo motion, not a message. This raises the cost of
a false positive from "an ignored beep" to "an unnecessary bed movement", which
drives P2.

**P2 — Geometry outranks the model on safety-critical calls.**
A learned classifier generalises imperfectly across beds, cameras and lighting.
Where an object physically sits in the frame is not a matter of opinion. The
system therefore lets the model propose and lets measured geometry veto
(§6.3). This is the central safety argument of the design.

**P3 — Raw data never leaves the bedside.**
Privacy is enforced structurally, by what the protocol is capable of
transmitting, rather than by policy (§7).

**P4 — Degrade, never stop.**
A monitor that fails silently is worse than none. Every stage has a defined
fallback, and a stalled pipeline is treated as an explicit state that returns
the bed to neutral (§8).

---

## 3. System overview

Four components, each with one responsibility, connected by HTTP over the local
network.

```
┌──────────────┐   MJPEG    ┌──────────────────┐   zone   ┌──────────┐  HTTP  ┌─────────────┐
│ Phone camera │──────────▶ │  img_process.py  │────────▶ │  app.py  │◀──────▶│   ESP32     │
│  IP Webcam   │  /video    │  vision pipeline │  JSON    │  :5000   │ 250 ms │  + 2 servos │
└──────────────┘            └──────────────────┘          └──────────┘        └─────────────┘
                                     │                          ▲                    │
                              position_log.csv                  │  annotated         │ measured
                                     │                          │  frame + tilt      │ tilt
                                     ▼                          │                    │
                            ┌──────────────────┐                └────────────────────┘
                            │   fl_client.py   │
                            │  local training  │        Clinician view: /live
                            └──────────────────┘
                                     │  model only (never data)
                                     ▼
                            ┌──────────────────┐
                            │   fl_server.py   │  aggregates models from every bed
                            │    :8081         │
                            └──────────────────┘
```

**The camera only streams.** It runs the IP Webcam Android app and serves MJPEG
at `/video` with single frames at `/shot.jpg`. No processing happens on the
phone; it is a commodity sensor, which keeps per-bed hardware cost near zero and
means a facility can deploy with devices it already owns.

**`img_process.py` does the thinking** — tracking, risk classification, zone
arbitration, and the annotated live view.

**`app.py` holds the answer and serves it.** Decoupling the decision-maker from
the decision-server means the vision pipeline can restart without the
microcontroller ever seeing a connection failure.

**The ESP32 acts**, polling four times a second and driving one servo per bed
side.

### Why the decision runs on the PC

Inference could in principle run on the microcontroller. It does not, for three
reasons: the detector is a 22 MB YOLOv8s network needing far more memory than an
ESP32 has; centralising the decision means one place to update models; and it
keeps the microcontroller's job small enough to reason about exhaustively —
poll, compare a string, move a servo.

---

## 4. Runtime workflow

### 4.1 Calibration (once per installation)

The operator clicks the four bed corners in view order (TL → TR → BR → BL). From
these, `cv2.getPerspectiveTransform` builds a homography **M** and its inverse
**M⁻¹**.

This step earns its place. It converts an oblique camera view into a
**640 × 640 bird's-eye canvas** in which the bed is axis-aligned and "distance
to the left edge" is a straight pixel measurement rather than a
perspective-distorted one. All reasoning happens in this warped space; results
are projected back through M⁻¹ only for display. The practical payoff is that
the camera may be mounted anywhere with a view of the bed — no fixed geometry,
no calibration target.

Two draggable margins, `LEFT` and `RIGHT`, then divide the canvas into a central
safe region and two danger strips, tunable per bed.

### 4.2 Per-frame loop

```
  1. ACQUIRE      newest frame; stale frames discarded (2 s budget)
  2. WARP         frame → 640×640 bird's-eye via M
  3. TRACK        YOLOv8s → box, centroid, track ID
  4. RE-ACQUIRE   recover identity by proximity if the ID changed
  5. KINEMATICS   velocity (vx, vy), speed, acceleration (ax)
  6. CLASSIFY     RandomForest → STABLE / DRIFT WARNING / FALL IMMINENT
  7. ARBITRATE    fuse with geometry; geometry vetoes (§6.3)
  8. SMOOTH       majority vote over 8 frames
  9. PUBLISH      status.json + annotated frame → app.py
```

Steps 7 and 8 are where most of the engineering judgement sits, and §6 is
devoted to them.

### 4.3 Control loop

The ESP32 polls `/status` every **250 ms** and maps the zone to an action:

| Zone | Servo action |
|---|---|
| `SAFE` / `EMPTY` | hold neutral |
| `WARNING_LEFT` / `WARNING_RIGHT` | brace the threatened side and **stay braced** |
| `DANGER_LEFT` / `DANGER_RIGHT` | lift that side to full travel (~65°); push further if the zone persists |
| `NOT_FOUND` | hold — deliberately *not* an alarm (§8) |
| `STALE` | return to neutral — the pipeline is not reporting |

Two behaviours here came from observed failure, not from theory. **Bracing
holds** rather than returning to neutral, because a patient shifting near an
edge produces alternating WARNING/SAFE readings, and dropping on every SAFE made
the servos pump audibly up and down. And **the zone names the side**, because an
earlier version defaulted to the left on a sideless WARNING and braced the wrong
motor for anything happening on the right.

---

## 5. Perception layer

### 5.1 Tracking — YOLOv8s

`models/yolov8s.pt` (22.6 MB), running with persistent tracking at conf 0.20,
IoU 0.45.

Two filters make general-purpose detection usable on a bed. **Scenery
rejection:** pointed at a bed, YOLO confidently labels the entire surface as one
large object — in testing, a "refrigerator" covering 47 % of the frame at 0.91
confidence, which out-scored the actual patient every frame and captured the
tracker permanently. Detections covering more than 25 % of the warped view are
therefore discarded as background, and those under 0.02 % as noise.

**Proximity re-acquisition:** on a low-resolution feed the same object is
re-detected as a different COCO class from frame to frame — bottle, then vase,
then tennis racket — each with a fresh track ID. Rather than trusting the ID, a
detection within **140 warp-space pixels** of the last known centroid is treated
as the same object regardless of its label. Identity is established by
*position*, which is stable, instead of by *class*, which is not.

This is a deliberate scope decision: the system tracks *the object on the bed*,
not *a recognised person*. It needs to know where the mass is, not what it is.

### 5.2 Risk classification — RandomForest

The brain is `backend/models/risk_classifier.pkl` — 200 trees over **13 motion
features**, not pixels:

| Group | Features |
|---|---|
| Position | `warp_x`, `warp_y`, `norm_x`, `norm_y` |
| Shape | `aspect_ratio` |
| Boundary proximity | `left_gap`, `right_gap`, `in_left`, `in_right` |
| Kinematics | `vx`, `vy`, `speed`, `ax` |

**The feature choice is the important decision.** Because the model reads
geometry and motion rather than appearance, it is camera-agnostic: it transfers
across resolutions, lighting and camera placements without retraining, and it
cannot memorise what a patient looks like. Privacy here is a property of the
representation, not only of the protocol.

**Measured performance** (`backend/reports/training_report.json`, 5 173 train /
1 294 test):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| STABLE | 0.948 | 0.895 | 0.921 | 731 |
| FALL IMMINENT | 0.870 | 0.868 | 0.869 | 471 |
| DRIFT WARNING | 0.440 | 0.641 | 0.522 | 92 |
| **Overall accuracy** | | | **86.71 %** | 1 294 |

Reported honestly, DRIFT WARNING is weak — 0.44 precision on 92 support. It is
an intermediate state with few examples and fuzzy boundaries against both
neighbours. This is precisely why the architecture does not let the classifier
act alone, and it motivates §6.3. The two classes that matter operationally,
STABLE and FALL IMMINENT, are both near 0.87 F1.

---

## 6. Decision layer

### 6.1 Geometric zoning

Independently of the model, the system computes a zone from box overlap with the
danger strips: **≥ 38 %** of the body past a boundary is DANGER, **≥ 15 %** is
WARNING.

Overlap is used rather than the centroid because a patient lying across the bed
has a wide box that clips a danger strip while their centre of mass is safe —
and conversely, a wide box keeps the centroid nominally inside long after the
patient is genuinely going over the edge.

### 6.2 Confidence gating

Predictions below **0.50** confidence are discarded in favour of geometry. An
unconfident DANGER call is worse than no call, because it moves the bed.

### 6.3 Geometric veto — the core safety mechanism

**Geometry has the final say on DANGER.**

The classifier was trained on one bed, camera and object; deployed elsewhere it
will confidently call DANGER while the body sits plainly inside the safe zone.
So three overrides are applied unconditionally:

1. Model says DANGER, geometry says SAFE or EMPTY → **geometry wins.**
2. Model says DANGER_LEFT, geometry says DANGER_RIGHT → **geometry wins** (and
   the mirror case).
3. Model says SAFE, geometry says WARNING → **the warning is raised.**

The asymmetry is intentional and is the design's central claim. The model may
freely *escalate* caution; it may not *override physical measurement* to justify
moving the bed. Where the patient is, is measurable. Whether they are about to
fall is a judgement — and only the judgement is delegated to a learned model.

### 6.4 Temporal smoothing

A zone is committed only if it wins a majority vote over the last **8 frames**,
preventing single-frame glitches from reaching the motors.

---

## 7. Federated learning layer

### 7.1 Motivation

Every bed sees a different patient, mattress, camera angle and lighting
condition. A single centrally trained model is a compromise across all of them.
The obvious fix — pool everyone's footage — is exactly what patient privacy
forbids.

Federated learning resolves the tension: **each bed trains locally and shares
only what it learned, never what it saw.**

### 7.2 Forest merging, not FedAvg

Standard FedAvg averages network weights. That is meaningless for a
RandomForest, whose parameters are discrete tree structures.

The correct aggregation for a bagged ensemble is a **forest merge**: pool the
clients' trees into one global ensemble. Each tree is already an independent
voter trained on a bootstrap sample, so pooling preserves exactly the mechanism
RandomForest relies on. Concretely (`src/fl_server.py`):

- Each client's tree quota is **proportional to its sample count** — the FedAvg
  weighting rule applied to an ensemble.
- The global forest is capped at **400 trees** so it cannot grow without bound.
- **25 % of the slots are reserved for the previous global model**, so knowledge
  persists across rounds instead of each round starting from only the newest
  clients.
- Uploads are validated before merging: a client whose class set, feature order
  or class count disagrees is **rejected**, since its trees would be voting on a
  different question.
- Aggregation triggers once ≥ 2 clients have reported.

### 7.3 What crosses the network

| Stays at the bedside | Leaves the bedside |
|---|---|
| Video frames | Trained tree structures |
| `position_log.csv` | Label encoder, feature order |
| Annotated live view | Sample count (for weighting) |

A client with fewer than 50 logged rows does not participate — below that a
local model is noise rather than knowledge.

### 7.4 Round protocol

```
   bedside                        aggregator (:8081)
   ───────                        ──────────────────
   train on position_log.csv
        │  POST /upload  (model only)
        ├──────────────────────────▶  validate schema
        │                             queue for round
        │                             ≥2 clients → merge
        │  GET /global                weight by n_samples
        ◀──────────────────────────   cap at 400 trees
   install as live brain              retain 25 % prior
```

Because the aggregator seeds from the committed classifier, a new bed starts at
86.71 % accuracy rather than from zero — no cold-start period during which the
bed is unprotected.

---

## 8. Failure handling

Every stage has a defined degradation path, following P4.

| Failure | Response |
|---|---|
| Camera drops frames | Retry and rebuild the connection every 15 s; publish `NOT_FOUND` |
| Frame older than 2 s | Discarded — a stale frame is worse than none |
| Truncated MJPEG frame | Skipped silently; expected on a live feed |
| Model unconfident (< 0.50) | Fall back to geometry |
| Model missing | Geometry alone; system still runs |
| Detection lost | Hold last state, then `NOT_FOUND` after 30 frames |
| **Pipeline stops publishing** | `/status` reports `STALE` after 10 s → **servos return to neutral** |
| Server unreachable | ESP32 idles at neutral after 3 failed polls |

Two of these deserve emphasis.

**`NOT_FOUND` is deliberately not an alarm.** The detector routinely drops a
stationary patient under bedding for several seconds. Treating every dropout as
an emergency would make the system cry wolf until staff stopped believing it.
The bed holds position and reports the uncertainty honestly.

**Staleness is fail-safe by construction.** If the vision pipeline crashes, the
last value served would otherwise be a stale DANGER, leaving the bed tilted
indefinitely. Instead the server marks any reading older than 10 s as `STALE`,
and the firmware returns the bed to neutral. **A dead monitor cannot hold the bed
in an emergency posture.**

---

## 9. Clinical interface

`app.py` serves a browser view at `/live` — annotated video with the bed
outline, safe/danger zones, tracked box, motion trail, current zone badge, and
the bed's real tilt as measured and reported back by the ESP32. Corner
calibration is available from the same page, so a bed can be set up without
touching the machine running the pipeline.

The tilt path closes the loop: the interface shows the bed's *actual* measured
angle, not the angle that was commanded.

---

## 10. Evaluation plan

The prototype is validated; the following establishes clinical claims.

**Bench validation (done).** Classifier: 86.71 % over 1 294 held-out samples.
Pipeline: end-to-end from stream to servo motion, with fallbacks exercised —
unreachable camera, missing model, stale status.

**Proposed protocol.**

1. *Detection performance* — scripted roll sequences (left, right, restless,
   return-to-centre) with manually annotated ground truth; report per-class
   precision/recall, with **time-to-detection** as the headline metric, since
   intervention latency determines whether the lift arrives in time.
2. *Intervention efficacy* — weighted mannequin roll trials; report the
   proportion arrested at WARNING versus DANGER.
3. *False-positive burden* — extended runs with normal sleep movement; report
   unnecessary activations per night, the metric that governs whether staff and
   patients tolerate the system.
4. *Federated benefit* — train per-bed models on disjoint conditions; compare
   local-only against post-aggregation accuracy on each bed's held-out data. The
   claim to test is that a bed's accuracy improves from other beds' experience
   without their data.

**Known limitations, stated plainly.** DRIFT WARNING precision is 0.44 and needs
more labelled intermediate examples. The classifier was trained on one bed and
depends on the geometric veto when transferred. Manual corner calibration is
required per installation. Evaluation to date is bench-scale, with no clinical
trial.

---

## 11. Component reference

| Path | Role |
|---|---|
| `src/img_process.py` | Vision pipeline: tracking, classification, arbitration |
| `src/app.py` | Status server (`/status`, `/live`, `/tilt`, `/video`) |
| `src/fl_server.py` | Federated aggregator — forest merging |
| `src/fl_client.py` | Per-bed client — local training, upload, pull |
| `backend/ml_trainer.py` | Trains the risk classifier from position logs |
| `backend/models/risk_classifier.pkl` | Live brain — 200-tree RandomForest |
| `firmware/motor_testing/` | ESP32 servo control, one motor per side |
| `models/yolov8s.pt` | Object detector for tracking |
| `tools/` | Dataset preparation and training utilities |

---

## 12. Summary

A commodity phone camera and a microcontroller convert a standard bed into one
that intervenes physically when a patient begins to roll toward an edge. A
YOLOv8s tracker locates the patient in a calibrated bird's-eye view; a
13-feature RandomForest judges risk from motion rather than appearance at
86.71 % accuracy; and a geometric veto ensures no learned model can move the bed
against physical measurement. Federated learning lets every bed improve from
every other bed's experience while raw video and position logs never leave the
bedside.

The system's defining property is that its *safety* does not rest on its
*intelligence*. The model contributes anticipation; measured geometry retains
authority; and every failure path terminates with the bed returning to neutral.
