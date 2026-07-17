# 🏥 V3D Sleeping Monitor - Clinical Simulation System

A high-fidelity, real-time safety monitoring simulation designed for clinical environments. This system uses a dual-engine approach (Three.js + 2D Canvas) to monitor patient positioning and identify potential fall risks using precision limb-detection logic.

## 🚀 Getting Started

The application uses **Three.js** to load high-resolution 3D assets. Because of modern browser security (CORS), the project must be run from a local web server.

### Prerequisites
- [Node.js](https://nodejs.org/) (for serving the app)

### Running Locally
1. Open your terminal in the project folder.
2. Start a local server:
   ```bash
   npx serve -p 3030 .
   ```
3. Access the application at: **`http://localhost:3030`**

---

## 🛠️ Key Features

### 1. Clinical Safety Logic (Vertical Strip)
The monitor defines a **central safe zone boundary**. Unlike standard proximity sensors, this engine distinguishes between **Limb movement** and **Torso displacement**:
- **SAFE (Cyan)**: Patient is centered within the boundaries.
- **WARNING (Orange)**: Any limb (arm/leg) crosses the boundary, or the torso enters a proximity buffer.
- **DANGER (Red)**: The patient's torso (center of mass) crosses the safety line.
- **FALLEN (Deep Red)**: The patient's torso is detected significantly outside the mattress area.

### 2. High-Fidelity Clinical Aesthetic
Designed to match modern medical dashboard standards:
- **Simulation Bed**: Dark Blueprint grid with a cream-colored medical mattress and solid brown framing.
- **Pristine Dashboard**: A secondary white high-contrast sidebar for real-time data analysis and control.
- **Dynamic 3D Human**: A reactive human model that physically changes color-state (SkyBlue -> Orange -> Red) based on safety triggers.

### 3. Integrated Control Suite
- **Interactive Morphing**: Real-time adjustment of bed width (300px - 800px) and height.
- **Adjustable Safety Strips**: Manual control over Left and Right margins. (Clinically tuned defaults: 122px / 114px).
- **Precision Dragging**: Raycast-enabled interaction—move the patient only by clicking directly on the 3D model.

---

## 🧬 Technology Stack
- **Three.js**: 3D Environment rendering & Raycasting.
- **OBJLoader**: Import and optimization of the `human.obj` model.
- **HTML5 Canvas**: High-performance 2D bed overlay and boundary rendering.
- **Vanilla JavaScript & CSS**: Logic and modern medical styling (Google Fonts: Inter).

---

## 📂 Project Structure
- `index.html`: Main UI architecture and control dashboard.
- `app.js`: Core simulation logic, 3D setup, and safety monitoring engine.
- `style.css`: Clinical theme definitions and responsive layout.
- `models/human.obj`: The primary 3D geometry for the patient.
