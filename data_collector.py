#!/usr/bin/env python3
"""
data_collector.py  —  Manual Dataset Collection Tool
=====================================================
Captures warped (bird's-eye) frames from the IP camera and saves
them into labelled class folders for training a custom model.

Features:
- Corner selection for perspective warp
- YOLOv8 visual feedback (shows bounding box on object, NOT saved to dataset)
- Zone boundary adjustments
- Selectively delete past images
- View random samples
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import csv
import time
import random
from datetime import datetime
from PIL import Image, ImageTk

# YOLO for visual feedback
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
IP   = "10.144.219.160"
PORT = 8080
URL  = f"http://{IP}:{PORT}/video"

WARP_SIZE       = 640
DEFAULT_SAMPLES = 100
PANEL_W         = 340
CLASSES         = ["safe", "warning", "danger_left", "danger_right", "empty"]

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
LOG_FILE    = os.path.join(BASE_DIR, "position_log.csv")
META_FILE   = os.path.join(DATASET_DIR, "metadata.csv")

for _cls in CLASSES:
    os.makedirs(os.path.join(DATASET_DIR, _cls), exist_ok=True)

# ─────────────────────────────────────────────────────────
# ZONE COLOURS
# ─────────────────────────────────────────────────────────
ZONE_BGR = {
    "safe":         (0,  180,  60),
    "warning":      (0,  220, 255),
    "danger_left":  (0,  165, 220),
    "danger_right": (0,  165, 220),
    "empty":        (150, 150, 150),
}
CLASS_HEX = {
    "safe":         "#00c840",
    "warning":      "#ffdd00",
    "danger_left":  "#ff3322",
    "danger_right": "#ff8800",
    "empty":        "#969696",
}

# ─────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────
def order_points(pts):
    pts  = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s       = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff    = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


DST = np.array([
    [0,           0          ],
    [WARP_SIZE-1, 0          ],
    [WARP_SIZE-1, WARP_SIZE-1],
    [0,           WARP_SIZE-1],
], dtype="float32")


def build_transforms(corners):
    return (cv2.getPerspectiveTransform(corners, DST),
            cv2.getPerspectiveTransform(DST, corners))


def warp_pts_to_frame(pts_w, M_inv):
    arr = np.array(pts_w, dtype="float32").reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, M_inv).reshape(-1, 2).astype(np.int32)


# ─────────────────────────────────────────────────────────
# ZONE OVERLAY
# ─────────────────────────────────────────────────────────
def draw_zones_on_frame(canvas, M_inv, lz, rz):
    W = H = WARP_SIZE - 1
    ov = canvas.copy()
    polys = [
        (np.array([[0,0],[lz,0],[lz,H],[0,H]],   dtype="float32"), ZONE_BGR["danger_left"]),
        (np.array([[rz,0],[W,0],[W,H],[rz,H]],   dtype="float32"), ZONE_BGR["danger_right"]),
        (np.array([[lz,0],[rz,0],[rz,H],[lz,H]], dtype="float32"), ZONE_BGR["safe"]),
    ]
    for pts_w, col in polys:
        cv2.fillPoly(ov, [warp_pts_to_frame(pts_w, M_inv)], col)
    cv2.addWeighted(ov, 0.28, canvas, 0.72, 0, canvas)

    for xw in (lz, rz):
        t = warp_pts_to_frame([[xw, 0]], M_inv)[0]
        b = warp_pts_to_frame([[xw, H]], M_inv)[0]
        cv2.line(canvas, tuple(t), tuple(b), (0, 80, 160), 9, cv2.LINE_AA)
        cv2.line(canvas, tuple(t), tuple(b), (0, 220, 255), 2, cv2.LINE_AA)

    box_f = warp_pts_to_frame(np.array([[0,0],[W,0],[W,H],[0,H]], dtype="float32"), M_inv)
    cv2.polylines(canvas, [box_f.reshape(-1,1,2)], True, (0, 90, 130), 7)
    cv2.polylines(canvas, [box_f.reshape(-1,1,2)], True, (0, 230, 255), 3)

    # ── Left / Right Text Markers ──
    l_mid = warp_pts_to_frame([[lz // 2, H // 2]], M_inv)[0]
    r_mid = warp_pts_to_frame([[(rz + W) // 2, H // 2]], M_inv)[0]
    
    # Shadow + Text for Left
    cv2.putText(canvas, "<-- LEFT", (l_mid[0]-40, l_mid[1]), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(canvas, "<-- LEFT", (l_mid[0]-40, l_mid[1]), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
    
    # Shadow + Text for Right
    cv2.putText(canvas, "RIGHT -->", (r_mid[0]-40, r_mid[1]), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(canvas, "RIGHT -->", (r_mid[0]-40, r_mid[1]), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)


def draw_zones_on_warp(warp_img, lz, rz):
    H = W = warp_img.shape[0]
    ov = warp_img.copy()
    cv2.rectangle(ov, (0, 0),   (lz, H), ZONE_BGR["danger_left"],  -1)
    cv2.rectangle(ov, (rz, 0),  (W,  H), ZONE_BGR["danger_right"], -1)
    cv2.rectangle(ov, (lz, 0),  (rz, H), ZONE_BGR["safe"],         -1)
    cv2.addWeighted(ov, 0.30, warp_img, 0.70, 0, warp_img)

    for x_mid, label, col in [
        (lz // 2,        "DANGER", (20, 100, 255)),
        ((rz + W) // 2,  "DANGER", (20, 100, 255)),
        ((lz + rz) // 2, "SAFE",   (20, 220, 80)),
    ]:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(warp_img, label, (x_mid - tw//2, H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    cv2.line(warp_img, (lz, 0), (lz, H), (0, 220, 255), 2)
    cv2.line(warp_img, (rz, 0), (rz, H), (0, 220, 255), 2)


# ─────────────────────────────────────────────────────────
# METADATA LOGGER
# ─────────────────────────────────────────────────────────
def open_meta(path):
    is_new = not os.path.isfile(path) or os.path.getsize(path) == 0
    fh     = open(path, "a", newline="")
    writer = csv.writer(fh)
    if is_new:
        writer.writerow(["timestamp", "filename", "class", "session"])
        fh.flush()
    return fh, writer


# ─────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────
class DataCollector:

    CORNER_LABELS = ["TL", "TR", "BR", "BL"]

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Sleeping Monitor — Dataset Collector")
        self.root.configure(bg="#12121f")

        self.root.state("zoomed")
        self.root.update()
        sw = self.root.winfo_width()
        sh = self.root.winfo_height()
        self.CANVAS_W = sw - PANEL_W - 30
        self.CANVAS_H = sh - 60

        self.cap = cv2.VideoCapture(URL)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Initialize YOLO for visual feedback ONLY
        print("Loading YOLOv8s for visual feedback...")
        self.yolo_model = YOLO("yolov8s.pt")

        self.phase         = "corner_select"
        self.corner_clicks = []
        self.corners       = None
        self.M             = None
        self.M_inv         = None

        self._disp_scale = 1.0
        self._disp_x0    = 0
        self._disp_y0    = 0

        self.paused     = False
        self.collecting = False

        self.current_frame  = None
        self.current_warped = None
        self.yolo_bboxes    = []  # list of (x, y, w, h) in warp space
        self.lock = threading.Lock()

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.meta_fh, self.meta_writer = open_meta(META_FILE)

        self.count_vars = {cls: tk.StringVar(value=str(self._count_existing(cls)))
                           for cls in CLASSES}

        self.lz_var = tk.IntVar(value=int(0.20 * WARP_SIZE))
        self.rz_var = tk.IntVar(value=int(0.80 * WARP_SIZE))

        self._build_ui()

        self.running = True
        threading.Thread(target=self._camera_loop, daemon=True).start()
        self._update_display()

        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

    def _count_existing(self, cls):
        d = os.path.join(DATASET_DIR, cls)
        if not os.path.isdir(d):
            return 0
        return len([f for f in os.listdir(d) if f.lower().endswith(".jpg")])

    # ─────────────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────────────
    def _build_ui(self):
        left = tk.Frame(self.root, bg="#12121f")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10,4), pady=10)

        self.canvas = tk.Canvas(left, width=self.CANVAS_W, height=self.CANVAS_H,
                                bg="#0a0a14", highlightthickness=2,
                                highlightbackground="#00e5ff")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        self.hint_lbl = tk.Label(
            left,
            text="STEP 1 — Click the 4 corners of the bed surface  (TL → TR → BR → BL)",
            bg="#12121f", fg="#00e5ff",
            font=("Consolas", 11, "bold")
        )
        self.hint_lbl.pack(pady=(4, 0))

        right = tk.Frame(self.root, bg="#1c1c30", width=PANEL_W)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 10), pady=10)
        right.pack_propagate(False)

        tk.Label(right, text="DATASET COLLECTOR",
                 bg="#1c1c30", fg="#00e5ff",
                 font=("Consolas", 13, "bold")).pack(pady=(14, 10))

        self._apply_styles()
        nb = ttk.Notebook(right)
        nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        collect_tab = tk.Frame(nb, bg="#1c1c30")
        delete_tab  = tk.Frame(nb, bg="#1c1c30")
        nb.add(collect_tab, text="  Collect  ")
        nb.add(delete_tab,  text="  Manage   ")

        self._build_collect_tab(collect_tab)
        self._build_delete_tab(delete_tab)

        tk.Button(
            right, text="↺  Reset Corners",
            command=self._reset_corners,
            bg="#1a3040", fg="#aaddff",
            font=("Consolas", 9),
            relief=tk.FLAT, cursor="hand2", pady=6
        ).pack(fill=tk.X, padx=10, pady=(0, 4))

        tk.Button(
            right, text="✕  Quit",
            command=self._on_quit,
            bg="#222233", fg="#cccccc",
            font=("Consolas", 10),
            relief=tk.FLAT, cursor="hand2", pady=7
        ).pack(fill=tk.X, padx=10, pady=(0, 10))

        self._set_controls_enabled(False)

    def _build_collect_tab(self, parent):
        pad = {"padx": 10, "pady": 4}

        tk.Label(parent, text="Zone margins (warp pixels):",
                 bg="#1c1c30", fg="#9999bb",
                 font=("Consolas", 9)).pack(anchor=tk.W, **pad)

        lz_row = tk.Frame(parent, bg="#1c1c30"); lz_row.pack(fill=tk.X, padx=10)
        tk.Label(lz_row, text="Left  ", bg="#1c1c30", fg="#ffaa44",
                 font=("Consolas", 9)).pack(side=tk.LEFT)
        tk.Scale(lz_row, variable=self.lz_var,
                 from_=0, to=WARP_SIZE//2, orient=tk.HORIZONTAL,
                 bg="#1c1c30", fg="#ffaa44", troughcolor="#0a0a14",
                 highlightthickness=0, length=240).pack(side=tk.LEFT)

        rz_row = tk.Frame(parent, bg="#1c1c30"); rz_row.pack(fill=tk.X, padx=10)
        tk.Label(rz_row, text="Right ", bg="#1c1c30", fg="#ffaa44",
                 font=("Consolas", 9)).pack(side=tk.LEFT)
        tk.Scale(rz_row, variable=self.rz_var,
                 from_=WARP_SIZE//2, to=WARP_SIZE, orient=tk.HORIZONTAL,
                 bg="#1c1c30", fg="#ffaa44", troughcolor="#0a0a14",
                 highlightthickness=0, length=240).pack(side=tk.LEFT)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=8)

        tk.Label(parent, text="Label (class):",
                 bg="#1c1c30", fg="#9999bb",
                 font=("Consolas", 10)).pack(anchor=tk.W, **pad)
        self.label_var = tk.StringVar(value=CLASSES[0])
        self.label_var.trace_add("write", self._on_label_change)
        self.label_dd = ttk.Combobox(parent, textvariable=self.label_var,
                                     values=CLASSES, state="readonly",
                                     font=("Consolas", 11), width=20)
        self.label_dd.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.pause_btn = tk.Button(
            parent, text="⏸  PAUSE", command=self._toggle_pause,
            bg="#0f3460", fg="#ffffff", font=("Consolas", 11, "bold"),
            relief=tk.FLAT, cursor="hand2", pady=9
        )
        self.pause_btn.pack(fill=tk.X, padx=10, pady=(0, 8))

        sp_row = tk.Frame(parent, bg="#1c1c30"); sp_row.pack(fill=tk.X, padx=10)
        tk.Label(sp_row, text="Samples per collect:",
                 bg="#1c1c30", fg="#9999bb", font=("Consolas", 9)).pack(side=tk.LEFT)
        self.samples_var = tk.IntVar(value=DEFAULT_SAMPLES)
        tk.Spinbox(sp_row, from_=10, to=2000, increment=50,
                   textvariable=self.samples_var, width=7, font=("Consolas", 10),
                   bg="#0a0a14", fg="#ffffff", buttonbackground="#0f3460").pack(side=tk.RIGHT)

        self.collect_btn = tk.Button(
            parent, text="⬤  COLLECT", command=self._start_collect,
            bg="#005500", fg="#ffffff", font=("Consolas", 12, "bold"),
            relief=tk.FLAT, cursor="hand2", pady=11
        )
        self.collect_btn.pack(fill=tk.X, padx=10, pady=8)

        self.progress = ttk.Progressbar(parent, orient=tk.HORIZONTAL, mode="determinate")
        self.progress.pack(fill=tk.X, padx=10, pady=(0, 2))
        self.prog_lbl = tk.Label(parent, text="Ready", bg="#1c1c30", fg="#666688", font=("Consolas", 9))
        self.prog_lbl.pack(anchor=tk.E, padx=10)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=10)

        tk.Label(parent, text="Images collected:", bg="#1c1c30", fg="#9999bb", font=("Consolas", 10, "bold")).pack(anchor=tk.W, padx=10)
        for cls in CLASSES:
            row = tk.Frame(parent, bg="#1c1c30"); row.pack(fill=tk.X, padx=10, pady=2)
            col = CLASS_HEX.get(cls, "#ffffff")
            tk.Label(row, text=f"  {cls:<16}", bg="#1c1c30", fg=col, font=("Consolas", 10)).pack(side=tk.LEFT)
            tk.Label(row, textvariable=self.count_vars[cls], bg="#1c1c30", fg="#ffffff", font=("Consolas", 10, "bold")).pack(side=tk.RIGHT)

    def _build_delete_tab(self, parent):
        pad = {"padx": 12, "pady": 6}

        tk.Label(parent, text="Target class:", bg="#1c1c30", fg="#9999bb", font=("Consolas", 10)).pack(anchor=tk.W, padx=12, pady=(10, 0))
        self.manage_class_var = tk.StringVar(value=CLASSES[0])
        ttk.Combobox(parent, textvariable=self.manage_class_var,
                     values=CLASSES, state="readonly",
                     font=("Consolas", 11), width=20).pack(fill=tk.X, padx=12, pady=(3,10))

        tk.Button(
            parent, text="🖼  VIEW RANDOM SAMPLE", command=self._view_random_sample,
            bg="#2a5a2a", fg="#aaffaa", font=("Consolas", 10, "bold"),
            relief=tk.FLAT, cursor="hand2", pady=9
        ).pack(fill=tk.X, padx=12, pady=10)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=10)

        dc_row = tk.Frame(parent, bg="#1c1c30"); dc_row.pack(fill=tk.X, padx=12)
        tk.Label(dc_row, text="Delete last N images:", bg="#1c1c30", fg="#9999bb", font=("Consolas", 9)).pack(side=tk.LEFT)
        self.del_count_var = tk.IntVar(value=100)
        tk.Spinbox(dc_row, from_=1, to=10000, increment=10,
                   textvariable=self.del_count_var, width=7, font=("Consolas", 10),
                   bg="#0a0a14", fg="#ffffff", buttonbackground="#400f0f").pack(side=tk.RIGHT)

        tk.Button(
            parent, text="🗑  DELETE LAST N", command=self._delete_last_n,
            bg="#5a0000", fg="#ffaaaa", font=("Consolas", 11, "bold"),
            relief=tk.FLAT, cursor="hand2", pady=11
        ).pack(fill=tk.X, padx=12, pady=12)

    def _apply_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TCombobox", fieldbackground="#0a0a14", background="#0f3460", foreground="#ffffff")
        s.configure("TProgressbar", troughcolor="#0a0a14", background="#00cc55", thickness=14)
        s.configure("TNotebook", background="#1c1c30", tabmargins=[2, 5, 2, 0])
        s.configure("TNotebook.Tab", background="#111128", foreground="#9999bb", padding=[10, 5], font=("Consolas", 10))
        s.map("TNotebook.Tab", background=[("selected", "#1c1c30")], foreground=[("selected", "#00e5ff")])

    def _set_controls_enabled(self, on: bool):
        st = tk.NORMAL if on else tk.DISABLED
        self.pause_btn.config(state=st)
        self.collect_btn.config(state=st)
        self.label_dd.config(state="readonly" if on else tk.DISABLED)

    # ─────────────────────────────────────────────────────
    # CAMERA LOOP  (background thread)
    # ─────────────────────────────────────────────────────
    def _camera_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            for _ in range(2):
                r, f = self.cap.read()
                if r:
                    frame = f

            # YOLO processing
            bboxes = []
            warped = None
            if self.M is not None:
                warped = cv2.warpPerspective(frame, self.M, (WARP_SIZE, WARP_SIZE))
                res = self.yolo_model.predict(warped, conf=0.15, verbose=False)
                if len(res) > 0 and len(res[0].boxes) > 0:
                    # Treat the highest confidence detection as the target
                    box = res[0].boxes[0]
                    x, y, w, h = box.xywh[0].cpu().numpy()
                    bboxes.append((int(x - w/2), int(y - h/2), int(w), int(h)))

            with self.lock:
                self.current_frame = frame.copy()
                self.current_warped = warped
                self.yolo_bboxes = bboxes

    # ─────────────────────────────────────────────────────
    # DISPLAY REFRESH  (main thread)
    # ─────────────────────────────────────────────────────
    def _update_display(self):
        with self.lock:
            frame  = self.current_frame.copy()  if self.current_frame  is not None else None
            warped = self.current_warped.copy() if self.current_warped is not None else None
            bboxes = list(self.yolo_bboxes)

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = self.CANVAS_W, self.CANVAS_H

        if frame is not None:
            disp = (self._draw_corner_overlay(frame)
                    if self.phase == "corner_select"
                    else self._draw_collection_overlay(frame, warped, bboxes))

            fh, fw = disp.shape[:2]
            scale  = min(cw / fw, ch / fh)
            nw     = int(fw * scale)
            nh     = int(fh * scale)

            self._disp_scale = scale
            self._disp_x0    = (cw - nw) // 2
            self._disp_y0    = (ch - nh) // 2

            disp = cv2.resize(disp, (nw, nh), interpolation=cv2.INTER_LINEAR)
            disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            img  = ImageTk.PhotoImage(Image.fromarray(disp))
            self.canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=img)
            self.canvas._img = img

        self.root.after(30, self._update_display)

    # ─────────────────────────────────────────────────────
    # OVERLAY DRAWINGS
    # ─────────────────────────────────────────────────────
    def _draw_corner_overlay(self, frame):
        out    = frame.copy()
        labels = self.CORNER_LABELS
        n      = len(self.corner_clicks)

        for i, (fx, fy) in enumerate(self.corner_clicks):
            cv2.circle(out, (fx, fy), 11, (0, 255, 0), -1)
            cv2.circle(out, (fx, fy), 13, (255, 255, 255), 2)
            cv2.putText(out, labels[i], (fx + 15, fy - 10), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 255), 2)
            if i > 0:
                cv2.line(out, self.corner_clicks[i-1], (fx, fy), (0, 255, 0), 2)

        if n >= 3:
            cv2.line(out, self.corner_clicks[-1], self.corner_clicks[0], (0, 200, 0), 1)

        if n < 4:
            cv2.putText(out, f"Click corner {n+1}/4  ({labels[n]})", (12, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 220, 255), 2)
        else:
            cv2.putText(out, "All 4 corners set!", (12, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 80), 2)
        return out

    def _draw_collection_overlay(self, frame, warped, bboxes):
        out    = frame.copy()
        fh, fw = out.shape[:2]
        cls    = self.label_var.get()
        lz     = self.lz_var.get()
        rz     = self.rz_var.get()

        if self.M_inv is not None:
            draw_zones_on_frame(out, self.M_inv, lz, rz)

        # Draw YOLO bounding boxes on the main frame
        for bx, by, bw, bh in bboxes:
            box_pts_w = np.array([[bx, by], [bx+bw, by], [bx+bw, by+bh], [bx, by+bh]], dtype="float32")
            box_pts_f = warp_pts_to_frame(box_pts_w, self.M_inv)
            cv2.polylines(out, [box_pts_f], True, (0, 60, 255), 3, cv2.LINE_AA)
            
            # Corner reticles
            tl = tuple(box_pts_f[0]); tr = tuple(box_pts_f[1])
            br = tuple(box_pts_f[2]); bl = tuple(box_pts_f[3])
            span = max(1.0, float(np.linalg.norm(np.array(tr) - np.array(tl))))
            t = min(18.0 / span, 0.35)
            def lerp(a, b, t): return (int(a[0] + (b[0]-a[0])*t), int(a[1] + (b[1]-a[1])*t))
            for corner, nb1, nb2 in [(tl, tr, bl), (tr, tl, br), (br, tr, bl), (bl, tl, br)]:
                cv2.line(out, corner, lerp(corner, nb1, t), (0, 60, 255), 4, cv2.LINE_AA)
                cv2.line(out, corner, lerp(corner, nb2, t), (0, 60, 255), 4, cv2.LINE_AA)

        col = tuple(int(CLASS_HEX[cls][i:i+2], 16) for i in (5, 3, 1))

        cv2.rectangle(out, (0, fh - 46), (fw, fh), (20, 20, 20), -1)
        state = "⏸ PAUSED" if self.paused else "▶ LIVE"
        cv2.putText(out, f"{state}  |  {cls.upper()}", (12, fh - 14), cv2.FONT_HERSHEY_DUPLEX, 0.72, col, 2)

        if self.collecting:
            done   = int(self.progress["value"])
            target = int(self.progress["maximum"])
            pct    = int(done / max(target, 1) * fw)
            cv2.rectangle(out, (0, 0), (pct, 8), col, -1)
            cv2.putText(out, f"COLLECTING  {done} / {target}", (12, 36), cv2.FONT_HERSHEY_DUPLEX, 0.9, col, 2)

        if self.paused and not self.collecting:
            ov = out.copy()
            cv2.rectangle(ov, (0, 0), (fw, 52), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.60, out, 0.40, 0, out)
            cv2.putText(out, "⏸  PAUSED", (fw//2 - 90, 36), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 200, 255), 2)

        if warped is not None:
            ins = 180
            inset = cv2.resize(warped, (ins, ins))
            draw_zones_on_warp(inset, int(lz * ins / WARP_SIZE), int(rz * ins / WARP_SIZE))
            
            # Draw YOLO boxes on the inset
            scale = ins / WARP_SIZE
            for bx, by, bw, bh in bboxes:
                ix, iy, iw, ih = int(bx*scale), int(by*scale), int(bw*scale), int(bh*scale)
                cv2.rectangle(inset, (ix, iy), (ix+iw, iy+ih), (0, 60, 255), 2)

            x0 = fw - ins - 8
            y0 = fh - ins - 52
            out[y0:y0+ins, x0:x0+ins] = inset
            cv2.rectangle(out, (x0, y0), (x0+ins, y0+ins), col, 2)
            cv2.putText(out, "BIRD'S-EYE", (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        return out

    # ─────────────────────────────────────────────────────
    # CORNER SELECTION
    # ─────────────────────────────────────────────────────
    def _on_canvas_click(self, event):
        if self.phase != "corner_select": return
        if len(self.corner_clicks) >= 4: return

        with self.lock: frame = self.current_frame
        if frame is None: return

        fh, fw = frame.shape[:2]
        fx = (event.x - self._disp_x0) / self._disp_scale
        fy = (event.y - self._disp_y0) / self._disp_scale
        fx = int(max(0, min(fx, fw - 1)))
        fy = int(max(0, min(fy, fh - 1)))

        self.corner_clicks.append((fx, fy))
        if len(self.corner_clicks) == 4:
            self._finalise_corners()

    def _finalise_corners(self):
        self.corners = order_points(self.corner_clicks)
        self.M, self.M_inv = build_transforms(self.corners)
        self.phase = "collecting"
        self.hint_lbl.config(text="STEP 2 — Select a label, set zone margins, then press  COLLECT", fg="#00ff88")
        self._set_controls_enabled(True)

    def _reset_corners(self):
        self.corner_clicks = []
        self.corners = self.M = self.M_inv = None
        with self.lock:
            self.current_warped = None
            self.yolo_bboxes = []
        self.phase = "corner_select"
        self.hint_lbl.config(text="STEP 1 — Click the 4 corners of the bed surface  (TL → TR → BR → BL)", fg="#00e5ff")
        self._set_controls_enabled(False)

    # ─────────────────────────────────────────────────────
    # UTILITY BUTTONS
    # ─────────────────────────────────────────────────────
    def _toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text="▶  PLAY" if self.paused else "⏸  PAUSE", bg="#1a4a1a" if self.paused else "#0f3460")

    def _start_collect(self):
        if self.collecting: return
        target = max(1, self.samples_var.get())
        self.progress["maximum"] = target
        self.progress["value"] = 0
        self.prog_lbl.config(text=f"0 / {target}")
        self.collecting = True
        self.collect_btn.config(state=tk.DISABLED, bg="#003300")
        threading.Thread(target=self._collect_worker, args=(self.label_var.get(), target), daemon=True).start()

    def _collect_worker(self, cls: str, target: int):
        save_dir = os.path.join(DATASET_DIR, cls)
        ts_base  = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved    = 0
        while saved < target:
            if not self.paused:
                with self.lock:
                    w = self.current_warped.copy() if self.current_warped is not None else None
                if w is not None:
                    fname = f"{cls}_{ts_base}_{saved:06d}.jpg"
                    cv2.imwrite(os.path.join(save_dir, fname), w, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    self.meta_writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        fname, cls, self.session_id
                    ])
                    self.meta_fh.flush()
                    saved += 1
                    self.root.after(0, self._tick_progress, saved, target, cls)
            time.sleep(0.05)

        self.collecting = False
        self.root.after(0, self._on_collect_done, cls, saved)

    def _tick_progress(self, done, target, cls):
        self.progress["value"] = done
        self.prog_lbl.config(text=f"{done} / {target}")
        self.count_vars[cls].set(str(self._count_existing(cls)))

    def _on_collect_done(self, cls, saved):
        self.collect_btn.config(state=tk.NORMAL, bg="#005500")
        total = self._count_existing(cls)
        self.count_vars[cls].set(str(total))
        self.prog_lbl.config(text=f"Done! {saved} saved.")
        messagebox.showinfo("Complete", f"Saved {saved} images.\nTotal in '{cls}': {total}")

    def _on_label_change(self, *_):
        col = CLASS_HEX.get(self.label_var.get(), "#ffffff")
        self.hint_lbl.config(fg=col)

    def _view_random_sample(self):
        cls = self.manage_class_var.get()
        d = os.path.join(DATASET_DIR, cls)
        if not os.path.isdir(d):
            messagebox.showinfo("Empty", f"Directory for '{cls}' does not exist.")
            return
        files = [f for f in os.listdir(d) if f.lower().endswith(".jpg")]
        if not files:
            messagebox.showinfo("Empty", f"No images found for class '{cls}'.")
            return
        f = random.choice(files)
        img = cv2.imread(os.path.join(d, f))
        if img is not None:
            # Show the random sample in a separate OpenCV window
            cv2.imshow(f"Random Sample: {cls} ({f})", img)
            # OpenCV waitKey is needed to process window events
            cv2.waitKey(1)

    def _delete_last_n(self):
        cls = self.manage_class_var.get()
        n   = self.del_count_var.get()
        save_dir = os.path.join(DATASET_DIR, cls)
        if not os.path.isdir(save_dir): return
        files = sorted([f for f in os.listdir(save_dir) if f.lower().endswith(".jpg")])
        total = len(files)
        if total == 0:
            messagebox.showinfo("Empty", f"No images in '{cls}'.")
            return
        to_delete = files[-min(n, total):]
        if not messagebox.askyesno("Confirm Delete", f"Delete last {len(to_delete)} images from '{cls}'?"):
            return
        for f in to_delete:
            os.remove(os.path.join(save_dir, f))
        rem = self._count_existing(cls)
        self.count_vars[cls].set(str(rem))
        messagebox.showinfo("Deleted", f"Deleted {len(to_delete)} images.\nRemaining: {rem}")

    def _on_quit(self):
        self.running = False
        self.meta_fh.close()
        self.cap.release()
        self.root.destroy()

# ─────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    DataCollector(root)
    root.mainloop()

if __name__ == "__main__":
    main()
