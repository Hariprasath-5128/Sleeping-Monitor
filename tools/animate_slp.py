import os
import json
import glob

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter

# ============================================================================
# Paths (as given)
# ============================================================================
input_dir = r"C:\Projects\sleeping-monitor\dataset_IR\train\train\00001\IR\uncover"
output_dir = r"C:\Projects\sleeping-monitor\preview_animation\00001\IR\uncover"
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# Motion narrative -- edit this to change the story, not the per-frame logic
# Each keyframe: (frame_index_1_based, position_fraction)
#   position_fraction: 0.0 = LEFT zone, 0.5 = SAFE/center, 1.0 = RIGHT zone
# Frames between keyframes are smoothstep-interpolated (eased, not linear-jerky).
# ============================================================================
MOTION_KEYFRAMES = [
    (1,  0.5),   # SAFE dwell start
    (7,  0.5),   # SAFE dwell end
    (15, 0.0),   # transition complete -> LEFT
    (21, 0.0),   # LEFT dwell end
    (28, 0.5),   # transition complete -> back to SAFE
    (31, 0.5),   # SAFE dwell end
    (39, 1.0),   # transition complete -> RIGHT
    (45, 1.0),   # RIGHT dwell end (final frame)
]

# ============================================================================
# Outpainting config -- adaptive to the small (120x160) SLP frame size
# ============================================================================
WIDTH_SCALE = 2.2            # target canvas width = original width * this
MARGIN_SMALL_FRAC = 0.12     # danger-zone margin, as a fraction of original width
FEATHER_FRAC = 0.35          # seam feather width, as a fraction of PATCH_SIZE-derived value
LOW_FREQ_SIGMA = 6           # smaller than the earlier 565px example -- scaled to image size
PATCH_SIZE = 10              # background texture patch size (px) -- small because margins are thin
PATCH_OVERLAP = 4
PERSON_MARGIN_COL_FRAC = 0.18  # fraction of width assumed to be pure background at each side
BAND_COUNT = 4
OUTLIER_STD_MULT = 2.5
TEXTURE_BOOST = 1.15
BG_PERCENTILE = 30


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return 3 * t ** 2 - 2 * t ** 3


def build_trajectory(n_frames, keyframes):
    """Interpolate position_fraction for every frame 1..n_frames via smoothstep easing."""
    kf_idx = np.array([k[0] for k in keyframes])
    kf_val = np.array([k[1] for k in keyframes])
    traj = np.zeros(n_frames)
    for f in range(1, n_frames + 1):
        if f <= kf_idx[0]:
            traj[f - 1] = kf_val[0]
            continue
        if f >= kf_idx[-1]:
            traj[f - 1] = kf_val[-1]
            continue
        i = np.searchsorted(kf_idx, f) - 1
        f0, f1 = kf_idx[i], kf_idx[i + 1]
        v0, v1 = kf_val[i], kf_val[i + 1]
        t = (f - f0) / (f1 - f0) if f1 != f0 else 0.0
        traj[f - 1] = v0 + (v1 - v0) * smoothstep(t)
    return traj


def zone_label(p):
    if p < 0.15:
        return "LEFT"
    if p > 0.85:
        return "RIGHT"
    return "SAFE"


# ----------------------------------------------------------------------------
# Background outpainting core (patch-quilting method from earlier in this chat,
# parameters adapted for small frame size)
# ----------------------------------------------------------------------------

def load_grayscale(path):
    img = Image.open(path).convert("L")
    return np.array(img).astype(np.float64), img.size  # (arr, (w,h))


def estimate_bg_stats(gray, low_freq_sigma):
    low_freq = gaussian_filter(gray, sigma=low_freq_sigma)
    thresh = np.percentile(gray, BG_PERCENTILE)
    bg_mask = gray <= thresh
    bg_level = low_freq[bg_mask].mean() if bg_mask.any() else float(gray.mean())
    return low_freq, bg_level


def build_patch_library(gray, low_freq, person_margin_col, patch_size, patch_overlap, band_count):
    h, w = gray.shape
    residual = gray - low_freq
    band_edges = np.linspace(0, h, band_count + 1).astype(int)

    left_src = residual[:, :person_margin_col]
    right_src = residual[:, w - person_margin_col:]

    library = {b: [] for b in range(band_count)}
    step = max(1, patch_size - patch_overlap)

    for src in (left_src, right_src):
        sh, sw = src.shape
        if sw < patch_size:
            continue
        for y0 in range(0, max(1, sh - patch_size), step):
            for x0 in range(0, max(1, sw - patch_size), step):
                patch = src[y0:y0 + patch_size, x0:x0 + patch_size]
                if patch.shape != (patch_size, patch_size):
                    continue
                band = min(band_count - 1, max(0, np.searchsorted(band_edges, y0, side="right") - 1))
                library[band].append(patch)

    for band, patches in library.items():
        if len(patches) < 4:
            continue
        stds = np.array([p.std() for p in patches])
        med = np.median(stds)
        mad = np.median(np.abs(stds - med)) + 1e-6
        keep = [p for p, s in zip(patches, stds) if abs(s - med) <= OUTLIER_STD_MULT * mad]
        library[band] = keep if len(keep) >= 4 else patches

    # Fallback: if margins were too thin to harvest any patches, synthesize a
    # tiny library from the overall background noise stats so the pipeline
    # never crashes on very tightly-cropped frames.
    total = sum(len(v) for v in library.values())
    if total == 0:
        bg_std = residual[gray <= np.percentile(gray, BG_PERCENTILE)].std()
        bg_std = bg_std if bg_std > 0 else 2.0
        rng = np.random.default_rng(0)
        for b in range(band_count):
            library[b] = [rng.normal(0, bg_std, size=(patch_size, patch_size)) for _ in range(8)]

    return library


def _cosine_window(size):
    return 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, size))


def quilt_residual_field(height, width, library, band_count, patch_size, patch_overlap, rng):
    field = np.zeros((height, width), dtype=np.float64)
    weight = np.zeros((height, width), dtype=np.float64)
    step = max(1, patch_size - patch_overlap)
    win2d = np.outer(_cosine_window(patch_size), _cosine_window(patch_size))
    band_edges = np.linspace(0, height, band_count + 1).astype(int)

    y0 = 0
    while y0 < height:
        ph = min(patch_size, height - y0)
        band = min(band_count - 1, max(0, np.searchsorted(band_edges, y0, side="right") - 1))
        patches = library.get(band) or [p for lst in library.values() for p in lst]

        x0 = 0
        while x0 < width:
            pw = min(patch_size, width - x0)
            patch = patches[rng.integers(0, len(patches))]
            if rng.random() < 0.5:
                patch = patch[:, ::-1]
            if rng.random() < 0.5:
                patch = patch[::-1, :]
            p = patch[:ph, :pw]
            wgt = win2d[:ph, :pw]
            field[y0:y0 + ph, x0:x0 + pw] += p * wgt
            weight[y0:y0 + ph, x0:x0 + pw] += wgt
            x0 += step
        y0 += step

    weight[weight == 0] = 1.0
    return (field / weight) * TEXTURE_BOOST


def synth_strip(height, width, edge_low_freq_col, bg_level, library, patch_size, patch_overlap, band_count, rng):
    x = np.linspace(0, 1, width)
    decay = np.exp(-x * 2.2)[None, :]
    edge_col = edge_low_freq_col[:, None]
    low_freq_strip = bg_level + (edge_col - bg_level) * decay
    texture = quilt_residual_field(height, width, library, band_count, patch_size, patch_overlap, rng)
    return np.clip(low_freq_strip + texture, 0, 255)


def feather_seam(canvas, x0, x1, feather_px):
    h, w = canvas.shape
    smoothed = uniform_filter(canvas, size=(1, 5))
    if x0 > 0:
        rs = max(0, x0 - feather_px)
        ramp = np.linspace(0, 1, x0 - rs)[None, :]
        canvas[:, rs:x0] = smoothed[:, rs:x0] * (1 - ramp) + canvas[:, rs:x0] * ramp
    if x1 < w:
        re = min(w, x1 + feather_px)
        ramp = np.linspace(1, 0, re - x1)[None, :]
        canvas[:, x1:re] = smoothed[:, x1:re] * (1 - ramp) + canvas[:, x1:re] * ramp
    return canvas


def outpaint_frame(gray, left_margin, right_margin, library, low_freq, bg_level,
                    patch_size, patch_overlap, band_count, feather_px, rng):
    h, w = gray.shape
    total_w = left_margin + w + right_margin
    canvas = np.zeros((h, total_w), dtype=np.float64)

    if left_margin > 0:
        left_strip = synth_strip(h, left_margin, low_freq[:, 0], bg_level, library,
                                  patch_size, patch_overlap, band_count, rng)[:, ::-1]
        canvas[:, :left_margin] = left_strip
    if right_margin > 0:
        right_strip = synth_strip(h, right_margin, low_freq[:, -1], bg_level, library,
                                   patch_size, patch_overlap, band_count, rng)
        canvas[:, left_margin + w:] = right_strip

    canvas[:, left_margin:left_margin + w] = gray
    canvas = feather_seam(canvas, left_margin, left_margin + w, feather_px)
    canvas = feather_seam(canvas, left_margin, left_margin + w, feather_px)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def main():
    paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No PNG frames found in {input_dir}")
    n_frames = len(paths)
    print(f"Found {n_frames} frames in {input_dir}")

    trajectory = build_trajectory(n_frames, MOTION_KEYFRAMES)

    # Use the first frame to size the canvas / derive scaled parameters
    first_gray, (orig_w, orig_h) = load_grayscale(paths[0])
    target_w = int(orig_w * WIDTH_SCALE)
    extra = target_w - orig_w
    margin_small = max(4, int(orig_w * MARGIN_SMALL_FRAC))
    person_margin_col = max(PATCH_SIZE + PATCH_OVERLAP, int(orig_w * PERSON_MARGIN_COL_FRAC))
    feather_px = max(4, int(patch_size_feather := PATCH_SIZE * 1.5))

    rng = np.random.default_rng(42)
    log = []

    for i, path in enumerate(paths):
        gray, (w, h) = load_grayscale(path)
        if (w, h) != (orig_w, orig_h):
            print(f"warning: {path} size {w}x{h} differs from first frame {orig_w}x{orig_h}, skipping")
            continue

        p = float(trajectory[i])
        zone = zone_label(p)
        left_margin = int(round(margin_small + p * (extra - 2 * margin_small)))
        left_margin = max(0, min(extra, left_margin))
        right_margin = extra - left_margin

        low_freq, bg_level = estimate_bg_stats(gray, LOW_FREQ_SIGMA)
        library = build_patch_library(gray, low_freq, person_margin_col, PATCH_SIZE,
                                       PATCH_OVERLAP, BAND_COUNT)

        canvas = outpaint_frame(gray, left_margin, right_margin, library, low_freq, bg_level,
                                 PATCH_SIZE, PATCH_OVERLAP, BAND_COUNT, feather_px, rng)

        fname = os.path.basename(path)
        out_path = os.path.join(output_dir, fname)
        Image.fromarray(canvas, mode="L").save(out_path)

        log.append({
            "frame": i + 1,
            "filename": fname,
            "position_fraction": p,
            "zone": zone,
            "left_margin": left_margin,
            "right_margin": right_margin,
        })
        print(f"[{i+1:03d}/{n_frames}] {fname} -> zone={zone:5s} "
              f"pos_frac={p:.2f} left_margin={left_margin} right_margin={right_margin}")

    with open(os.path.join(output_dir, "motion_log.json"), "w") as f:
        json.dump({
            "input_dir": input_dir,
            "output_dir": output_dir,
            "canvas_width": target_w,
            "canvas_height": orig_h,
            "original_width": orig_w,
            "motion_keyframes": MOTION_KEYFRAMES,
            "frames": log,
        }, f, indent=2)

    print(f"\nDone. {len(log)} frames written to {output_dir}")
    print(f"Motion log: {os.path.join(output_dir, 'motion_log.json')}")


if __name__ == "__main__":
    main()
