"""
Shared Stage A (segmentation) + Stage B (feature extraction) utilities.

Everything downstream of extract_features() operates only on the small
feature dict it returns -- never on raw pixels -- which is what makes the
rest of the pipeline background-invariant. See IMPLEMENTATION_PLAN.md, Stage B.
"""

import numpy as np
from PIL import Image
from scipy import ndimage

# --------------------------------------------------------------------------
# Config -- tune to your camera / bed setup
# --------------------------------------------------------------------------
MORPH_CLOSE_SIZE = 5
MORPH_OPEN_SIZE = 3
MIN_BLOB_AREA = 500          # px^2, discard components smaller than this
MAX_PERSON_AREA_FRAC = 0.6   # sanity bound: a single person shouldn't fill >60% of frame
                              # (helps flag "largest blob is NOT the person", e.g. a
                              # heating pad or lighting artifact -- see plan Sec.5)

LEFT_ZONE_FRAC = 0.30
RIGHT_ZONE_FRAC = 0.30


def load_grayscale(path):
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)[:, :, 0].astype(np.float64)
    return arr


def otsu_threshold(gray):
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 255))
    hist = hist.astype(np.float64)
    total = hist.sum()
    sum_all = np.dot(np.arange(256), hist)
    sum_bg, w_bg, max_var, thresh = 0.0, 0.0, 0.0, 0
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_all - sum_bg) / w_fg
        var_between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            thresh = t
    return thresh


def segment_person(gray, noise_jitter_std=0.0, rng=None):
    """
    Stage A: returns a clean boolean mask of the person.
    noise_jitter_std: optional Gaussian noise added before thresholding, used
      during TRAINING as segmentation-robustness augmentation (see plan Stage C).
      Leave at 0.0 for real inference.
    """
    g = gray
    if noise_jitter_std > 0:
        rng = rng or np.random.default_rng()
        g = gray + rng.normal(0, noise_jitter_std, size=gray.shape)

    t = otsu_threshold(g)
    mask = g > t

    mask = ndimage.binary_closing(mask, structure=np.ones((MORPH_CLOSE_SIZE, MORPH_CLOSE_SIZE)))
    mask = ndimage.binary_opening(mask, structure=np.ones((MORPH_OPEN_SIZE, MORPH_OPEN_SIZE)))

    labeled, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask), t, False

    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    largest_label = np.argmax(sizes) + 1
    largest_area = sizes[largest_label - 1]

    if largest_area < MIN_BLOB_AREA:
        return np.zeros_like(mask), t, False

    area_frac = largest_area / mask.size
    plausible = area_frac <= MAX_PERSON_AREA_FRAC  # False => likely NOT the person (see Sec.5)

    person_mask = labeled == largest_label
    return person_mask, t, plausible


def extract_features(mask, img_shape):
    """Stage B: reduce a person mask to a small, background-invariant feature dict."""
    h, w = img_shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    width, height = x_max - x_min, y_max - y_min
    centroid_x, centroid_y = xs.mean(), ys.mean()

    cx, cy = centroid_x, centroid_y
    xs_c, ys_c = xs - cx, ys - cy
    cov = np.cov(np.stack([xs_c, ys_c]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, np.argmax(eigvals)]
    angle_deg = np.degrees(np.arctan2(major_axis[1], major_axis[0])) % 180

    return {
        "centroid_x_frac": float(centroid_x / w),
        "centroid_y_frac": float(centroid_y / h),
        "bbox": (int(x_min), int(y_min), int(x_max), int(y_max)),
        "aspect_ratio": float(width / max(height, 1)),
        "area_frac": float(len(xs) / (h * w)),
        "orientation_angle_deg": float(angle_deg),
    }


def classify_zone(centroid_x_frac):
    if centroid_x_frac < LEFT_ZONE_FRAC:
        return "LEFT"
    elif centroid_x_frac > (1 - RIGHT_ZONE_FRAC):
        return "RIGHT"
    return "CENTER"


def feature_vector(features):
    """Fixed-order numeric vector for ML models -- pixels never appear here."""
    return np.array([
        features["centroid_x_frac"],
        features["centroid_y_frac"],
        features["aspect_ratio"],
        features["area_frac"],
        features["orientation_angle_deg"] / 180.0,  # normalize to [0,1]
    ])


def process_image(path, noise_jitter_std=0.0, rng=None):
    """Convenience: load -> segment -> extract features, in one call."""
    gray = load_grayscale(path)
    mask, thresh, plausible = segment_person(gray, noise_jitter_std, rng)
    feats = extract_features(mask, gray.shape)
    return feats, mask, thresh, plausible
