#!/usr/bin/env python3
"""
fl_client.py — Federated Learning client (one per bed)
======================================================

Runs beside img_process.py. Trains a risk classifier on the positions this
bed actually recorded, uploads only the model, then pulls the aggregated
global model back and installs it as the live brain.

The position log never leaves the machine — that is the point of doing this
federated: every bed contributes what it learned without sharing footage or
patient positions.

Usage
    python fl_client.py                  # one round: train, upload, pull
    python fl_client.py --loop 600       # repeat every 10 minutes
    python fl_client.py --pull-only      # just fetch the latest global model
    python fl_client.py --server http://192.168.1.100:8081
"""

import argparse
import os
import pickle
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "position_log.csv")
LIVE_MODEL = os.path.join(BASE_DIR, "backend", "models", "risk_classifier.pkl")
LOCAL_PKL  = os.path.join(BASE_DIR, "fl_store", "local_model.pkl")

DEFAULT_SERVER = os.environ.get("FL_SERVER", "http://127.0.0.1:8081")
CLIENT_ID      = os.environ.get("FL_CLIENT_ID", f"bed-{os.getpid()}")

# Feature order must match backend/ml_trainer.py and img_process.classify_state.
FEATURES = [
    "warp_x", "warp_y", "aspect_ratio",
    "norm_x", "norm_y",
    "left_gap", "right_gap",
    "in_left", "in_right",
    "vx", "vy", "speed", "ax",
]

MIN_ROWS = 50   # below this, a local model is noise rather than knowledge


# ─────────────────────────────────────────────────────────
# Local data -> features
# ─────────────────────────────────────────────────────────
def build_features(df, left_z=150.0, right_z=490.0):
    """Recreate ml_trainer's feature engineering from the raw position log."""
    df = df.copy()

    # img_process.py logs the centroid as warp_cx/warp_cy; the trained model
    # names those features warp_x/warp_y.
    if "warp_x" not in df.columns and "warp_cx" in df.columns:
        df["warp_x"] = df["warp_cx"]
    if "warp_y" not in df.columns and "warp_cy" in df.columns:
        df["warp_y"] = df["warp_cy"]

    if "left_z" not in df.columns:
        df["left_z"] = left_z
    if "right_z" not in df.columns:
        df["right_z"] = right_z

    safe_width = (df["right_z"] - df["left_z"]).clip(lower=1)
    df["norm_x"] = (df["warp_x"] - df["left_z"]) / safe_width
    df["norm_y"] = df["warp_y"] / 400.0
    df["left_gap"]  = df["warp_x"] - df["left_z"]
    df["right_gap"] = df["right_z"] - df["warp_x"]
    df["in_left"]   = (df["warp_x"] < df["left_z"]).astype(int)
    df["in_right"]  = (df["warp_x"] > df["right_z"]).astype(int)

    if "aspect_ratio" not in df.columns:
        df["aspect_ratio"] = 1.0

    df["vx"] = df["warp_x"].diff().fillna(0.0)
    df["vy"] = df["warp_y"].diff().fillna(0.0)
    df["speed"] = np.hypot(df["vx"], df["vy"])
    df["ax"] = df["vx"].diff().fillna(0.0)
    return df


def label_rows(df):
    """Derive the risk label from the zone img_process.py recorded."""
    zone = df.get("zone", pd.Series(["SAFE"] * len(df))).astype(str).str.upper()
    risk = pd.Series("STABLE", index=df.index)
    risk[zone.str.contains("WARNING")] = "DRIFT WARNING"
    risk[zone.str.contains("DANGER") | zone.str.contains("NOT_FOUND")] = "FALL IMMINENT"
    return risk


def load_local_dataset():
    if not os.path.isfile(CSV_PATH):
        print(f"  No local log at {CSV_PATH} — run img_process.py first.")
        return None, None

    df = pd.read_csv(CSV_PATH)
    if len(df) < MIN_ROWS:
        print(f"  Only {len(df)} rows logged; need >= {MIN_ROWS} to train.")
        return None, None

    df = build_features(df)
    df["risk"] = label_rows(df)
    df = df.dropna(subset=FEATURES + ["risk"])

    X = df[FEATURES].to_numpy(dtype=float)
    y = df["risk"].to_numpy()
    return X, y


# ─────────────────────────────────────────────────────────
# Local training
# ─────────────────────────────────────────────────────────
def train_local():
    X, y = load_local_dataset()
    if X is None:
        return None

    classes = sorted(set(y))
    if len(classes) < 2:
        print(f"  Local data only contains '{classes[0]}' — nothing to learn yet.")
        return None

    le = LabelEncoder().fit(["DRIFT WARNING", "FALL IMMINENT", "STABLE"])
    y_enc = le.transform(y)

    stratify = y_enc if min(np.bincount(y_enc)[np.bincount(y_enc) > 0]) >= 2 else None
    if len(X) >= 100:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=stratify)
    else:
        X_tr, X_te, y_tr, y_te = X, X, y_enc, y_enc

    # Every client must fit over the FULL label space, even when this bed has
    # never seen a fall. Otherwise its trees emit narrower probability vectors
    # and cannot be merged with trees from a bed that saw all three classes.
    missing = sorted(set(range(len(le.classes_))) - set(np.unique(y_tr)))
    if missing:
        absent = [le.classes_[m] for m in missing]
        print(f"  Padding unseen classes so trees stay mergeable: {absent}")
        # One synthetic row per absent class, placed at that class's typical
        # position, is enough to fix the tree's output width without
        # meaningfully shifting what the model learned from real data.
        pads_X, pads_y = [], []
        for m in missing:
            proto = X_tr.mean(axis=0).copy()
            pads_X.append(proto)
            pads_y.append(m)
        X_tr = np.vstack([X_tr, np.array(pads_X)])
        y_tr = np.concatenate([y_tr, np.array(pads_y)])

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    acc = float(clf.score(X_te, y_te)) * 100.0

    print(f"  Trained on {len(X_tr)} rows — local accuracy {acc:.2f}%")
    print(f"  Classes present: {classes}")

    bundle = {
        "clf": clf,
        "le": le,
        "features": FEATURES,
        "client_id": CLIENT_ID,
        "n_samples": int(len(X)),
        "accuracy": acc,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }

    os.makedirs(os.path.dirname(LOCAL_PKL), exist_ok=True)
    with open(LOCAL_PKL, "wb") as f:
        pickle.dump(bundle, f)
    return bundle


# ─────────────────────────────────────────────────────────
# Server exchange
# ─────────────────────────────────────────────────────────
def upload(bundle, server):
    payload = pickle.dumps(bundle)
    print(f"  Uploading {len(payload)/1024:.0f} KB to {server}/upload …")
    r = requests.post(f"{server}/upload", data=payload,
                      headers={"Content-Type": "application/octet-stream"},
                      timeout=30)
    r.raise_for_status()
    info = r.json()
    if info.get("aggregated"):
        print(f"  Server aggregated round {info['round']}.")
    else:
        print(f"  Queued — {info.get('pending')} of {info.get('needed')} clients in.")
    return info


def pull_global(server, install=True):
    print(f"  Fetching global model from {server}/global …")
    r = requests.get(f"{server}/global", timeout=30)
    if r.status_code == 404:
        print("  No global model published yet.")
        return None
    r.raise_for_status()

    bundle = pickle.loads(r.content)
    trees = getattr(bundle.get("clf"), "n_estimators", "?")
    print(f"  Global model: round {bundle.get('round', '?')}, {trees} trees, "
          f"contributors={bundle.get('contributors', [])}")

    if install:
        os.makedirs(os.path.dirname(LIVE_MODEL), exist_ok=True)
        if os.path.isfile(LIVE_MODEL):
            backup = LIVE_MODEL + ".bak"
            shutil.copy2(LIVE_MODEL, backup)
            print(f"  Previous brain backed up -> {os.path.basename(backup)}")
        tmp = LIVE_MODEL + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(bundle, f)
        os.replace(tmp, LIVE_MODEL)
        print(f"  Installed as the live brain: {LIVE_MODEL}")
        print("  Restart img_process.py to pick it up.")
    return bundle


def run_round(server, pull_only=False):
    print("-" * 58)
    print(f"  Client {CLIENT_ID} @ {datetime.now().strftime('%H:%M:%S')}")
    try:
        if not pull_only:
            bundle = train_local()
            if bundle is not None:
                upload(bundle, server)
            else:
                print("  Skipping upload (no usable local model).")
        pull_global(server)
    except requests.exceptions.RequestException as exc:
        print(f"  Server unreachable: {exc}")
        return False
    except Exception as exc:
        print(f"  Round failed: {exc}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description="Federated learning client")
    ap.add_argument("--server", default=DEFAULT_SERVER, help="FL server base URL")
    ap.add_argument("--loop", type=int, metavar="SECONDS",
                    help="repeat every N seconds instead of running once")
    ap.add_argument("--pull-only", action="store_true",
                    help="only download the global model")
    args = ap.parse_args()

    server = args.server.rstrip("/")
    print("=" * 58)
    print("  Sleeping Monitor — Federated Learning Client")
    print("=" * 58)
    print(f"  Client ID : {CLIENT_ID}")
    print(f"  Server    : {server}")
    print(f"  Local log : {CSV_PATH}")

    if not args.loop:
        run_round(server, args.pull_only)
        return

    print(f"  Looping every {args.loop}s — Ctrl+C to stop.")
    try:
        while True:
            run_round(server, args.pull_only)
            time.sleep(args.loop)
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    main()
