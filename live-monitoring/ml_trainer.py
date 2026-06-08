"""
ml_trainer.py
=============
Trains a Random Forest classifier on the logged position data and exports:
  • models/risk_classifier.pkl  — the serialised model
  • models/model_meta.json      — class labels, feature names, last-trained ts
  • reports/training_report.json — per-class metrics, confusion matrix, accuracy
  • reports/live_predictions.json — rolling predictions for dashboard

Run:
    python ml_trainer.py              (one-shot training)
    python ml_trainer.py --watch      (retrain every 30 s if CSV grows)
"""

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CSV_PATH     = os.path.join(BASE_DIR, "position_log.csv")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
REPORTS_DIR  = os.path.join(BASE_DIR, "reports")
MODEL_FILE   = os.path.join(MODELS_DIR, "risk_classifier.pkl")
META_FILE    = os.path.join(MODELS_DIR, "model_meta.json")
REPORT_FILE  = os.path.join(REPORTS_DIR, "training_report.json")
LIVE_FILE    = os.path.join(REPORTS_DIR, "live_predictions.json")

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ─── Feature Engineering ─────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a rich feature set from the raw CSV columns.
    Key additions:
      • normalised_x  — position as fraction of [left_z, right_z] range
      • left_gap / right_gap — distance to each danger boundary (px, warp-space)
      • aspect_ratio kept as is (whitener tilt)
      • rolling velocity via diff on warp_x and warp_y
      • in_left_danger / in_right_danger one-hot flags
    """
    df = df.copy()

    # Guard against zero-width safe zone
    safe_width = (df["right_z"] - df["left_z"]).clip(lower=1)
    df["norm_x"]      = (df["warp_x"] - df["left_z"]) / safe_width
    df["norm_y"]      = df["warp_y"] / 400.0          # WARP_SIZE = 400
    df["left_gap"]    = (df["warp_x"] - df["left_z"]).clip(lower=-200)
    df["right_gap"]   = (df["right_z"] - df["warp_x"]).clip(lower=-200)
    df["in_left"]     = (df["warp_x"] < df["left_z"]).astype(int)
    df["in_right"]    = (df["warp_x"] > df["right_z"]).astype(int)

    # Rolling velocity (finite differences) — handles detection misses
    df["vx"] = df["warp_x"].diff().fillna(0).clip(-50, 50)
    df["vy"] = df["warp_y"].diff().fillna(0).clip(-50, 50)
    df["speed"] = np.sqrt(df["vx"]**2 + df["vy"]**2)

    # Acceleration
    df["ax"] = df["vx"].diff().fillna(0)

    return df


FEATURES = [
    "warp_x", "warp_y", "aspect_ratio",
    "norm_x", "norm_y",
    "left_gap", "right_gap",
    "in_left", "in_right",
    "vx", "vy", "speed", "ax",
]

# ─── Training ─────────────────────────────────────────────────────────────────
def train(verbose=True):
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV not found: {CSV_PATH}", file=sys.stderr)
        return None

    df = pd.read_csv(CSV_PATH)
    if len(df) < 50:
        print("[WARN] Too few rows to train meaningfully. Need >= 50.")
        return None

    df = build_features(df)
    df = df.dropna(subset=FEATURES + ["risk"])

    # Encode target labels
    le = LabelEncoder()
    y  = le.fit_transform(df["risk"])
    X  = df[FEATURES].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Model ──────────────────────────────────────────────────────────────
    clf = RandomForestClassifier(
        n_estimators   = 200,
        max_depth       = 12,
        min_samples_leaf = 2,
        class_weight    = "balanced",   # handles class imbalance
        random_state    = 42,
        n_jobs          = -1,
    )
    clf.fit(X_train, y_train)

    # ── Evaluation ─────────────────────────────────────────────────────────
    y_pred  = clf.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    cm      = confusion_matrix(y_test, y_pred).tolist()
    report  = classification_report(
        y_test, y_pred,
        target_names=list(le.classes_),
        output_dict=True
    )

    if verbose:
        print(f"\n{'='*52}")
        print(f"  Model trained on {len(X_train)} samples")
        print(f"  Test accuracy:   {acc*100:.1f}%")
        print(f"  Classes:         {list(le.classes_)}")
        print(f"{'='*52}")
        for cls, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"  {cls:20s}  precision={metrics['precision']:.2f}  "
                      f"recall={metrics['recall']:.2f}  f1={metrics['f1-score']:.2f}")

    # ── Save model ─────────────────────────────────────────────────────────
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"clf": clf, "le": le, "features": FEATURES}, f)

    meta = {
        "trained_at":   datetime.now().isoformat(),
        "n_samples":    int(len(X_train)),
        "n_test":       int(len(X_test)),
        "accuracy":     round(acc * 100, 2),
        "classes":      list(le.classes_),
        "features":     FEATURES,
        "model_type":   "RandomForestClassifier",
        "n_estimators": 200,
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    report_data = {
        "accuracy":          round(acc * 100, 2),
        "trained_at":        meta["trained_at"],
        "n_train":           int(len(X_train)),
        "n_test":            int(len(X_test)),
        "confusion_matrix":  cm,
        "class_labels":      list(le.classes_),
        "per_class_metrics": {
            k: {kk: round(vv, 4) for kk, vv in v.items()}
            for k, v in report.items()
            if isinstance(v, dict)
        },
    }
    with open(REPORT_FILE, "w") as f:
        json.dump(report_data, f, indent=2)

    if verbose:
        print(f"\n  Model saved -> {MODEL_FILE}")
        print(f"  Report saved -> {REPORT_FILE}")

    # ── Generate live predictions on the last 200 rows ──────────────────
    generate_live_predictions(df.tail(200), clf, le, acc)

    return clf, le, acc, report_data


# ─── Live prediction feed ─────────────────────────────────────────────────────
def generate_live_predictions(df, clf, le, accuracy):
    """
    Write a JSON file that the web dashboard polls every 3 seconds.
    Contains the last 200 predictions, feature importances, and summary stats.
    """
    X = df[FEATURES].values
    proba = clf.predict_proba(X)           # (N, n_classes)
    preds = clf.predict(X)
    labels = le.inverse_transform(preds)
    classes = list(le.classes_)

    records = []
    for i, row in enumerate(df.itertuples(index=False)):
        rec = {
            "timestamp":   getattr(row, "timestamp", ""),
            "warp_x":      round(float(row.warp_x), 1),
            "warp_y":      round(float(row.warp_y), 1),
            "risk_actual": getattr(row, "risk", ""),
            "risk_pred":   labels[i],
            "confidence":  round(float(proba[i].max()) * 100, 1),
            "proba": {
                cls: round(float(proba[i][j]) * 100, 1)
                for j, cls in enumerate(classes)
            },
        }
        records.append(rec)

    # Feature importances
    fi = dict(zip(FEATURES, clf.feature_importances_.tolist()))
    fi = {k: round(v, 4) for k, v in sorted(fi.items(), key=lambda x: -x[1])}

    # Summary stats
    unique, counts = np.unique(labels, return_counts=True)
    dist = {k: int(v) for k, v in zip(unique, counts)}

    live = {
        "generated_at":       datetime.now().isoformat(),
        "model_accuracy":     round(accuracy * 100, 2),
        "n_predictions":      len(records),
        "risk_distribution":  dist,
        "feature_importances": fi,
        "predictions":        records,
    }

    with open(LIVE_FILE, "w") as f:
        json.dump(live, f, indent=2)

    print(f"  Live predictions saved -> {LIVE_FILE}")


# ─── Watch mode ──────────────────────────────────────────────────────────────
def watch_mode(interval=30):
    print(f"[Watch] Retraining every {interval}s when CSV grows…  (Ctrl+C to stop)")
    last_size = 0
    while True:
        try:
            size = os.path.getsize(CSV_PATH) if os.path.exists(CSV_PATH) else 0
            if size != last_size:
                print(f"\n[{datetime.now():%H:%M:%S}] CSV changed (size={size})  — retraining…")
                train(verbose=True)
                last_size = size
            else:
                print(f"  [{datetime.now():%H:%M:%S}] No change (size={size})", end="\r")
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[Watch] Stopped.")
            break


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sleeping Monitor ML Trainer")
    parser.add_argument("--watch", action="store_true",
                        help="Keep running and retrain when CSV grows")
    parser.add_argument("--interval", type=int, default=30,
                        help="Re-check interval in seconds (default: 30)")
    args = parser.parse_args()

    if args.watch:
        watch_mode(interval=args.interval)
    else:
        result = train(verbose=True)
        if result is None:
            sys.exit(1)
