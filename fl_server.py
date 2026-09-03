#!/usr/bin/env python3
"""
fl_server.py — Federated Learning aggregator for the sleeping monitor
=====================================================================

Each bed runs its own monitor (ESP32-CAM + img_process.py) and trains a local
risk classifier on the positions it actually saw. Raw frames and position logs
never leave the bedside; only the trained model is uploaded. This server
averages those models into a global one and hands it back.

The brain is the RandomForest from commit 8e887afd, so aggregation is a forest
merge: the global ensemble is the union of every client's trees. That is the
standard way to federate a bagged ensemble — each tree is already an
independent voter, so pooling them preserves exactly what RandomForest does.

Endpoints
    POST /upload        client submits its locally trained model
    GET  /global        client downloads the current global model
    GET  /status        round number, client count, accuracy
    GET  /              human-readable dashboard

Run:  python fl_server.py        (listens on 0.0.0.0:8081)
"""

import copy
import io
import json
import os
import pickle
import threading
import time
from datetime import datetime

import numpy as np
from flask import Flask, jsonify, request, send_file

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
FL_DIR     = os.path.join(BASE_DIR, "fl_store")
GLOBAL_PKL = os.path.join(FL_DIR, "global_model.pkl")
HISTORY    = os.path.join(FL_DIR, "fl_history.json")
SEED_MODEL = os.path.join(BASE_DIR, "backend", "models", "risk_classifier.pkl")

# Aggregate once this many clients have reported in since the last round.
MIN_CLIENTS_PER_ROUND = int(os.environ.get("FL_MIN_CLIENTS", "2"))

# Cap the global forest so it cannot grow without bound across many rounds.
MAX_GLOBAL_TREES = 400

os.makedirs(FL_DIR, exist_ok=True)

_lock = threading.Lock()
_pending = []          # client bundles waiting to be aggregated
_state = {
    "round": 0,
    "clients_total": 0,
    "clients_this_round": 0,
    "last_aggregated": None,
    "global_trees": 0,
    "history": [],
}


# ─────────────────────────────────────────────────────────
# Model store
# ─────────────────────────────────────────────────────────
def _load_global():
    """Current global model, seeded from the committed classifier on first run."""
    for path in (GLOBAL_PKL, SEED_MODEL):
        if os.path.isfile(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as exc:
                print(f"  [FL] could not read {path}: {exc}")
    return None


def _save_global(bundle):
    tmp = GLOBAL_PKL + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(bundle, f)
    os.replace(tmp, GLOBAL_PKL)


def _save_history():
    try:
        with open(HISTORY, "w", encoding="utf-8") as f:
            json.dump(_state, f, indent=2)
    except OSError:
        pass


# ─────────────────────────────────────────────────────────
# Aggregation — FedAvg adapted to a bagged forest
# ─────────────────────────────────────────────────────────
def aggregate(client_bundles):
    """Merge client forests into one global forest.

    Each client contributes a share of trees proportional to how much data it
    trained on, which is the FedAvg weighting rule applied to an ensemble.
    """
    base = _load_global()
    contributors = [b for b in client_bundles if b.get("clf") is not None]
    if not contributors:
        return base, 0

    # Every client must agree on the label set and feature order, otherwise
    # their trees are not voting on the same question.
    ref = contributors[0]
    ref_classes = list(ref["le"].classes_)
    usable = []
    for b in contributors:
        if list(b["le"].classes_) != ref_classes:
            print(f"  [FL] skipping client {b.get('client_id')}: class mismatch")
            continue
        if list(b["features"]) != list(ref["features"]):
            print(f"  [FL] skipping client {b.get('client_id')}: feature mismatch")
            continue
        # Trees that vote over a narrower class space cannot be averaged with
        # the rest — their probability vectors have the wrong width.
        if getattr(b["clf"], "n_classes_", None) != len(ref_classes):
            print(f"  [FL] skipping client {b.get('client_id')}: "
                  f"trained on {getattr(b['clf'], 'n_classes_', '?')} of "
                  f"{len(ref_classes)} classes")
            continue
        usable.append(b)

    if not usable:
        return base, 0

    total_n = sum(max(1, b.get("n_samples", 1)) for b in usable)
    merged = copy.deepcopy(usable[0]["clf"])
    trees, quota_used = [], []

    for b in usable:
        clf = b["clf"]
        share = max(1, b.get("n_samples", 1)) / total_n
        take = max(1, int(round(share * MAX_GLOBAL_TREES)))
        picked = list(clf.estimators_)[:take]
        trees.extend(picked)
        quota_used.append((b.get("client_id", "?"), len(picked), b.get("n_samples", 0)))

    # Carry a slice of the previous global model so knowledge persists across
    # rounds instead of each round starting from only the newest clients.
    if base is not None and getattr(base.get("clf"), "estimators_", None):
        keep = max(1, MAX_GLOBAL_TREES // 4)
        trees.extend(list(base["clf"].estimators_)[:keep])

    trees = trees[:MAX_GLOBAL_TREES]
    merged.estimators_ = trees
    merged.n_estimators = len(trees)

    bundle = {
        "clf": merged,
        "le": usable[0]["le"],
        "features": list(usable[0]["features"]),
        "federated": True,
        "round": _state["round"] + 1,
        "contributors": [c[0] for c in quota_used],
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }

    print(f"  [FL] merged {len(trees)} trees from {len(usable)} client(s):")
    for cid, n_trees, n_samp in quota_used:
        print(f"        {cid}: {n_trees} trees  ({n_samp} samples)")
    return bundle, len(usable)


def _maybe_aggregate():
    """Run a round once enough clients have reported. Caller holds the lock."""
    if len(_pending) < MIN_CLIENTS_PER_ROUND:
        return False

    bundle, n = aggregate(_pending)
    if bundle is None or n == 0:
        _pending.clear()
        return False

    _save_global(bundle)
    _state["round"] += 1
    _state["clients_this_round"] = n
    _state["global_trees"] = bundle["clf"].n_estimators
    _state["last_aggregated"] = datetime.now().isoformat(timespec="seconds")
    _state["history"].append({
        "round": _state["round"],
        "clients": n,
        "trees": _state["global_trees"],
        "at": _state["last_aggregated"],
    })
    _state["history"] = _state["history"][-50:]
    _pending.clear()
    _save_history()
    print(f"  [FL] === round {_state['round']} complete "
          f"({n} clients, {_state['global_trees']} trees) ===")
    return True


# ─────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload():
    """Receive one client's locally trained model."""
    raw = request.get_data()
    if not raw:
        return jsonify({"ok": False, "error": "empty body"}), 400

    try:
        bundle = pickle.loads(raw)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"unpickle failed: {exc}"}), 400

    if not all(k in bundle for k in ("clf", "le", "features")):
        return jsonify({"ok": False, "error": "missing clf/le/features"}), 400

    cid = str(bundle.get("client_id", request.remote_addr))
    with _lock:
        _pending.append(bundle)
        _state["clients_total"] += 1
        pending_now = len(_pending)
        aggregated = _maybe_aggregate()
        rnd = _state["round"]

    print(f"  [FL] upload from {cid} "
          f"({bundle.get('n_samples', '?')} samples) — pending {pending_now}")

    return jsonify({
        "ok": True,
        "client_id": cid,
        "pending": 0 if aggregated else pending_now,
        "aggregated": aggregated,
        "round": rnd,
        "needed": MIN_CLIENTS_PER_ROUND,
    })


@app.route("/global")
def global_model():
    """Serve the current global model for clients to pull."""
    bundle = _load_global()
    if bundle is None:
        return jsonify({"ok": False, "error": "no global model yet"}), 404
    buf = io.BytesIO(pickle.dumps(bundle))
    buf.seek(0)
    return send_file(buf, mimetype="application/octet-stream",
                     as_attachment=True, download_name="global_model.pkl")


@app.route("/status")
def status():
    with _lock:
        st = dict(_state)
        st["pending"] = len(_pending)
    st["min_clients"] = MIN_CLIENTS_PER_ROUND
    st["has_global"] = os.path.isfile(GLOBAL_PKL)
    return jsonify(st)


@app.route("/")
def index():
    with _lock:
        rnd, pending = _state["round"], len(_pending)
        trees, total = _state["global_trees"], _state["clients_total"]
        hist = list(_state["history"])[-10:]
    rows = "".join(
        f"<tr><td>{h['round']}</td><td>{h['clients']}</td>"
        f"<td>{h['trees']}</td><td>{h['at']}</td></tr>" for h in reversed(hist)
    ) or "<tr><td colspan=4>no rounds yet</td></tr>"
    return (
        "<!doctype html><meta charset='utf-8'>"
        "<title>Federated Learning Server</title>"
        "<style>body{background:#111;color:#eee;font-family:monospace;padding:24px}"
        "b{color:#0cf;font-size:1.6em}"
        "table{border-collapse:collapse;margin-top:14px}"
        "td,th{border:1px solid #333;padding:5px 12px;text-align:left}</style>"
        "<h2>Sleeping Monitor &mdash; Federated Learning</h2>"
        f"<p>Round: <b>{rnd}</b> &nbsp; Global trees: <b>{trees}</b></p>"
        f"<p>Uploads total: {total} &nbsp;|&nbsp; pending this round: {pending}"
        f" / {MIN_CLIENTS_PER_ROUND}</p>"
        "<table><tr><th>Round</th><th>Clients</th><th>Trees</th><th>At</th></tr>"
        f"{rows}</table>"
    )


if __name__ == "__main__":
    print("=" * 58)
    print("  Sleeping Monitor — Federated Learning Server")
    print("=" * 58)
    print("  Listening on http://0.0.0.0:8081")
    print(f"  Aggregating every {MIN_CLIENTS_PER_ROUND} client upload(s)")
    print(f"  Seed model: {SEED_MODEL}")
    print("=" * 58)
    app.run(host="0.0.0.0", port=8081, threaded=True)
