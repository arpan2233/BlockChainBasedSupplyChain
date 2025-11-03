# ai_model.py
"""
AI module: trains a simple IsolationForest on synthetic examples if model missing,
and exposes detect_anomalies(blockchain) which returns list of alerts with reasons.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta
import random

MODEL_FILE = "model.joblib"
SCALER_FILE = "scaler.joblib"

def generate_synthetic_dataset(n_normal=800, n_anom=200):
    """
    Create synthetic features:
      - transit_time_hours: typical small number; anomalies are very large
      - skipped_stage: 0 or 1
      - is_duplicate: 0 or 1
    """
    rows = []
    for _ in range(n_normal):
        transit = max(0.5, random.gauss(24, 8))  # avg 24 hours
        skipped = 0
        dup = 0
        rows.append([transit, skipped, dup])
    for _ in range(n_anom):
        typ = random.choice(["long_delay", "skipped", "duplicate"])
        if typ == "long_delay":
            rows.append([random.uniform(200, 1000), 0, 0])
        elif typ == "skipped":
            rows.append([random.uniform(1, 100), 1, 0])
        else:
            rows.append([random.uniform(1, 100), 0, 1])
    df = pd.DataFrame(rows, columns=["transit_time_hours", "skipped_stage", "is_duplicate"])
    return df

def train_if_missing():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        return
    print("Training AI model on synthetic data (this may take a few seconds)...")
    df = generate_synthetic_dataset()
    X = df[["transit_time_hours", "skipped_stage", "is_duplicate"]].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = IsolationForest(n_estimators=200, contamination=0.15, random_state=42)
    model.fit(Xs)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print("Model trained and saved.")

def load_model():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        train_if_missing()
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return model, scaler

def extract_features_from_events(events):
    """
    events: list of dicts as returned by blockchain.get_all_events()
    Each event data is expected to include keys:
      - product_id (str)
      - stage (str) e.g., Manufactured, Shipped, Delivered
      - transit_time_hours (float)
      - skipped_stage (0/1) optional
      - is_duplicate (0/1) optional
    """
    rows = []
    meta_rows = []
    for e in events:
        d = e["data"]
        transit = float(d.get("transit_time_hours", 0.0))
        skipped = int(d.get("skipped_stage", 0))
        dup = int(d.get("is_duplicate", 0))
        rows.append([transit, skipped, dup])
        meta_rows.append({"block_index": e["block_index"], "product_id": d.get("product_id"), "raw": d})
    if not rows:
        return None, None
    df = pd.DataFrame(rows, columns=["transit_time_hours", "skipped_stage", "is_duplicate"])
    return df, meta_rows

def detect_anomalies(blockchain):
    """
    Returns list of anomalies with human readable reason:
      - 'model_anomaly' if IsolationForest flagged it
      - 'rule_skip' if stage order is violated
      - 'rule_duplicate' if duplicate product id occurs in multiple concurrent chains
      - 'rule_large_delay' if transit_time_hours very large
    """
    model, scaler = load_model()
    events = blockchain.get_all_events()
    df, meta = extract_features_from_events(events)
    alerts = []

    # quick rule checks
    # duplicate product id: if same product_id appears with different route/time suspicious
    pid_counts = {}
    for m in meta or []:
        pid = m["product_id"]
        pid_counts[pid] = pid_counts.get(pid, 0) + 1

    for m in meta or []:
        pid = m["product_id"]
        d = m["raw"]
        reasons = []
        if pid and pid_counts.get(pid, 0) > 4:  # heuristic: many events for same id across chains
            reasons.append("possible_duplicate_id")
        if float(d.get("transit_time_hours", 0)) > 200:
            reasons.append("large_delay")
        if int(d.get("skipped_stage", 0)) == 1:
            reasons.append("skipped_stage")

        alerts.append({
            "block_index": m["block_index"],
            "product_id": pid,
            "reasons": reasons,
            "raw": d
        })

    # model based anomalies
    if df is not None and len(df) > 0:
        Xs = scaler.transform(df[["transit_time_hours", "skipped_stage", "is_duplicate"]].values)
        preds = model.predict(Xs)  # -1 anomaly, 1 normal
        for i, p in enumerate(preds):
            if p == -1:
                # add or augment reason for corresponding alert
                if i < len(alerts):
                    alerts[i]["reasons"].append("model_anomaly")
                else:
                    alerts.append({
                        "block_index": meta[i]["block_index"],
                        "product_id": meta[i]["product_id"],
                        "reasons": ["model_anomaly"],
                        "raw": meta[i]["raw"]
                    })
    # keep only alerts that have reasons
    filtered = [a for a in alerts if a["reasons"]]
    # attach readable reason string
    for a in filtered:
        a["reason_text"] = ", ".join(a["reasons"])
    return filtered
