# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

DATA_CSV = "transactions.csv"
MODEL_FILE = "anomaly_model.joblib"
SCALER_FILE = "scaler.joblib"

def prepare_features(df: pd.DataFrame):
    # Select / create numeric features for anomaly detection
    # transit_time_hours, hops_remaining, is_duplicate
    X = df[["transit_time_hours", "hops_remaining", "is_duplicate"]].copy()
    # fill na
    X["transit_time_hours"] = X["transit_time_hours"].fillna(X["transit_time_hours"].median())
    X["hops_remaining"] = X["hops_remaining"].fillna(X["hops_remaining"].median())
    return X

def train_and_save():
    df = pd.read_csv(DATA_CSV)
    X = prepare_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # IsolationForest (unsupervised) — good for anomaly detection
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(X_scaled)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"Trained IsolationForest model saved to {MODEL_FILE}")
    print(f"Scaler saved to {SCALER_FILE}")

if __name__ == "__main__":
    train_and_save()
