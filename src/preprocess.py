import os
import glob
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
RAW_DIRS          = [
    "data/raw/sepsis/training_setA/training",
    "data/raw/sepsis/training_setB/training_setB"
]
PROCESSED_DIR     = "data/processed"
FEATURES          = ["O2Sat", "HR", "Resp"]
HYPOXIA_THRESHOLD = 90.0
WINDOW_SIZE       = 30
PREDICTION_HORIZON = 15
MIN_VALID_RATIO   = 0.6
SUBSET_SIZE       = None   # set to None to use full dataset


def load_patient(filepath):
    df = pd.read_csv(filepath, sep="|")
    cols = [c for c in FEATURES if c in df.columns]
    if len(cols) < len(FEATURES):
        return None
    return df[cols].copy()


def forward_fill_limited(df, limit=4):
    return df.ffill(limit=limit).bfill(limit=2)


def label_window(future_spo2):
    valid = future_spo2[~np.isnan(future_spo2)]
    if len(valid) == 0:
        return -1   # unknown — will be discarded
    return 1 if np.any(valid < HYPOXIA_THRESHOLD) else 0


def build_windows(df):
    windows, labels = [], []
    values = df.values
    T = len(values)

    for i in range(T - WINDOW_SIZE - PREDICTION_HORIZON):
        window = values[i : i + WINDOW_SIZE].copy()
        future = values[i + WINDOW_SIZE : i + WINDOW_SIZE + PREDICTION_HORIZON, 0]

        valid_ratio = np.sum(~np.isnan(window)) / window.size
        if valid_ratio < MIN_VALID_RATIO:
            continue

        col_medians = np.nanmedian(window, axis=0)
        # If entire column is NaN, use global safe defaults
        col_medians = np.where(np.isnan(col_medians), [95.0, 80.0, 16.0], col_medians)
        for col_idx in range(window.shape[1]):
            nan_mask = np.isnan(window[:, col_idx])
            window[nan_mask, col_idx] = col_medians[col_idx]

        # Final safety check — skip window if any NaN remains
        if np.any(np.isnan(window)):
            continue

        label = label_window(future)
        if label == -1:
            continue

        windows.append(window)
        labels.append(label)

    return windows, labels


def run_preprocessing():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    all_windows, all_labels = [], []

    for raw_dir in RAW_DIRS:
        psv_files = glob.glob(os.path.join(raw_dir, "*.psv"))
        print(f"\nProcessing {len(psv_files)} patients from {raw_dir}...")

        for filepath in tqdm(psv_files):
            df = load_patient(filepath)
            if df is None:
                continue
            df = forward_fill_limited(df)
            windows, labels = build_windows(df)
            all_windows.extend(windows)
            all_labels.extend(labels)

    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels,  dtype=np.int32)

    print(f"\nTotal windows before subset: {len(X)}")
    print(f"Hypoxia (1): {np.sum(y == 1)} | Stable (0): {np.sum(y == 0)}")

    # ── Subset for local training ─────────────────────────────────────────────
    if SUBSET_SIZE and len(X) > SUBSET_SIZE:
        # Stratified subset — keep class balance
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]
        ratio  = len(idx_1) / len(y)
        n1     = int(SUBSET_SIZE * ratio)
        n0     = SUBSET_SIZE - n1

        chosen_0 = np.random.choice(idx_0, min(n0, len(idx_0)), replace=False)
        chosen_1 = np.random.choice(idx_1, min(n1, len(idx_1)), replace=False)
        chosen   = np.concatenate([chosen_0, chosen_1])
        np.random.shuffle(chosen)

        X, y = X[chosen], y[chosen]
        print(f"Subset to {len(X)} windows (stratified)")
        print(f"Hypoxia (1): {np.sum(y == 1)} | Stable (0): {np.sum(y == 0)}")

    # ── Normalize ─────────────────────────────────────────────────────────────
    N, T, F  = X.shape
    X_flat   = X.reshape(-1, F)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat).reshape(N, T, F)

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(os.path.join(PROCESSED_DIR, "X.npy"), X_scaled)
    np.save(os.path.join(PROCESSED_DIR, "y.npy"), y)
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, "scaler.pkl"))

    print(f"\nSaved → X: {X_scaled.shape} | y: {y.shape}")
    print("Preprocessing complete.")


if __name__ == "__main__":
    np.random.seed(42)
    run_preprocessing()
