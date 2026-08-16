import sys
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler



NON_FEATURE_COLUMNS = [
    "Flow ID", "Source IP", "Destination IP", "Timestamp",
    "Source Port", "Label", "entity",
    # Zero variance - constant across all traffic, no information
    "CWE Flag Count", "Fwd Avg Bulk Rate", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd URG Flags",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bytes/Bulk", "Fwd URG Flags",
    "Bwd PSH Flags",
    # Redundant - correlation > 0.95 with a kept feature
    "Total Backward Packets", "Total Length of Bwd Packets",
    "Fwd IAT Total", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Fwd Packets/s",
    "Max Packet Length", "SYN Flag Count", "ECE Flag Count",
    "Average Packet Size"
]

def get_feature_columns(df):
    """Everything except IDs, timestamps, ports, and the label is a feature."""
    return [col for col in df.columns if col not in NON_FEATURE_COLUMNS]

def train_entity_model(entity_df, feature_cols, min_flows=10, target_fpr=0.05):
    if len(entity_df) < min_flows:
        print(f"  Skipping - only {len(entity_df)} flows (need {min_flows}+).")
        return None, None, None

    # Handle missing or infinite values
    X_df = entity_df[feature_cols].select_dtypes(include=[np.number])
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    shuffled = X_df.sample(frac=1, random_state=42)
    split_point = int(len(shuffled) * 0.8)

    # Fallback to full set if train or validation split is too small
    if split_point == 0 or split_point == len(shuffled):
        train_df = shuffled
        val_df = shuffled
    else:
        train_df = shuffled.iloc[:split_point]
        val_df = shuffled.iloc[split_point:]

    X_train = train_df.values
    X_val = val_df.values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled)

    val_scores = model.decision_function(X_val_scaled)
    threshold = np.percentile(val_scores, target_fpr * 100)

    return model, scaler, threshold

def save_entity_model(entity_ip, model, scaler, threshold, feature_cols, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    safe_name = entity_ip.replace(".", "_")
    path = os.path.join(out_dir, f"{safe_name}.pkl")
    bundle = {
        "model": model,
        "scaler": scaler,
        "threshold": threshold,
        "feature_cols": feature_cols,
        "entity_ip": entity_ip
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  Saved -> {path} (threshold={threshold:.4f})")

if __name__ == "__main__":
    input_paths = sys.argv[1:] if len(sys.argv) > 1 else ["data/processed/monday_clean.csv"]
    print(f"Loading baseline from {len(input_paths)} file(s)...")

    frames = []
    for path in input_paths:
        if not os.path.exists(path):
            print(f"  Warning: File not found: {path}")
            continue
        day_df = pd.read_csv(path)
        day_df.columns = day_df.columns.str.strip()
        before = len(day_df)
        day_df = day_df[day_df["Label"] == "BENIGN"]
        print(f"  {path}: {before} rows -> {len(day_df)} BENIGN rows")
        frames.append(day_df)

    if not frames:
        print("No valid baseline data found!")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    print(f"Combined baseline: {len(df)} BENIGN rows, {df['entity'].nunique()} entities\n")

    feature_cols = get_feature_columns(df)
    print(f"Using {len(feature_cols)} feature columns.")

    entities = df["entity"].unique()
    print(f"Training models for {len(entities)} entities...\n")

    trained_count = 0
    skipped_count = 0

    for entity_ip in entities:
        print(f"Entity: {entity_ip}")
        entity_df = df[df["entity"] == entity_ip]
        model, scaler, threshold = train_entity_model(entity_df, feature_cols)
        if model is None:
            skipped_count += 1
            continue
        save_entity_model(entity_ip, model, scaler, threshold, feature_cols)
        trained_count += 1

    print(f"\nDone. Trained: {trained_count}, Skipped: {skipped_count}")
