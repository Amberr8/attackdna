import sys
import pandas as pd
import numpy as np
import pickle
import os
import time
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


NON_FEATURE_COLUMNS = [
    "Flow ID", "Source IP", "Destination IP", "Timestamp",
    "Source Port", "Label", "entity", "row_id"
]


def get_feature_columns(df):
    return [col for col in df.columns if col not in NON_FEATURE_COLUMNS]


def train_entity_lof(entity_df, feature_cols, min_flows=100, target_fpr=0.05,
                      n_neighbors=20, max_train_rows=20000):
    """Train one LOF model per entity, same target-FPR calibration
    approach as the Isolation Forest version. LOF training data is
    capped at max_train_rows for very large entities, since LOF's
    memory/time cost grows fast with training set size - this is a
    documented, deliberate trade-off, not an oversight."""
    if len(entity_df) < min_flows:
        print(f"  Skipping - only {len(entity_df)} flows (need {min_flows}+).")
        return None, None, None

    shuffled = entity_df.sample(frac=1, random_state=42)
    if len(shuffled) > max_train_rows:
        print(f"  Capping training rows: {len(shuffled)} -> {max_train_rows}")
        shuffled = shuffled.iloc[:max_train_rows]

    split_point = int(len(shuffled) * 0.8)
    train_df = shuffled.iloc[:split_point]
    val_df = shuffled.iloc[split_point:]

    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LocalOutlierFactor(
        n_neighbors=min(n_neighbors, len(X_train_scaled) - 1),
        novelty=True,
        n_jobs=-1
    )
    model.fit(X_train_scaled)

    val_scores = model.decision_function(X_val_scaled)
    threshold = np.percentile(val_scores, target_fpr * 100)

    return model, scaler, threshold


def save_entity_model(entity_ip, model, scaler, threshold, feature_cols, out_dir="models_lof"):
    os.makedirs(out_dir, exist_ok=True)
    safe_name = entity_ip.replace(".", "_")
    path = os.path.join(out_dir, f"{safe_name}.pkl")

    bundle = {
        "model": model, "scaler": scaler, "threshold": threshold,
        "feature_cols": feature_cols, "entity_ip": entity_ip
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  Saved -> {path} (threshold={threshold:.4f})")


if __name__ == "__main__":
    input_paths = sys.argv[1:] if len(sys.argv) > 1 else ["data/processed/monday_tw.csv"]

    print(f"Loading baseline from {len(input_paths)} file(s)...")
    frames = []
    for path in input_paths:
        day_df = pd.read_csv(path)
        day_df = day_df[day_df["Label"] == "BENIGN"]
        frames.append(day_df)
    df = pd.concat(frames, ignore_index=True)
    print(f"Combined baseline: {len(df)} BENIGN rows, {df['entity'].nunique()} entities\n")

    feature_cols = get_feature_columns(df)
    print(f"Using {len(feature_cols)} feature columns.")

    entities = df["entity"].unique()
    print(f"Training LOF models for {len(entities)} entities...\n")

    start = time.time()
    trained_count = skipped_count = 0

    for entity_ip in entities:
        print(f"Entity: {entity_ip} ({len(df[df['entity']==entity_ip])} rows)")
        entity_df = df[df["entity"] == entity_ip]
        model, scaler, threshold = train_entity_lof(entity_df, feature_cols)
        if model is None:
            skipped_count += 1
            continue
        save_entity_model(entity_ip, model, scaler, threshold, feature_cols)
        trained_count += 1

    elapsed = time.time() - start
    print(f"\nDone. Trained: {trained_count}, Skipped: {skipped_count}")
    print(f"Total training time: {elapsed:.1f}s")
