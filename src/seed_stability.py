import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

NON_FEATURE_COLUMNS = [
    "Flow ID", "Source IP", "Destination IP", "Timestamp",
    "Source Port", "Label", "entity", "row_id"
]


def get_feature_columns(df):
    return [col for col in df.columns if col not in NON_FEATURE_COLUMNS]


def train_and_score(baseline_df, test_df, feature_cols, seed, min_flows=100, target_fpr=0.05):
    """Train all entities with the given random seed, then score the
    test set and return network-wide TP/FP/FN/TN. Nothing is saved to
    disk - this is purely for measuring seed-to-seed stability."""
    all_predictions = []

    for entity_ip in baseline_df["entity"].unique():
        entity_df = baseline_df[baseline_df["entity"] == entity_ip]
        if len(entity_df) < min_flows:
            continue

        shuffled = entity_df.sample(frac=1, random_state=seed)
        split_point = int(len(shuffled) * 0.8)
        train_part = shuffled.iloc[:split_point]
        val_part = shuffled.iloc[split_point:]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_part[feature_cols].values)
        X_val = scaler.transform(val_part[feature_cols].values)

        model = IsolationForest(n_estimators=100, contamination="auto",
                                 random_state=seed, n_jobs=-1)
        model.fit(X_train)

        val_scores = model.decision_function(X_val)
        threshold = np.percentile(val_scores, target_fpr * 100)

        test_entity = test_df[test_df["entity"] == entity_ip]
        if len(test_entity) == 0:
            continue
        X_test = scaler.transform(test_entity[feature_cols].values)
        test_scores = model.decision_function(X_test)
        predicted = np.where(test_scores < threshold, -1, 1)

        result = test_entity[["Label"]].copy()
        result["predicted"] = predicted
        all_predictions.append(result)

    all_df = pd.concat(all_predictions, ignore_index=True)
    is_attack = all_df["Label"] != "BENIGN"
    is_flagged = all_df["predicted"] == -1

    tp = (is_attack & is_flagged).sum()
    fn = (is_attack & ~is_flagged).sum()
    fp = (~is_attack & is_flagged).sum()
    tn = (~is_attack & ~is_flagged).sum()

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return precision, recall, f1


if __name__ == "__main__":
    baseline_paths = sys.argv[1].split(",")
    test_path = sys.argv[2]
    seeds = [1, 7, 21, 99, 123]

    print(f"Loading baseline from {len(baseline_paths)} file(s)...")
    frames = [pd.read_csv(p) for p in baseline_paths]
    baseline_df = pd.concat(frames, ignore_index=True)
    baseline_df = baseline_df[baseline_df["Label"] == "BENIGN"]
    print(f"Baseline: {len(baseline_df)} BENIGN rows")

    print(f"Loading test file: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"Test: {len(test_df)} rows\n")

    feature_cols = get_feature_columns(baseline_df)

    results = []
    for seed in seeds:
        print(f"Training with seed={seed}...")
        precision, recall, f1 = train_and_score(baseline_df, test_df, feature_cols, seed)
        print(f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
        results.append({"seed": seed, "precision": precision, "recall": recall, "f1": f1})

    results_df = pd.DataFrame(results)
    print("\n--- Stability across seeds ---")
    print(results_df.to_string(index=False))
    print(f"\nPrecision: mean={results_df['precision'].mean():.3f}  std={results_df['precision'].std():.4f}")
    print(f"Recall:    mean={results_df['recall'].mean():.3f}  std={results_df['recall'].std():.4f}")
    print(f"F1:        mean={results_df['f1'].mean():.3f}  std={results_df['f1'].std():.4f}")
