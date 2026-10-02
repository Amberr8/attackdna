import pandas as pd
import numpy as np
import pickle
import os
import sys

from config import CFG


def load_entity_model(entity_ip, models_dir="models"):
    """Load a saved model+scaler+threshold bundle for one entity. Returns
    None if no model exists for this entity."""
    safe_name = str(entity_ip).replace(".", "_")
    path = os.path.join(models_dir, f"{safe_name}.pkl")

    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    return bundle


def score_entity(entity_df, bundle):
    """Score all of one entity's flows using its trained model and its
    own calibrated threshold (not Isolation Forest's built-in default).
    A flow is flagged as anomalous if its score falls below the
    entity's threshold, which was calibrated to target a 5% false
    positive rate on clean validation data."""
    model = bundle["model"]
    scaler = bundle["scaler"]
    threshold = bundle["threshold"]
    feature_cols = bundle["feature_cols"]

    X = entity_df[feature_cols].values
    X_scaled = scaler.transform(X)

    anomaly_score = model.decision_function(X_scaled)
    predicted = np.where(anomaly_score < threshold, -1, 1)

    entity_df = entity_df.copy()
    entity_df["predicted"] = predicted
    entity_df["anomaly_score"] = anomaly_score

    return entity_df


if __name__ == "__main__":
    input_path = sys.argv[1]
    label_col = CFG["label_col"]

    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows.")

    entities = df["entity"].unique()
    print(f"Scoring {len(entities)} entities...\n")

    results = []
    no_model_count = 0

    for entity_ip in entities:
        bundle = load_entity_model(entity_ip)

        if bundle is None:
            print(f"Entity {entity_ip}: no model found, skipping.")
            no_model_count += len(df[df["entity"] == entity_ip])
            continue

        entity_df = df[df["entity"] == entity_ip]
        scored_df = score_entity(entity_df, bundle)
        results.append(scored_df)

        n_anomalies = (scored_df["predicted"] == -1).sum()
        print(f"Entity {entity_ip}: {len(scored_df)} flows, "
              f"{n_anomalies} flagged as anomalies.")

    if not results:
        print("\nNo entities were scored - no matching models found.")
        sys.exit(1)

    all_results = pd.concat(results, ignore_index=True)

    print(f"\nRows skipped (no model): {no_model_count}")
    print(f"Total scored: {len(all_results)}")

    print("\n--- Detection results by true label ---")
    summary = all_results.groupby(label_col)["predicted"].apply(
        lambda x: (x == -1).sum()
    )
    total_by_label = all_results[label_col].value_counts()

    for label in total_by_label.index:
        flagged = summary.get(label, 0)
        total = total_by_label[label]
        print(f"{label}: {flagged}/{total} flagged as anomaly "
              f"({flagged/total:.1%})")

    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/genis_test_scored.csv"
    all_results.to_csv(output_path, index=False)
    print(f"\nSaved scored results to {output_path}")
