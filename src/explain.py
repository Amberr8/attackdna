import pandas as pd
import numpy as np
import sys

from config import CFG

label_col = CFG["label_col"]
benign_value = CFG["benign_value"]

NON_FEATURE_COLUMNS = [
    CFG["src_ip_col"], CFG["dst_ip_col"], CFG["dport_col"],
    label_col, "entity", "predicted", "anomaly_score",
    "Flow ID", "Timestamp", "Source Port", "BinaryLabel", "SubCategoryLabel",
    "row_id", "scan_flag", "final_flag",
    # Derived from the label - would be leakage in an explanation too.
    "frac_attack_in_bucket",
]


def get_feature_columns(df):
    return [col for col in df.columns if col not in NON_FEATURE_COLUMNS]


def build_baseline_stats(monday_df, feature_cols):
    """For each entity, compute mean and std of every feature."""
    means = monday_df.groupby("entity")[feature_cols].mean()
    stds = monday_df.groupby("entity")[feature_cols].std()
    return means, stds


def explain_entity_batch(entity_df, feature_cols, mean_row, std_row, top_n=3):
    """Vectorized explanation for ALL anomalies of one entity at once,
    instead of looping row by row."""
    X = entity_df[feature_cols].values.astype(float)
    mean_arr = mean_row[feature_cols].values.astype(float)
    std_arr = std_row[feature_cols].values.astype(float)

    std_arr_safe = np.where((std_arr == 0) | np.isnan(std_arr), np.nan, std_arr)

    z_scores = np.abs(X - mean_arr) / std_arr_safe
    z_scores = np.nan_to_num(z_scores, nan=0.0)

    top_idx = np.argsort(-z_scores, axis=1)[:, :top_n]

    explanations = []
    for row_i in range(len(entity_df)):
        parts = []
        for col_i in top_idx[row_i]:
            z = z_scores[row_i, col_i]
            if z < 1:
                continue
            col_name = feature_cols[col_i]
            actual = X[row_i, col_i]
            mean = mean_arr[col_i]
            direction = "higher" if actual > mean else "lower"
            if mean != 0:
                ratio = actual / mean
                parts.append(f"{col_name} is {abs(ratio):.1f}x {direction} than normal (z={z:.1f})")
            else:
                parts.append(f"{col_name} is unusually {direction} (z={z:.1f})")
        if not parts:
            explanations.append("Deviation detected but no single feature stands out strongly.")
        else:
            explanations.append("; ".join(parts))

    return explanations


def compute_risk_score(anomaly_score, entity_criticality, mitre_severity, threat_intel=0):
    return (0.40 * anomaly_score +
            0.30 * entity_criticality +
            0.20 * mitre_severity +
            0.10 * threat_intel)


if __name__ == "__main__":

    baseline_paths = sys.argv[1].split(",")
    scored_path = sys.argv[2]
    output_path = sys.argv[3]
    print(f"Loading baseline from {len(baseline_paths)} file(s)...")
    monday_df = pd.concat([pd.read_csv(p) for p in baseline_paths], ignore_index=True)
    monday_df = monday_df[monday_df[label_col] == benign_value]

    print(f"Loading scored data: {scored_path} ...")
    scored_df = pd.read_csv(scored_path)

    feature_cols = get_feature_columns(monday_df)
    print(f"Using {len(feature_cols)} feature columns.")

    print("Building per-entity baseline statistics...")
    means, stds = build_baseline_stats(monday_df, feature_cols)

    anomalies = scored_df[scored_df["predicted"] == -1].copy()
    print(f"Explaining {len(anomalies)} flagged anomalies (vectorized)...")

    all_explanations = pd.Series(index=anomalies.index, dtype=object)

    for entity_ip, entity_df in anomalies.groupby("entity"):
        if entity_ip not in means.index:
            all_explanations.loc[entity_df.index] = "No baseline available for this entity."
            continue

        mean_row = means.loc[entity_ip]
        std_row = stds.loc[entity_ip]

        explanations = explain_entity_batch(entity_df, feature_cols, mean_row, std_row)
        all_explanations.loc[entity_df.index] = explanations
        print(f"  Entity {entity_ip}: explained {len(entity_df)} anomalies.")

    anomalies["explanation"] = all_explanations

    anomalies.to_csv(output_path, index=False)
    print(f"\nSaved {len(anomalies)} explained anomalies to {output_path}")

    print("\n--- Sample explanations ---")
    for i in range(min(5, len(anomalies))):
        row = anomalies.iloc[i]
        print(f"\nEntity: {row['entity']} | True label: {row[label_col]}")
        print(f"  {row['explanation']}")
