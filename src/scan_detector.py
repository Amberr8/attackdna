import pandas as pd
import numpy as np
import sys


def build_scan_thresholds(baseline_df, k_sigma=5.0):
    """Per-entity mean+std of the two time-window features, using
    ONLY benign baseline traffic. k=5 is intentionally stricter than
    the main model's k=3, since this check targets a specific,
    high-confidence signal rather than general anomalies."""
    stats = baseline_df.groupby("entity")[
        ["connections_per_minute", "distinct_ports_per_minute"]
    ].agg(["mean", "std"])
    stats.columns = ["_".join(c) for c in stats.columns]

    stats["conn_threshold"] = stats["connections_per_minute_mean"] + k_sigma * stats["connections_per_minute_std"]
    stats["port_threshold"] = stats["distinct_ports_per_minute_mean"] + k_sigma * stats["distinct_ports_per_minute_std"]
    return stats[["conn_threshold", "port_threshold"]]


def flag_scans(df, thresholds):
    df = df.copy()
    df = df.merge(thresholds, on="entity", how="left")
    df["scan_flag"] = (
        (df["connections_per_minute"] > df["conn_threshold"]) |
        (df["distinct_ports_per_minute"] > df["port_threshold"])
    )
    return df


if __name__ == "__main__":
    baseline_paths = sys.argv[1].split(",")
    test_path = sys.argv[2]
    output_path = sys.argv[3]

    baseline = pd.concat([pd.read_csv(p) for p in baseline_paths], ignore_index=True)
    baseline = baseline[baseline["Label"] == "BENIGN"]

    thresholds = build_scan_thresholds(baseline)
    print(thresholds)

    df = pd.read_csv(test_path)
    df = flag_scans(df, thresholds)

    print("\n--- Scan-flag results by true label ---")
    for label, group in df.groupby("Label"):
        flagged = group["scan_flag"].sum()
        print(f"{label}: {flagged}/{len(group)} flagged ({flagged/len(group):.1%})")

    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
