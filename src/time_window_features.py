import pandas as pd
import numpy as np
import sys


def add_time_window_features(df, window="60s"):
    """Add two time-window features per flow: how many connections this
    entity made, and how many distinct destination ports it touched,
    in the same 60-second bucket as this flow. Computed per-entity so
    one host's burst doesn't inflate another host's numbers."""
    df = df.copy()
    df["_ts"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    n_bad = df["_ts"].isna().sum()
    if n_bad > 0:
        print(f"  Warning: {n_bad} rows had unparseable timestamps "
              f"(features will be NaN -> filled with 0 for these)")

    df["_bucket"] = df["_ts"].dt.floor(window)

    conn_counts = (
        df.groupby(["entity", "_bucket"])
        .size()
        .rename("connections_per_minute")
        .reset_index()
    )
    port_counts = (
        df.groupby(["entity", "_bucket"])["Destination Port"]
        .nunique()
        .rename("distinct_ports_per_minute")
        .reset_index()
    )

    df = df.merge(conn_counts, on=["entity", "_bucket"], how="left")
    df = df.merge(port_counts, on=["entity", "_bucket"], how="left")

    df["connections_per_minute"] = df["connections_per_minute"].fillna(0)
    df["distinct_ports_per_minute"] = df["distinct_ports_per_minute"].fillna(0)

    df = df.drop(columns=["_ts", "_bucket"])
    return df


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows.")

    df = add_time_window_features(df)

    print("\n--- New feature summary ---")
    print(df[["connections_per_minute", "distinct_ports_per_minute"]].describe())

    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
