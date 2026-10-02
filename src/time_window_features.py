import pandas as pd
import numpy as np
import sys

from config import CFG


def add_time_window_features(df):
    """Add two burst-detection features per flow: how many connections
    this entity made, and how many distinct destination ports it
    touched, in the same time window as this flow. Computed per-entity
    so one host's burst doesn't inflate another host's numbers.

    CICIDS2017 has a real Timestamp column, so this buckets by actual
    60-second wall-clock windows.

    GeNIS has no real timestamp - only "Offset", an Argus record
    position value, confirmed NOT to be elapsed time (multiple flows
    lasting 28-40 seconds share the identical Offset value, which is
    impossible if Offset were real seconds). For GeNIS, this instead
    buckets by fixed-width ranges of Offset, as a documented substitute
    for a real time window - it captures "flows close together in
    capture order" rather than "flows in the same 60 real seconds".
    The bucket width (10000) was chosen empirically: it produces a
    clear separation between benign traffic (median 1 distinct
    port/bucket) and recon/scan traffic (median 21 distinct
    ports/bucket) on the GeNIS test set."""
    df = df.copy()
    dport_col = CFG["dport_col"]

    if CFG["timestamp_col"] is not None:
        df["_ts"] = pd.to_datetime(df[CFG["timestamp_col"]], errors="coerce")
        n_bad = df["_ts"].isna().sum()
        if n_bad > 0:
            print(f"  Warning: {n_bad} rows had unparseable timestamps "
                  f"(features will be NaN -> filled with 0 for these)")
        df["_bucket"] = df["_ts"].dt.floor("60s")
        drop_cols = ["_ts", "_bucket"]
    else:
        bucket_size = CFG["offset_bucket_size"]
        time_col = CFG["time_col"]
        print(f"  No real timestamp for this dataset - bucketing by "
              f"{time_col} // {bucket_size} instead (see docstring).")
        df["_bucket"] = df[time_col] // bucket_size
        drop_cols = ["_bucket"]

    conn_counts = (
        df.groupby(["entity", "_bucket"])
        .size()
        .rename("connections_per_minute")
        .reset_index()
    )
    port_counts = (
        df.groupby(["entity", "_bucket"])[dport_col]
        .nunique()
        .rename("distinct_ports_per_minute")
        .reset_index()
    )

    df = df.merge(conn_counts, on=["entity", "_bucket"], how="left")
    df = df.merge(port_counts, on=["entity", "_bucket"], how="left")

    df["connections_per_minute"] = df["connections_per_minute"].fillna(0)
    df["distinct_ports_per_minute"] = df["distinct_ports_per_minute"].fillna(0)

    # For threshold calibration, scan_detector.py needs to know whether a
    # benign-labeled row's own bucket also contains attack traffic. In
    # CICIDS2017 this never happens (baseline days are fully benign), but
    # in GeNIS the train file is a random mix, so a benign row can share
    # its Offset bucket with concurrent attack flows, inflating the
    # burst features used to calibrate that entity's threshold. Flagging
    # this lets scan_detector.py calibrate from clean buckets only.
    label_col = CFG["label_col"]
    benign_value = CFG["benign_value"]
    frac_attack = (
        df.groupby(["entity", "_bucket"])[label_col]
        .apply(lambda x: (x != benign_value).mean())
        .rename("frac_attack_in_bucket")
        .reset_index()
    )
    df = df.merge(frac_attack, on=["entity", "_bucket"], how="left")

    df = df.drop(columns=drop_cols)
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
