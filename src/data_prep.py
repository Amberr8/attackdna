import pandas as pd
import numpy as np
import sys

from config import CFG, DATASET


def load_raw_csv(path):
    """Load the CSV and strip whitespace from column names.

    CICIDS2017 needs latin1 encoding because some files (notably
    Thursday-Morning-WebAttacks) contain a non-UTF-8 byte (0x96) in the
    Label column. GeNIS's CSVs are already clean UTF-8, so use the
    default encoding for it."""
    if DATASET == "cicids2017":
        df = pd.read_csv(path, low_memory=False, encoding="latin1")
    else:
        df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def extract_entity(df):
    """Add an 'entity' column identifying which host each flow belongs to.

    CICIDS2017: CICFlowMeter assigns Source/Destination based on who sent
    the first packet, not internal vs external, so a given internal
    host's traffic is split across both columns. We fix that by filtering
    to the internal subnet and pointing 'entity' at whichever side is
    internal.

    GeNIS: Ssaddr/Sdaddr are anonymized integer host IDs inside a single
    closed testbed. There is no internal/external subnet to filter on,
    so every host ID is treated as its own entity.

    In both cases, if both sides of a flow are the same kind of host
    (both internal, or both inside the GeNIS testbed), the flow is
    counted once for each participating entity, since it's genuine
    traffic for both."""
    src_col = CFG["src_ip_col"]
    dst_col = CFG["dst_ip_col"]

    if CFG["internal_prefix"] is not None:
        internal_prefix = CFG["internal_prefix"]
        src_internal = df[src_col].astype(str).str.startswith(internal_prefix)
        dst_internal = df[dst_col].astype(str).str.startswith(internal_prefix)

        src_rows = df[src_internal].copy()
        src_rows["entity"] = src_rows[src_col]

        dst_rows = df[dst_internal].copy()
        dst_rows["entity"] = dst_rows[dst_col]

        combined = pd.concat([src_rows, dst_rows], ignore_index=True)

        dropped = len(df) - (src_internal | dst_internal).sum()
        print(f"Dropped {dropped} flows with neither side internal.")
    else:
        src_rows = df.copy()
        src_rows["entity"] = src_rows[src_col]

        dst_rows = df.copy()
        dst_rows["entity"] = dst_rows[dst_col]

        combined = pd.concat([src_rows, dst_rows], ignore_index=True)
        print(f"No internal/external filtering for dataset '{DATASET}'; "
              f"all {combined['entity'].nunique()} host IDs treated as entities.")

    return combined


def clean_flow_stats(df):
    """Fix inf and NaN values in the numeric columns. Some flow exporters
    write 'inf' when a duration is 0 and a rate can't be computed."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=numeric_cols)
    print(f"Dropped {before - len(df)} rows with NaN/inf values.")
    return df


def normalize_labels(df):
    """Clean up the label column. CICIDS2017 also needs a known broken
    character fixed; GeNIS's labels are already clean strings."""
    label_col = CFG["label_col"]
    df[label_col] = df[label_col].astype(str).str.strip()
    if DATASET == "cicids2017":
        df[label_col] = df[label_col].str.replace("\x96", "-", regex=False)
    return df


def drop_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows.")
    return df


def cast_bool_columns(df):
    """GeNIS's one-hot columns (Proto_tcp, State_CON, etc.) load as pandas
    bool dtype. select_dtypes(include=[np.number]), used later in
    train.py/feature_selection.py, silently excludes bool columns. If we
    leave them as bool, train.py would train on fewer columns than
    get_feature_columns() reports, and detect.py would then pass the
    full column list at scoring time - a mismatch that breaks scoring.
    Casting to int here keeps them numeric everywhere downstream."""
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"Cast {len(bool_cols)} boolean columns to int: {list(bool_cols)}")
    return df


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Dataset mode: {DATASET}")
    df = load_raw_csv(input_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
    df = normalize_labels(df)
    df = drop_duplicates(df)
    df = extract_entity(df)
    df = clean_flow_stats(df)
    df = cast_bool_columns(df)
    df["row_id"] = range(len(df))
    print(f"Final cleaned dataset: {len(df)} rows.")
    print(f"Unique entities: {df['entity'].nunique()}")
    print(df[CFG["label_col"]].value_counts())
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
