import pandas as pd
import numpy as np
import sys


def load_raw_csv(path):
    """Load the CSV and strip whitespace from column names.
    CICIDS2017 CSVs have leading spaces in most headers, e.g.
    ' Flow Duration' instead of 'Flow Duration'. Some files (notably
    Thursday-Morning-WebAttacks) contain a non-UTF-8 byte (0x96) in
    the Label column, so we read as latin1 - this maps every byte
    1-to-1 to a character instead of failing, and keeps \x96 as the
    literal character normalize_labels() already knows how to fix."""
    df = pd.read_csv(path, low_memory=False, encoding="latin1")
    df.columns = df.columns.str.strip()
    return df




def extract_entity(df):
    """Figure out which IP in each flow is the internal host, since
    CICFlowMeter assigns Source/Destination based on who sent the first
    packet - not based on internal vs external. So a given internal
    host's traffic is split across both columns. We fix that by adding
    an 'entity' column that always points to the internal host.

    If both sides are internal (two internal hosts talking), the flow
    is counted for both entities, since it's real traffic for both."""
    internal_prefix = "192.168.10."

    src_internal = df["Source IP"].str.startswith(internal_prefix)
    dst_internal = df["Destination IP"].str.startswith(internal_prefix)

    src_rows = df[src_internal].copy()
    src_rows["entity"] = src_rows["Source IP"]

    dst_rows = df[dst_internal].copy()
    dst_rows["entity"] = dst_rows["Destination IP"]

    combined = pd.concat([src_rows, dst_rows], ignore_index=True)

    dropped = len(df) - (src_internal | dst_internal).sum()
    print(f"Dropped {dropped} flows with neither side internal.")

    return combined


def clean_flow_stats(df):
    """Fix inf and NaN values in the numeric columns. Flow Bytes/s and
    Flow Packets/s can become 'inf' when Flow Duration is 0, since you
    can't divide by zero - CICFlowMeter still writes 'inf' into the file
    when that happens."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=numeric_cols)
    print(f"Dropped {before - len(df)} rows with NaN/inf values.")
    return df
def normalize_labels(df):
    """Clean up the Label column - remove extra spaces and fix a known
    broken character that shows up in some attack labels."""
    df["Label"] = df["Label"].astype(str).str.strip().str.replace("\x96", "-", regex=False)
    return df
def drop_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows.")
    return df



if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    df = load_raw_csv(input_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
    df = normalize_labels(df)
    df = drop_duplicates(df)
    df = extract_entity(df)
    df = clean_flow_stats(df)
    df["row_id"] = range(len(df))
    print(f"Final cleaned dataset: {len(df)} rows.")
    print(f"Unique entities: {df['entity'].nunique()}")
    print(df["Label"].value_counts())
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


