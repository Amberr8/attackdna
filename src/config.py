"""
Central configuration for AttackDNA.

Change DATASET below to switch which dataset-specific settings the rest
of the pipeline uses. Every script that needs a column name, the benign
label value, or the internal-host rule should import CFG from here
instead of hard-coding a value, so the same code works on either dataset.
"""

DATASET = "genis"  # "genis" or "cicids2017"

DATASET_CONFIGS = {
    "cicids2017": {
        "src_ip_col": "Source IP",
        "dst_ip_col": "Destination IP",
        "label_col": "Label",
        "benign_value": "BENIGN",
        "timestamp_col": "Timestamp",
        "time_col": None,
        "offset_bucket_size": None,
        "dport_col": "Destination Port",
        # CICIDS2017 has real internal (192.168.10.x) vs external hosts.
        "default_target_entity": "192.168.10.50",
        "internal_prefix": "192.168.10.",
    },
    "genis": {
        "src_ip_col": "Ssaddr",
        "default_target_entity": 1,
        "dst_ip_col": "Sdaddr",
        "label_col": "CategoryLabel",
        "benign_value": "benign",
        # GeNIS has no real wall-clock timestamp column. "Offset" is an
        # Argus record-position value (confirmed empirically: multiple
        # flows lasting 28-40s share the identical Offset value, which
        # is impossible if Offset were elapsed seconds). It's only a
        # rough capture-order proxy, not real time - so time_window_features.py
        # buckets by Offset ranges instead of by real seconds. This is a
        # documented substitution, not literal per-minute bucketing.
        "timestamp_col": None,
        "time_col": "Offset",
        # Chosen empirically: bucket_size=10000 gives clear separation
        # between benign (median 1 distinct port/bucket) and recon/scan
        # traffic (median 21 distinct ports/bucket) on the GeNIS test set.
        "offset_bucket_size": 10000,
        "dport_col": "Dport",
        # GeNIS is a closed testbed; every host ID is "internal", there
        # is no subnet to filter external traffic out with.
        "internal_prefix": None,
    },
}

CFG = DATASET_CONFIGS[DATASET]
