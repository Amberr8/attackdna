import pandas as pd
import re
import sys

from config import CFG

SCAN_FEATURES = [
    "connections_per_minute", "distinct_ports_per_minute"
]

# Timing/latency-related Argus fields (duration, inter-packet timing,
# round-trip time, jitter). Verified against the actual GeNIS column
# list, not ported blind from CICFlowMeter's names.
TIMING_FEATURES = [
    "Dur", "RunTime", "Mean", "Min", "Max", "Sum",
    "TcpRtt", "SynAck", "AckDat",
    "SIntPkt", "DIntPkt", "SIntPktMin", "SIntPktMax", "SIntPktAct",
    "DIntPktMin", "DIntPktMax", "DIntPktAct", "SIntPktIdl",
    "SrcJitter", "SrcJitAct", "DstJitter", "DstJitAct",
]

# Volume-related Argus fields (bytes, packets, rate, loss).
VOLUME_FEATURES = [
    "TotBytes", "SrcBytes", "DstBytes", "TotAppByte", "SAppBytes", "DAppBytes",
    "TotPkts", "SrcPkts", "DstPkts",
    "Rate", "SrcRate", "DstRate", "Load", "SrcLoad", "DstLoad",
    "Loss", "SrcLoss", "DstLoss", "pLoss", "PCRatio",
    "sMeanPktSz", "dMeanPktSz", "sMinPktSz", "dMinPktSz", "sMaxPktSz", "dMaxPktSz",
]


def parse_explanation(explanation):
    """Break 'FeatureA is 3.2x higher than normal (z=4.1); FeatureB is ...'
    into a list of (feature_name, z_score) pairs, instead of only
    looking at the first one."""
    if not isinstance(explanation, str):
        return []
    parsed = []
    for part in explanation.split("; "):
        if " is " not in part:
            continue
        feature = part.split(" is ")[0].strip()
        match = re.search(r"z=([\d.]+)", part)
        z = float(match.group(1)) if match else 0.0
        parsed.append((feature, z))
    return parsed


def map_to_mitre(explanation):
    """Score ALL features mentioned in the explanation (not just the
    top one) by category, and classify based on which category has
    the stronger combined evidence. Still fully rule-based and
    traceable - just using more of the available evidence per anomaly."""
    features = parse_explanation(explanation)

    if not features:
        return ("T1499 - Network Denial of Service (default)",
                "Unclassified", 0.0, 0.0)

    timing_score = sum(z for f, z in features if f in TIMING_FEATURES)
    volume_score = sum(z for f, z in features if f in VOLUME_FEATURES)
    scan_score = sum(z for f, z in features if f in SCAN_FEATURES)

    if scan_score > 0 and scan_score >= max(timing_score, volume_score):
        return ("T1046 - Network Service Scanning",
                f"High connection/port-diversity burst pattern - scan "
                f"evidence ({scan_score:.1f}) dominates.",
                timing_score, volume_score)
    elif timing_score == 0 and volume_score == 0:
        return ("T1499 - Network Denial of Service (general)",
                "Anomalous behavior detected but does not match a "
                "specific known DoS sub-pattern - recommend manual "
                "analyst review.", timing_score, volume_score)
    elif timing_score > volume_score:
        return ("T1499.002 - Endpoint DoS: Service Exhaustion Flood",
                f"Slow-rate resource exhaustion pattern - timing evidence "
                f"({timing_score:.1f}) outweighs volume evidence "
                f"({volume_score:.1f}) across the flagged features.",
                timing_score, volume_score)
    else:
        return ("T1498 - Network Denial of Service",
                f"High-volume flood pattern - volume evidence "
                f"({volume_score:.1f}) outweighs timing evidence "
                f"({timing_score:.1f}) across the flagged features.",
                timing_score, volume_score)


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    label_col = CFG["label_col"]

    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path)

    print(f"Mapping {len(df)} anomalies to MITRE ATT&CK techniques...")

    mitre_ids, mitre_reasons, timing_scores, volume_scores = [], [], [], []

    for explanation in df["explanation"]:
        technique, reason, t_score, v_score = map_to_mitre(explanation)
        mitre_ids.append(technique)
        mitre_reasons.append(reason)
        timing_scores.append(t_score)
        volume_scores.append(v_score)

    df["mitre_technique"] = mitre_ids
    df["mitre_reasoning"] = mitre_reasons
    df["timing_evidence_score"] = timing_scores
    df["volume_evidence_score"] = volume_scores

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    print("\n--- MITRE technique distribution ---")
    print(df["mitre_technique"].value_counts())

    print("\n--- Cross-check: MITRE mapping vs true attack label ---")
    print(pd.crosstab(df[label_col], df["mitre_technique"]))
