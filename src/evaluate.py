import pandas as pd
import sys

from config import CFG


def compute_metrics(df, label_col, benign_value, pred_col="predicted"):
    """Compute precision, recall, F1, and FPR treating any non-benign
    label as the positive (attack) class. df must already contain the
    true label and the model's prediction (-1 = anomaly, 1 = normal)."""
    is_attack = df[label_col] != benign_value
    is_flagged = df[pred_col] == -1

    tp = (is_attack & is_flagged).sum()
    fn = (is_attack & ~is_flagged).sum()
    fp = (~is_attack & is_flagged).sum()
    tn = (~is_attack & ~is_flagged).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Precision": precision, "Recall": recall,
        "F1": f1, "FPR": fpr
    }


def print_metrics(name, metrics):
    print(f"\n--- {name} ---")
    print(f"TP={metrics['TP']}  FP={metrics['FP']}  "
          f"FN={metrics['FN']}  TN={metrics['TN']}")
    print(f"Precision: {metrics['Precision']:.3f}")
    print(f"Recall:    {metrics['Recall']:.3f}")
    print(f"F1-score:  {metrics['F1']:.3f}")
    print(f"FPR:       {metrics['FPR']:.3f}")


if __name__ == "__main__":
    scored_path = sys.argv[1]
    # Optional: override which entity gets its own breakdown printed.
    # Defaults to CFG["default_target_entity"] (entity 1 for GeNIS,
    # 192.168.10.50 for CICIDS2017).
    target_entity = sys.argv[2] if len(sys.argv) > 2 else CFG["default_target_entity"]

    df = pd.read_csv(scored_path)
    label_col = CFG["label_col"]
    benign_value = CFG["benign_value"]

    print(f"Loaded {len(df)} scored rows from {scored_path}")

    overall = compute_metrics(df, label_col, benign_value)
    print_metrics("Network-wide", overall)

    # entity IDs are ints for GeNIS, strings for CICIDS2017 - cast to
    # match whatever dtype the 'entity' column actually is.
    entity_dtype = df["entity"].dtype
    try:
        target_entity_cast = entity_dtype.type(target_entity)
    except (ValueError, TypeError):
        target_entity_cast = target_entity

    entity_df = df[df["entity"] == target_entity_cast]
    if len(entity_df) > 0:
        entity_metrics = compute_metrics(entity_df, label_col, benign_value)
        print_metrics(f"Entity {target_entity}", entity_metrics)
    else:
        print(f"\nNo rows found for entity {target_entity}")
