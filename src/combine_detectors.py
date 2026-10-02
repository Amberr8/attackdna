import pandas as pd
import sys
import numpy as np

from config import CFG

if_path = sys.argv[1]
scan_path = sys.argv[2]
output_path = sys.argv[3]
label_col = CFG["label_col"]
benign_value = CFG["benign_value"]

if_df = pd.read_csv(if_path)
scan_df = pd.read_csv(scan_path)[["row_id", "scan_flag"]]
df = if_df.merge(scan_df, on="row_id", how="left")
if df["scan_flag"].isna().any():
    print(f"Warning: {df['scan_flag'].isna().sum()} rows had no scan_flag match")
df["scan_flag"] = df["scan_flag"].fillna(False)

df["final_flag"] = (df["predicted"] == -1) | (df["scan_flag"] == True)
df["predicted"] = np.where(df["final_flag"], -1, 1)

is_attack = df[label_col] != benign_value
tp = (is_attack & df["final_flag"]).sum()
fn = (is_attack & ~df["final_flag"]).sum()
fp = (~is_attack & df["final_flag"]).sum()
tn = (~is_attack & ~df["final_flag"]).sum()

precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

print(f"Combined system: TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")

df.to_csv(output_path, index=False)
