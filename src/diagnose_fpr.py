import pandas as pd
import sys

from config import CFG

label_col = CFG["label_col"]
benign_value = CFG["benign_value"]

df = pd.read_csv(sys.argv[1])

fp = df[(df[label_col] == benign_value) & (df["predicted"] == -1)]
tn = df[(df[label_col] == benign_value) & (df["predicted"] == 1)]

print(f"Total benign flows: {len(df[df[label_col] == benign_value])}")
print(f"False positives: {len(fp)}\n")

print("--- False positives by entity ---")
fp_by_entity = fp["entity"].value_counts()
total_by_entity = df[df[label_col] == benign_value]["entity"].value_counts()
fpr_by_entity = (fp_by_entity / total_by_entity).sort_values(ascending=False)
print(fpr_by_entity.head(10))

print("\n--- Anomaly score distribution: FP vs TN ---")
print("False positives (wrongly flagged):")
print(fp["anomaly_score"].describe())
print("\nTrue negatives (correctly left alone):")
print(tn["anomaly_score"].describe())
