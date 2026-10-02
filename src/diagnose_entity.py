import pandas as pd
import sys

from config import CFG

label_col = CFG["label_col"]
benign_value = CFG["benign_value"]

# Usage: python3 diagnose_entity.py <baseline_file1,baseline_file2,...> <attack_file> [entity]
baseline_files = sys.argv[1].split(",")
attack_file = sys.argv[2]
entity_arg = sys.argv[3] if len(sys.argv) > 3 else CFG["default_target_entity"]

baseline_all = pd.concat([pd.read_csv(f) for f in baseline_files], ignore_index=True)
entity_dtype = baseline_all["entity"].dtype
try:
    entity = entity_dtype.type(entity_arg)
except (ValueError, TypeError):
    entity = entity_arg

print(f"--- Entity {entity} BASELINE (training, benign only) ---")
baseline = baseline_all[(baseline_all["entity"] == entity) & (baseline_all[label_col] == benign_value)]
print(baseline[["connections_per_minute", "distinct_ports_per_minute"]].describe())

print(f"\n--- Entity {entity} DURING ATTACK TRAFFIC ---")
attack = pd.read_csv(attack_file)
attack_entity = attack[attack["entity"] == entity]
print(attack_entity.groupby(label_col)[["connections_per_minute", "distinct_ports_per_minute"]].describe())
