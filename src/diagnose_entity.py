import pandas as pd
import sys

entity = "192.168.10.25"

baseline_files = [
    "data/processed/monday_tw.csv",
    "data/processed/tuesday_tw.csv",
    "data/processed/thursday_morning_tw.csv",
    "data/processed/thursday_afternoon_tw.csv",
]
attack_file = "data/processed/wednesday_tw.csv"


print(f"--- {entity} BASELINE (training, BENIGN only) ---")
baseline = pd.concat([pd.read_csv(f) for f in baseline_files], ignore_index=True)
baseline = baseline[(baseline["entity"] == entity) & (baseline["Label"] == "BENIGN")]
print(baseline[["connections_per_minute", "distinct_ports_per_minute"]].describe())

print(f"\n--- {entity} DURING PORTSCAN ATTACK ---")
attack = pd.read_csv(attack_file)
attack_entity = attack[attack["entity"] == entity]
print(attack_entity.groupby("Label")[["connections_per_minute", "distinct_ports_per_minute"]].describe())
