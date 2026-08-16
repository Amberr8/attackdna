import pandas as pd
import numpy as np
import sys

df = pd.read_csv(sys.argv[1])
df = df[df["Label"] == "BENIGN"]
non_features = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "Source Port", "Label", "entity"]
feature_cols = [c for c in df.columns if c not in non_features]

X = df[feature_cols].select_dtypes(include=[np.number])
variances = X.var().sort_values()
print("--- Lowest-variance features (candidates to drop) ---")
print(variances.head(10))

corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr = [(col, row, upper.loc[row, col]) for col in upper.columns for row in upper.index if upper.loc[row, col] > 0.95]
print(f"\n--- {len(high_corr)} feature pairs with correlation > 0.95 (redundant) ---")
for a, b, v in high_corr[:15]:
    print(f"{a} <-> {b}: {v:.3f}")
