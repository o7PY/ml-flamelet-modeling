import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from evaluate import print_metrics
from plots import save_pred_vs_true
import os
import joblib
from utils import set_seed
set_seed(42)

# 1. Load data
df = pd.read_csv("data/processed/big_dataset.csv")
input_cols = ["Z", "C", "T_inlet", "P_bar", "mdot"]
target_col = "log_omega_C"

X = df[input_cols].values
y = df[[target_col]].values

# 2. Standardize features and target
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y).ravel()  # flatten for sklearn

set_seed()
# 3. Split: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 4. Train Gradient Boosted Tree
gbt = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
gbt.fit(X_train, y_train)

# 5. Predict + inverse transform
y_val_pred = gbt.predict(X_val).reshape(-1, 1)
y_test_pred = gbt.predict(X_test).reshape(-1, 1)

y_val_pred_inv = y_scaler.inverse_transform(y_val_pred)
y_val_true_inv = y_scaler.inverse_transform(y_val.reshape(-1, 1))

y_test_pred_inv = y_scaler.inverse_transform(y_test_pred)
y_test_true_inv = y_scaler.inverse_transform(y_test.reshape(-1, 1))

# 6. Print Metrics
val_stats = print_metrics(y_val_true_inv, y_val_pred_inv, label="Validation")
test_stats = print_metrics(y_test_true_inv, y_test_pred_inv, label="Test")

# 7. Save plot
os.makedirs("results/graph/gbt", exist_ok=True)
save_pred_vs_true(y_val_true_inv, y_val_pred_inv, model_name="gbt")
with open("results/graph/gbt/gbt_metrics.json", "w") as f:
    json.dump({"val": val_stats, "test": test_stats}, f, indent=2)

joblib.dump(gbt, "results/graph/gbt/gbt_model.joblib")

# Save prediction data for plotting
df = pd.read_csv("data/processed/big_dataset.csv")
Z = df["Z"].values
C = df["C"].values

val_indices = X_val.shape[0]
Z_val = Z[-(val_indices + X_test.shape[0]):-X_test.shape[0]]
C_val = C[-(val_indices + X_test.shape[0]):-X_test.shape[0]]

np.savez("results/graph/gbt/gbt_raw.npz",  # or gbt_raw.npz
         y_true_val=y_val_true_inv,
         y_pred_val=y_val_pred_inv,
         Z_val=Z_val,
         C_val=C_val)
