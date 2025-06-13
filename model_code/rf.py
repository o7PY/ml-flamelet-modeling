import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from evaluate import print_metrics
from plots import save_pred_vs_true
import os

# 1. Load data
df = pd.read_csv("data/processed/big_dataset.csv")
input_cols = ["Z", "C", "T_inlet", "P_bar", "mdot"]
target_col = "log_omega_C"

X = df[input_cols].values
y = df[[target_col]].values  # shape: (N, 1)

# 2. Scale features and target
from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y).ravel()  # flatten for scikit-learn

# 3. Split: train (70%), val (15%), test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 4. Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 5. Predict + inverse scale
y_val_pred = rf.predict(X_val).reshape(-1, 1)
y_test_pred = rf.predict(X_test).reshape(-1, 1)

y_val_pred_inv = y_scaler.inverse_transform(y_val_pred)
y_val_true_inv = y_scaler.inverse_transform(y_val.reshape(-1, 1))

y_test_pred_inv = y_scaler.inverse_transform(y_test_pred)
y_test_true_inv = y_scaler.inverse_transform(y_test.reshape(-1, 1))

# 6. Metrics
val_stats = print_metrics(y_val_true_inv, y_val_pred_inv, label="Validation")
test_stats = print_metrics(y_test_true_inv, y_test_pred_inv, label="Test")

# 7. Save plot
os.makedirs("results/graph/rf", exist_ok=True)
save_pred_vs_true(y_val_true_inv, y_val_pred_inv, model_name="rf")
with open("results/graph/rf/rf_metrics.json", "w") as f:
    json.dump({"val": val_stats, "test": test_stats}, f, indent=2)
