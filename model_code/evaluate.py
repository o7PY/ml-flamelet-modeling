import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def maape(y_true, y_pred):
    epsilon = 1e-6
    error = np.arctan(np.abs((y_true - y_pred) / (np.clip(np.abs(y_true), epsilon, None))))
    return np.mean(error)

def accuracy_within_tolerance(y_true, y_pred, tol=0.1):
    """
    Percentage of predictions where absolute error is ≤ tolerance.
    """
    return np.mean(np.abs(y_true - y_pred) <= tol)

def print_metrics(y_true, y_pred, label="Validation", tol=0.1):
    rmse_val = float(rmse(y_true, y_pred))
    r2_val = float(r2(y_true, y_pred))
    maape_val = float(maape(y_true, y_pred))
    acc = float(accuracy_within_tolerance(y_true, y_pred, tol))

    print(f"\n📊 {label} Metrics:")
    print(f"RMSE   = {rmse_val:.6f}")
    print(f"R²     = {r2_val:.6f}")
    print(f"MAAPE  = {maape_val:.6f}")
    print(f"Accuracy (|Δ| ≤ {tol}) = {acc*100:.2f}%")

    return {
        "split": label.lower(),
        "rmse": rmse_val,
        "r2": r2_val,
        "maape": maape_val,
        "accuracy_within_tol": acc
    }