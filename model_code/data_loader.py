import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

def load_dataset(csv_path="data/processed/big_dataset.csv", target_col="log_omega_C", batch_size=256):
    df = pd.read_csv(csv_path)
    input_cols = ["Z", "C", "T_inlet", "P_bar", "mdot"]

    X = df[input_cols].values
    y = df[[target_col]].values

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader, X_scaler, y_scaler