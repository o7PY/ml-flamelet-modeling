import json
import torch
import torch.nn as nn
from data_loader import load_dataset
from evaluate import print_metrics
from plots import save_loss_plot, save_pred_vs_true
import numpy as np
from utils import set_seed
set_seed(42)

# 1. Load data
train_loader, val_loader, test_loader, X_scaler, y_scaler, X_full = load_dataset(batch_size=256)

# 2. Define deeper ANN2 model
class ANN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

model = ANN2()

# 3. Loss and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4. Train loop
EPOCHS = 100
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    running_train_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * len(xb)
    train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    running_val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            running_val_loss += loss.item() * len(xb)
    val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(val_loss)

    print(f"📉 Epoch {epoch+1:03d}: Train={train_loss:.6f}, Val={val_loss:.6f}")

# 5. Validation Evaluation
model.eval()
y_val_true, y_val_pred = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        pred = model(xb)
        y_val_true.append(yb.numpy())
        y_val_pred.append(pred.numpy())

y_val_true = np.vstack(y_val_true)
y_val_pred = np.vstack(y_val_pred)

y_val_true_inv = y_scaler.inverse_transform(y_val_true)
y_val_pred_inv = y_scaler.inverse_transform(y_val_pred)

val_stats = print_metrics(y_val_true_inv, y_val_pred_inv, label="Validation")

# 6. Test Evaluation
y_test_true, y_test_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        pred = model(xb)
        y_test_true.append(yb.numpy())
        y_test_pred.append(pred.numpy())

y_test_true = np.vstack(y_test_true)
y_test_pred = np.vstack(y_test_pred)

y_test_true_inv = y_scaler.inverse_transform(y_test_true)
y_test_pred_inv = y_scaler.inverse_transform(y_test_pred)

test_stats = print_metrics(y_test_true_inv, y_test_pred_inv, label="Test")



# For val set only:
val_indices = val_loader.dataset.indices
X_val = X_full[val_indices]
Z_val = X_val[:, 0]  # column 0 = Z
C_val = X_val[:, 1]  # column 1 = C


# 7. Save plots
save_loss_plot(train_losses, val_losses, model_name="ann2")
save_pred_vs_true(y_val_true_inv, y_val_pred_inv, model_name="ann2")
with open("results/graph/ann2/ann2_metrics.json", "w") as f:
    json.dump({"val": val_stats, "test": test_stats}, f, indent=2)
np.savez(f"results/graph/ann2/ann2_raw.npz",
         train_losses=train_losses,
         val_losses=val_losses,
         y_true_val=y_val_true_inv,
         y_pred_val=y_val_pred_inv,
         Z_val=Z_val,
         C_val=C_val)

