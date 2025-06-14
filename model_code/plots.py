import matplotlib.pyplot as plt
import numpy as np
import os

def save_loss_plot(train_losses, val_losses, model_name, output_dir="results/graph"):
    os.makedirs(f"{output_dir}/{model_name}", exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name.upper()} Training Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}/{model_name}_loss_curve.png")
    plt.close()

def save_pred_vs_true(y_true, y_pred, model_name, output_dir="results/graph"):
    os.makedirs(f"{output_dir}/{model_name}", exist_ok=True)
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{model_name.upper()} Predictions vs Ground Truth")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}/{model_name}_pred_vs_true.png")
    plt.close()

def plot_combined_loss(model_names=["ann1", "ann2"], output="results/graph/combined_loss.png"):
    plt.figure()
    for model in model_names:
        file = f"results/graph/{model}/{model}_raw.npz"
        if os.path.exists(file):
            data = np.load(file)
            plt.plot(data["train_losses"], label=f"{model.upper()} Train")
            plt.plot(data["val_losses"], label=f"{model.upper()} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Combined Loss Curves")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()

def plot_combined_pred_vs_true(model_names=["ann1", "ann2", "rf", "gbt"], output="results/graph/combined_pred_vs_true.png"):
    plt.figure()
    for model in model_names:
        file = f"results/graph/{model}/{model}_raw.npz"
        if os.path.exists(file):
            data = np.load(file)
            y_true = data["y_true_val"]
            y_pred = data["y_pred_val"]
            plt.scatter(y_true, y_pred, label=model.upper(), alpha=0.4, s=10)
    lims = plt.gca().get_xlim()
    plt.plot(lims, lims, 'k--', alpha=0.6)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Combined Predictions vs Ground Truth")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()

def plot_heatmap(Z, C, values, title, output, cmap="plasma"):
    from scipy.interpolate import griddata

    # Grid
    xi = np.linspace(Z.min(), Z.max(), 200)
    yi = np.linspace(C.min(), C.max(), 200)
    zi = griddata((Z, C), values, (xi[None, :], yi[:, None]), method="linear")

    # Plot
    plt.figure(figsize=(6, 5))
    cp = plt.contourf(xi, yi, zi, 100, cmap=cmap)
    plt.colorbar(cp)
    plt.xlabel("Z")
    plt.ylabel("C")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()

def plot_heatmaps_for_model(model):
    file = f"results/graph/{model}/{model}_raw.npz"
    data = np.load(file)
    Z, C = data["Z_val"], data["C_val"]
    y_true = data["y_true_val"].flatten()
    y_pred = data["y_pred_val"].flatten()
    err = y_pred - y_true

    plot_heatmap(Z, C, y_true, f"{model.upper()} True log(ω̇C)", f"results/graph/{model}/{model}_true_heat.png")
    plot_heatmap(Z, C, y_pred, f"{model.upper()} Predicted log(ω̇C)", f"results/graph/{model}/{model}_pred_heat.png")
    plot_heatmap(Z, C, err, f"{model.upper()} Prediction Error", f"results/graph/{model}/{model}_error_heat.png")
