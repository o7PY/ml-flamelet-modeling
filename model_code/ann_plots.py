import os
import json
import numpy as np
import matplotlib.pyplot as plt
from plots import (
    plot_combined_loss,
    plot_combined_pred_vs_true,
    plot_heatmaps_for_model,
    plot_heatmap
)

# Models to evaluate
ann_models = ["ann1", "ann2"]
metric_names = ["rmse", "r2", "maape", "accuracy_within_tol"]
val_scores = {metric: [] for metric in metric_names}
test_scores = {metric: [] for metric in metric_names}

# Load metrics
for model in ann_models:
    metric_path = f"results/graph/{model}/{model}_metrics.json"
    with open(metric_path, "r") as f:
        data = json.load(f)
    for metric in metric_names:
        val_scores[metric].append(data["val"][metric])
        test_scores[metric].append(data["test"][metric])

# Plot bar graphs for ANN1 and ANN2

def plot_ann_bars(metric_dict, title, filename):
    x = np.arange(len(ann_models))
    width = 0.2
    plt.figure(figsize=(8, 5))
    for i, metric in enumerate(metric_names):
        plt.bar(x + i * width, metric_dict[metric], width, label=metric.upper())
    plt.xticks(x + width * 1.5, [m.upper() for m in ann_models])
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/graph/{filename}")
    plt.close()

plot_ann_bars(val_scores, "ANN Validation Metrics", "ann_val_metrics_bar.png")
plot_ann_bars(test_scores, "ANN Test Metrics", "ann_test_metrics_bar.png")

# Plot combined loss and prediction scatter
plot_combined_loss(model_names=ann_models, output="results/graph/ann_combined_loss.png")
plot_combined_pred_vs_true(model_names=ann_models, output="results/graph/ann_combined_pred_vs_true.png")

# Plot AAPE heatmaps for ANN1 and ANN2

def plot_ann_aape_heatmap(model):
    data = np.load(f"results/graph/{model}/{model}_raw.npz")
    y_true = data["y_true_val"].flatten()
    y_pred = data["y_pred_val"].flatten()
    Z, C = data["Z_val"], data["C_val"]
    eps = 1e-6
    aape = np.arctan(np.abs((y_pred - y_true) / np.clip(np.abs(y_true), eps, None)))
    plot_heatmap(Z, C, aape,
                 f"{model.upper()} AAPE Map",
                 f"results/graph/{model}/{model}_aape_heat.png",
                 cmap="inferno")

for model in ann_models:
    plot_heatmaps_for_model(model)  # true, pred, error
    plot_ann_aape_heatmap(model)    # aape

print("\n✅ All ANN-related plots and metrics generated!")
