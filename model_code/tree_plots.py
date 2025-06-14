# tree_plots.py (RF + GBT only graphs and comparison)

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from plots import plot_heatmap

# Target models
tree_models = ["rf", "gbt"]
metric_names = ["rmse", "r2", "maape", "accuracy_within_tol"]
val_scores = {metric: [] for metric in metric_names}
test_scores = {metric: [] for metric in metric_names}

# Load metrics
for model in tree_models:
    metric_path = f"results/graph/{model}/{model}_metrics.json"
    if not os.path.exists(metric_path):
        print(f"⚠️ Metrics file not found for {model}: {metric_path}")
        continue
    print(f"📂 Loading metrics for {model.upper()} from {metric_path}")
    with open(metric_path, "r") as f:
        data = json.load(f)
    for metric in metric_names:
        val_scores[metric].append(data["val"][metric])
        test_scores[metric].append(data["test"][metric])

# Bar plots

def plot_tree_bars(metric_dict, title, filename):
    x = np.arange(len(tree_models))
    width = 0.2
    plt.figure(figsize=(8, 5))
    for i, metric in enumerate(metric_names):
        plt.bar(x + i * width, metric_dict[metric], width, label=metric.upper())
    plt.xticks(x + width * 1.5, [m.upper() for m in tree_models])
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/graph/{filename}")
    plt.close()

plot_tree_bars(val_scores, "Tree Model Validation Metrics", "tree_val_metrics_bar.png")
plot_tree_bars(test_scores, "Tree Model Test Metrics", "tree_test_metrics_bar.png")

# AAPE Heatmaps only

def plot_tree_aape(model):
    path = f"results/graph/{model}/{model}_raw.npz"
    if not os.path.exists(path):
        print(f"⚠️ Missing .npz file for {model}, skipping AAPE heatmap.")
        return
    data = np.load(path)
    y_true = data["y_true_val"].flatten()
    y_pred = data["y_pred_val"].flatten()
    Z, C = data["Z_val"], data["C_val"]

    # AAPE heatmap
    eps = 1e-6
    aape = np.arctan(np.abs((y_pred - y_true) / np.clip(np.abs(y_true), eps, None)))
    plot_heatmap(Z, C, aape,
                 f"{model.upper()} AAPE Map",
                 f"results/graph/{model}/{model}_aape_heat.png",
                 cmap="inferno")

# Run per model
for model in tree_models:
    plot_tree_aape(model)

print("\n🌲 RF + GBT visualizations saved in results/graph/")
