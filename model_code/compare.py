# main.py (load-only version)

import os
import json
import matplotlib.pyplot as plt
import numpy as np

models = ["ann1", "ann2", "rf", "gbt"]
metric_names = ["rmse", "r2", "maape", "accuracy_within_tol"]
val_scores = {metric: [] for metric in metric_names}
test_scores = {metric: [] for metric in metric_names}

# 1. Load metrics for all models
for model in models:
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

# 2. Plot bar charts for validation and test scores
def plot_bar(metric_dict, title, filename):
    plt.figure(figsize=(8, 5))
    x = np.arange(len(models))
    width = 0.2

    for i, metric in enumerate(metric_names):
        plt.bar(x + i * width, metric_dict[metric], width, label=metric.upper())

    plt.xticks(x + width * 1.5, [m.upper() for m in models])
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/graph/{filename}")
    plt.close()

def reorganize(metric_data):
    return {metric: metric_data[metric] for metric in metric_names}

plot_bar(reorganize(val_scores), "Validation Metric Comparison", "val_metrics_bar.png")
plot_bar(reorganize(test_scores), "Test Metric Comparison", "test_metrics_bar.png")

print("\n✅ Metric comparison plots saved to results/graph/")
