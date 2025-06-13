import os
import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np
from plots import plot_combined_loss, plot_combined_pred_vs_true

models = ["ann1", "ann2", "rf", "gbt"]
metric_names = ["rmse", "r2", "maape", "accuracy_within_tol"]

val_scores = {metric: [] for metric in metric_names}
test_scores = {metric: [] for metric in metric_names}

# 1. Run all models
for model in models:
    print(f"\n🚀 Running {model.upper()}...")
    subprocess.run(["python3", f"model_code/{model}.py"])

    # 2. Load metrics
    metric_path = f"results/graph/{model}/{model}_metrics.json"
    with open(metric_path, "r") as f:
        data = json.load(f)

    for metric in metric_names:
        val_scores[metric].append(data["val"][metric])
        test_scores[metric].append(data["test"][metric])

# 3. Plot bar graphs
def plot_bar(metric_dict, title, filename):
    plt.figure(figsize=(8, 5))
    x = np.arange(len(models))
    width = 0.35

    val_vals = [metric_dict[m][0] for m in metric_names]
    test_vals = [metric_dict[m][1] for m in metric_names]

    for i, metric in enumerate(metric_names):
        plt.bar(x + i*width/len(metric_names), metric_dict[metric], width/len(metric_names), label=metric.upper())

    plt.xticks(x + width / 2.5, models)
    plt.title(title)
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/graph/{filename}")
    plt.close()

# Repackage for grouped plotting
def reorganize(metric_data):
    return {metric: metric_data[metric] for metric in metric_names}

plot_bar(reorganize(val_scores), "Validation Metric Comparison", "val_metrics_bar.png")
plot_bar(reorganize(test_scores), "Test Metric Comparison", "test_metrics_bar.png")

print("\n✅ Metric comparison plots saved to results/graph/")

plot_combined_loss()
plot_combined_pred_vs_true()
print("\n📊 Combined plots saved to results/graph/")
