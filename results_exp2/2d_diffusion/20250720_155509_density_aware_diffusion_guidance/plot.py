import json
import os
import os.path as osp
import pickle

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# LOAD FINAL RESULTS:
datasets = ["circle", "dino", "line", "moons"]
folders = os.listdir("./")
final_results = {}
train_info = {}


def smooth(x, window_len=10, window='hanning'):
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)
        all_results = pickle.load(open(osp.join(folder, "all_results.pkl"), "rb"))
        train_info[folder] = all_results

# CREATE LEGEND -- PLEASE FILL IN YOUR RUN NAMES HERE
# Keep the names short, as these will be in the legend.
labels = {
    "run_0": "Baseline",
    "run_1": "Fixed (0.1)",
    "run_2": "Adaptive (0.2→0)", 
    "run_3": "Smoothed (0.1→0)",
    "run_4": "Minimal (0.01→0)"
}

# Only plot these runs for clarity
runs_to_plot = ["run_0", "run_1", "run_3", "run_4"]

# Use the run key as the default label if not specified
runs = list(final_results.keys())
for run in runs:
    if run not in labels:
        labels[run] = run


# CREATE PLOTS

# Create a programmatic color palette
def generate_color_palette(n):
    cmap = plt.get_cmap('tab20')  # You can change 'tab20' to other colormaps like 'Set1', 'Set2', 'Set3', etc.
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]


# Filter runs to plot and generate color palette
runs = [r for r in final_results.keys() if r in runs_to_plot]
colors = generate_color_palette(len(runs))

# Make baseline stand out with thicker line
linewidths = [3 if r == "run_0" else 1.5 for r in runs]

# Plot 1: Line plot of training loss for each dataset across the runs with labels
fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

for j, dataset in enumerate(datasets):
    row = j // 2
    col = j % 2
    for i, run in enumerate(runs):
        mean = train_info[run][dataset]["train_losses"]
        mean = smooth(mean, window_len=25)
        axs[row, col].plot(mean, label=labels[run], color=colors[i], 
                          linewidth=linewidths[i], alpha=0.8 if run != "run_0" else 1.0)
        axs[row, col].set_title(dataset)
        axs[row, col].legend()
        axs[row, col].set_xlabel("Training Step")
        axs[row, col].set_ylabel("Loss")

plt.suptitle("Training Loss Across Different Density Guidance Strategies", y=1.02)
plt.tight_layout()
plt.savefig("train_loss.png", bbox_inches='tight', dpi=300)
plt.close()

# Plot 2: Visualize generated samples
# If there is more than 1 run, these are added as extra rows.
num_runs = len(runs)
fig, axs = plt.subplots(num_runs, 4, figsize=(14, 3 * num_runs))

for i, run in enumerate(runs):
    for j, dataset in enumerate(datasets):
        images = train_info[run][dataset]["images"]
        if num_runs == 1:
            axs[j].scatter(images[:, 0], images[:, 1], alpha=0.2, color=colors[i])
            axs[j].set_title(dataset)
        else:
            axs[i, j].scatter(images[:, 0], images[:, 1], alpha=0.2, color=colors[i])
            axs[i, j].set_title(dataset)
    if num_runs == 1:
        axs[0].set_ylabel(labels[run])
    else:
        axs[i, 0].set_ylabel(labels[run])

plt.suptitle("Generated Samples Comparison", y=1.02)
plt.tight_layout()
plt.savefig("generated_samples.png", bbox_inches='tight', dpi=300)
plt.close()

# Add KL divergence comparison plot
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(datasets))
width = 0.2

for i, run in enumerate(runs):
    kl_values = [final_results[run][d]["means"]["kl_divergence"] for d in datasets]
    ax.bar(x + i*width, kl_values, width, label=labels[run], color=colors[i])

ax.set_ylabel('KL Divergence (lower is better)')
ax.set_title('KL Divergence Comparison Across Datasets')
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(datasets)
ax.legend()
plt.savefig("kl_comparison.png", bbox_inches='tight', dpi=300)
plt.close()
