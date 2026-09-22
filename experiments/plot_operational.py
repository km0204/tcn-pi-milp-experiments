#!/usr/bin/env python3
"""Create the paper's two-panel closed-loop operational comparison."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "output_10s" / "policy_metrics.csv",
        help="Closed-loop policy metrics CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output_10s" / "operational_performance.png",
        help="Destination PNG path.",
    )
    parser.add_argument("--closed-loop-length", type=int, default=15)
    parser.add_argument("--backlog-weight", type=float, default=12.0)
    return parser.parse_args()


args = parse_args()
data = pd.read_csv(args.input)
# Older result files store cumulative backlog cost but not the corresponding
# physical backlog level.  The experiment uses a backlog weight of 12 and a
# 15-period closed-loop horizon, so recover the common, policy-independent
# indicator for plotting.
if "mean_backlog" not in data.columns:
    data["mean_backlog"] = data["backlog_cost"] / (
        args.backlog_weight * args.closed_loop_length)
order = ["heuristic", "deterministic_rh", "scenario_block", "scenario_rh"]
labels = ["Heuristic", "Deterministic RH", "Scenario block", "Scenario RH"]
colors = ["#A5A5A5", "#5B9BD5", "#ED7D31", "#70AD47"]

rng = np.random.default_rng(20260920)
cost_mean, cost_lo, cost_hi = [], [], []
for policy in order:
    values = data.loc[data.policy == policy, "total_cost"].to_numpy()
    boots = np.array([rng.choice(values, len(values), replace=True).mean()
                      for _ in range(20000)])
    cost_mean.append(values.mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    cost_lo.append(lo); cost_hi.append(hi)

mean = data.groupby("policy").mean(numeric_only=True)

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.65), gridspec_kw={"width_ratios": [1.0, 1.25]})
x = np.arange(len(order))
err = np.vstack([np.array(cost_mean) - np.array(cost_lo),
                 np.array(cost_hi) - np.array(cost_mean)])
axes[0].bar(x, cost_mean, yerr=err, capsize=4, color=colors, edgecolor="white")
axes[0].set_xticks(x, labels, rotation=18, ha="right")
axes[0].set_ylabel("Mean cumulative objective value")
axes[0].set_title("(a) Realized closed-loop objective")

metric_keys = ["fill_rate", "mean_backlog", "mean_wip", "cleaning_time"]
metric_labels = ["Fill rate $\\uparrow$", "Mean backlog $\\downarrow$",
                 "Mean WIP $\\downarrow$", "Cleaning $\\downarrow$"]
raw = np.array([[mean.loc[p, k] for k in metric_keys] for p in order], dtype=float)
score = np.full_like(raw, np.nan)
for j in range(raw.shape[1]):
    valid = np.isfinite(raw[:, j])
    lo, hi = np.nanmin(raw[:, j]), np.nanmax(raw[:, j])
    scaled = (raw[valid, j] - lo) / max(hi - lo, 1e-9)
    score[valid, j] = scaled if j == 0 else 1.0 - scaled
cmap = plt.cm.RdYlGn.copy(); cmap.set_bad("#E6E6E6")
heat = axes[1].imshow(np.ma.masked_invalid(score), cmap=cmap, vmin=0, vmax=1, aspect="auto")
axes[1].set_xticks(np.arange(len(metric_labels)), metric_labels, rotation=18, ha="right")
axes[1].set_yticks(np.arange(len(labels)), labels)
axes[1].set_title("(b) Operational indicators (green is better)")
axes[1].grid(False)
for i in range(raw.shape[0]):
    for j in range(raw.shape[1]):
        if not np.isfinite(raw[i, j]):
            text = "N/A"
        elif j == 0:
            text = f"{raw[i, j]:.3f}"
        else:
            text = f"{raw[i, j]:.1f}"
        axes[1].text(j, i, text, ha="center", va="center", fontsize=9,
                     color="black", fontweight="semibold")
cbar = fig.colorbar(heat, ax=axes[1], fraction=0.035, pad=0.025)
cbar.set_ticks([0.0, 0.5, 1.0], labels=["Worst", "Middle", "Best"])
cbar.set_label("Relative score within each metric", fontsize=8)
cbar.ax.tick_params(labelsize=8)

fig.tight_layout()
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=300, bbox_inches="tight")
plt.close(fig)
print(args.output)
