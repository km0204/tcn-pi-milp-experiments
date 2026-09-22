#!/usr/bin/env python3
"""Aggregate the three temporal/physics training seeds into one paper figure."""

import argparse
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[20260920, 20260921, 20260922])
    parser.add_argument("--input-prefix", default="curriculum_")
    parser.add_argument("--output", type=Path, default=ROOT / "output_temporal_final")
    return parser.parse_args()


args = parse_args()
SEEDS = tuple(args.seeds)
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)

sur_parts, roll_parts = [], []
for seed in SEEDS:
    folder = ROOT / f"{args.input_prefix}{seed}"
    sur = pd.read_csv(folder / "surrogate_metrics.csv")
    roll = pd.read_csv(folder / "rollout_by_horizon.csv")
    sur["seed"], roll["seed"] = seed, seed
    sur_parts.append(sur); roll_parts.append(roll)

sur = pd.concat(sur_parts, ignore_index=True)
roll = pd.concat(roll_parts, ignore_index=True)
sur.to_csv(OUT / "surrogate_metrics_all_seeds.csv", index=False)
roll.to_csv(OUT / "rollout_by_horizon_all_seeds.csv", index=False)

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
styles = [
    ("current_mlp", "Current MLP", "#7F7F7F", "o"),
    ("window_mlp", "Window MLP", "#ED7D31", "s"),
    ("data_tcn", "Data TCN", "#4472C4", "^"),
    ("pi_tcn", "PI-TCN", "#70AD47", "D"),
]
for key, label, color, marker in styles:
    g = roll[roll.model == key].groupby("horizon").wip_nrmse
    mean, sem = g.mean(), g.sem()
    axes[0].plot(mean.index, mean.values, marker=marker, color=color, label=label)
    axes[0].fill_between(mean.index, mean - 1.96 * sem, mean + 1.96 * sem,
                         color=color, alpha=0.12, linewidth=0)
axes[0].set_xlabel("Rollout horizon")
axes[0].set_ylabel("WIP NRMSE")
axes[0].set_title("(a) Temporal rollout accuracy")
axes[0].legend(frameon=False, fontsize=8)

physical = (sur[sur.model.isin(["data_tcn", "pi_tcn"])]
            .groupby("model")[["ood_physical_violation_pct", "monotonicity_violation_pct"]]
            .agg(["mean", "sem"]))
order = ["data_tcn", "pi_tcn"]
x = np.arange(2); width = 0.34
for j, (metric, label, color) in enumerate([
    ("ood_physical_violation_pct", "OOD physical violation", "#C00000"),
    ("monotonicity_violation_pct", "Monotonicity violation", "#FFC000"),
]):
    means = np.array([physical.loc[m, (metric, "mean")] for m in order])
    errs = 1.96 * np.array([physical.loc[m, (metric, "sem")] for m in order])
    axes[1].bar(x + (j - 0.5) * width, means, width, yerr=errs, capsize=3,
                label=label, color=color)
axes[1].set_xticks(x, ["Data TCN", "PI-TCN"])
axes[1].set_ylabel("Violation rate (%)")
axes[1].set_title("(b) Physical consistency under OOD stress")
axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "temporal_physics_evidence.png", dpi=300)
plt.close(fig)

mean = sur.groupby("model").mean(numeric_only=True)
summary = {
    "seeds": list(SEEDS),
    "mean_metrics": mean.round(4).to_dict("index"),
    "tcn_vs_current_rollout_reduction_pct": round(
        100 * (1 - mean.loc["data_tcn", "rollout_wip_nrmse"] /
               mean.loc["current_mlp", "rollout_wip_nrmse"]), 2),
    "tcn_vs_window_rollout_reduction_pct": round(
        100 * (1 - mean.loc["data_tcn", "rollout_wip_nrmse"] /
               mean.loc["window_mlp", "rollout_wip_nrmse"]), 2),
    "pi_vs_data_ood_violation_reduction_pct": round(
        100 * (1 - mean.loc["pi_tcn", "ood_physical_violation_pct"] /
               mean.loc["data_tcn", "ood_physical_violation_pct"]), 2),
    "pi_vs_data_monotonicity_reduction_pct": round(
        100 * (1 - mean.loc["pi_tcn", "monotonicity_violation_pct"] /
               mean.loc["data_tcn", "monotonicity_violation_pct"]), 2),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
