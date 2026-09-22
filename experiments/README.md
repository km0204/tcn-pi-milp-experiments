# Synthetic temporal-surrogate and rolling-horizon experiments

This directory contains a reproducible pilot implementation of the associated
numerical study. It is not calibrated to a real plant. Plant data or validated
DES parameters must replace the synthetic constants before the results are
used as industrial evidence.

Create the environment from the project root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r experiments/requirements.txt
```

Run the three surrogate seeds:

```bash
for seed in 20260920 20260921 20260922; do
  MPLCONFIGDIR=/private/tmp/mpl-cache \
  .venv/bin/python experiments/run_pilot.py \
    --output experiments/curriculum_${seed} \
    --train-trajectories 30 --val-trajectories 12 \
    --test-trajectories 18 --trajectory-length 32 \
    --epochs 28 --skip-closed-loop --seed ${seed}
done
```

Aggregate the temporal and physics-informed comparison:

```bash
MPLCONFIGDIR=/private/tmp/mpl-cache \
.venv/bin/python experiments/aggregate_temporal.py
```

Run the closed-loop study:

```bash
MPLCONFIGDIR=/private/tmp/mpl-cache \
.venv/bin/python experiments/run_pilot.py \
  --output experiments/output_10s \
  --train-trajectories 30 --val-trajectories 12 \
  --test-trajectories 18 --trajectory-length 32 --epochs 28 \
  --closed-loop-reps 5 --closed-loop-length 15 --horizon 3 \
  --solver-time-limit 10 \
  --seed 20260920
```

Regenerate the closed-loop figure:

```bash
MPLCONFIGDIR=/tmp/mpl-cache XDG_CACHE_HOME=/tmp/codex-cache \
.venv/bin/python experiments/plot_operational.py \
  --input experiments/output_10s/policy_metrics.csv \
  --output experiments/output_10s/operational_performance.png \
  --closed-loop-length 15 --backlog-weight 12
```

`run_pilot.py` implements the three-stage discrete-event simulator, Current
MLP, parameter-matched Window MLP, Data TCN, PI-TCN, PWA fitting, MILP planning,
and closed-loop evaluation. `aggregate_temporal.py` combines the three training
seeds and produces the paper figure with uncertainty intervals.

Main outputs:

- `output_temporal_final/temporal_physics_evidence.png`
- `output_temporal_final/surrogate_metrics_all_seeds.csv`
- `output_10s/closed_loop_cost.png`
- `output_10s/operational_performance.png`
- `output_10s/policy_metrics.csv`
