# Physics-informed temporal surrogate and MILP planning

This repository contains a reproducible synthetic study of closed-loop
production planning for a three-stage, three-product process. It combines:

- a discrete-event simulator with demand, availability, yield, and cleaning
  uncertainty;
- current-state and window MLP baselines;
- data-only and physics-informed temporal convolutional networks (TCNs);
- a piecewise-affine approximation embedded in a mixed-integer linear program;
- deterministic, scenario, block, and rolling-horizon planning policies.

> **Research-code status.** The study is synthetic and is not calibrated to a
> real plant. The numerical constants and results must not be presented as
> industrial evidence without calibration and external validation.

## Repository layout

```text
.
├── experiments/
│   ├── run_pilot.py              # simulator, learning, PWA fitting, MILP, evaluation
│   ├── aggregate_temporal.py     # three-seed surrogate aggregation
│   ├── plot_operational.py       # closed-loop paper figure
│   ├── requirements.txt
│   ├── output_10s/               # curated 10-second MILP results
│   └── output_temporal_final/    # curated three-seed surrogate results
├── paper/
│   └── experimental_results_pilot.tex
├── tests/
│   └── test_simulator.py
└── Makefile
```

Intermediate runs are excluded by `.gitignore`; the two curated output folders
remain versioned so that the reported figures can be checked without rerunning
the full study.

## Installation

Python 3.11 or 3.12 is recommended. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r experiments/requirements.txt
```

PuLP uses the bundled CBC solver when available. Confirm the solver on a new
machine before starting the full experiment.

## Reproduction

A quick integrity check is:

```bash
make test
make smoke
```

The complete experiment can be reproduced with:

```bash
make reproduce
```

This command trains the four surrogate variants for three random seeds,
aggregates the temporal and physics-consistency results, runs the paired
closed-loop experiment with a 10-second MILP limit, and regenerates the
operational figure. The full run is computationally expensive.

Equivalent commands and individual stages are documented in
[`experiments/README.md`](experiments/README.md).

## Curated results

With the 10-second MILP limit, the mean closed-loop results were:

| Policy | Objective | Fill rate | Mean backlog | Mean WIP | Cleaning |
|---|---:|---:|---:|---:|---:|
| Heuristic | 10075.1 | 0.273 | 52.5 | 115.9 | 23.9 |
| Deterministic RH | 8400.0 | 0.329 | 45.7 | 29.0 | 10.8 |
| Scenario block | 8363.5 | 0.346 | 45.5 | 30.1 | 7.7 |
| Scenario RH | 8318.8 | 0.358 | 45.3 | 29.2 | 7.2 |

Scenario RH had the lowest mean objective value, but its paired 95% bootstrap
intervals versus deterministic RH and scenario block included zero. These are
pilot results, not evidence of statistical superiority. The objective is a
weighted penalty, not a calibrated monetary cost.

## Reproducibility notes

- Training, validation, and test data are split by complete trajectory.
- Closed-loop policies use common random numbers and identical initial states.
- Both nominal and stress conditions are evaluated.
- The default planning horizon is three periods with two scenarios.
- The default CBC time limit is 10 seconds. Solver version and hardware can
  affect wall time and incumbent solutions.
- Random seeds and all experiment settings are saved in each `summary.json`.

## Public-release checklist

Before publishing this repository, the authors should:

1. choose and add an explicit software license;
2. add author names, affiliations, and a `CITATION.cff` file;
3. confirm that the accompanying manuscript and any plant data may be shared;
4. create a tagged release and archive its DOI if required by the venue.

No license is included yet because the appropriate reuse terms must be chosen
by the authors.
