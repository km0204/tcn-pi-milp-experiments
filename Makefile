PYTHON ?= .venv/bin/python
MPL_ENV = MPLCONFIGDIR=/tmp/mpl-cache XDG_CACHE_HOME=/tmp/codex-cache
SEEDS = 20260920 20260921 20260922

.PHONY: test smoke surrogates aggregate closed-loop plot reproduce

test:
	$(MPL_ENV) $(PYTHON) -m unittest discover -s tests -v

smoke:
	$(MPL_ENV) $(PYTHON) experiments/run_pilot.py \
		--output /tmp/tcn-milp-smoke --train-trajectories 5 \
		--val-trajectories 3 --test-trajectories 3 \
		--trajectory-length 20 --epochs 1 --skip-closed-loop \
		--seed 20260920

surrogates:
	@for seed in $(SEEDS); do \
		$(MPL_ENV) $(PYTHON) experiments/run_pilot.py \
			--output experiments/curriculum_$$seed \
			--train-trajectories 30 --val-trajectories 12 \
			--test-trajectories 18 --trajectory-length 32 \
			--epochs 28 --skip-closed-loop --seed $$seed || exit 1; \
	done

aggregate:
	$(MPL_ENV) $(PYTHON) experiments/aggregate_temporal.py

closed-loop:
	$(MPL_ENV) $(PYTHON) experiments/run_pilot.py \
		--output experiments/output_10s \
		--train-trajectories 30 --val-trajectories 12 \
		--test-trajectories 18 --trajectory-length 32 --epochs 28 \
		--closed-loop-reps 5 --closed-loop-length 15 --horizon 3 \
		--solver-time-limit 10 --seed 20260920

plot:
	$(MPL_ENV) $(PYTHON) experiments/plot_operational.py \
		--input experiments/output_10s/policy_metrics.csv \
		--output experiments/output_10s/operational_performance.png \
		--closed-loop-length 15 --backlog-weight 12

reproduce: surrogates aggregate closed-loop plot
