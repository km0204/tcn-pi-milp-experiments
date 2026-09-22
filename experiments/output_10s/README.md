# Curated closed-loop results

These files report the paired nominal and stress-condition experiment with a
three-period horizon, two forecast scenarios, five replications per condition,
and a 10-second CBC time limit.

- `policy_metrics.csv`: one row per policy, condition, and random seed.
- `pwa_metrics.csv`: PWA approximation diagnostics by production stage.
- `summary.json`: complete run configuration and aggregated metrics.
- `operational_performance.png`: paper-ready closed-loop comparison.

The objective values are synthetic weighted penalties, not monetary costs.
