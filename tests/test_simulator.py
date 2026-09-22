import unittest

import numpy as np

from experiments.run_pilot import (
    P,
    S,
    deterministic_scenario,
    execute_period,
    exogenous_path,
    initial_state,
)


class SimulatorTest(unittest.TestCase):
    def test_deterministic_scenario_shapes(self) -> None:
        horizon = 3
        scenario = deterministic_scenario(t=0, horizon=horizon, stress=False)
        self.assertEqual(scenario["demand"].shape, (horizon, P))
        self.assertEqual(scenario["availability"].shape, (horizon, S))
        self.assertEqual(scenario["yield"].shape, (horizon, S, P))
        self.assertEqual(scenario["clean_mult"].shape, (horizon, S))

    def test_one_period_execution_is_nonnegative_and_cost_balances(self) -> None:
        state = initial_state(1.0)
        exogenous = exogenous_path(length=1, seed=7, stress=False)
        release = np.ones(P)
        active = np.arange(S) % P
        next_state, info = execute_period(
            state, release, active, exogenous, t=0, rng=np.random.default_rng(8))

        for values in (next_state.wip, next_state.inventory, next_state.backlog):
            self.assertTrue(np.all(values >= 0.0))
        component_sum = sum(
            info[name]
            for name in ("inventory_cost", "backlog_cost", "wip_cost", "cleaning_cost")
        )
        self.assertAlmostEqual(info["total_cost"], component_sum)


if __name__ == "__main__":
    unittest.main()
