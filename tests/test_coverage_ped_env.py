import numpy as np

from coverage_sim.rl.coverage_ped_env import CoveragePedRLEnv


def test_coverage_ped_env_reset_step() -> None:
    env = CoveragePedRLEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    a = env.action_space.sample()
    obs2, r, term, trunc, info = env.step(a)
    assert obs2.shape == env.observation_space.shape
    assert "coverage" in info
    assert np.isfinite(r)


def test_action_delta_mapping() -> None:
    from coverage_sim.rl.coverage_ped_env import _action_to_delta

    assert _action_to_delta(0) == (-1, -1)
    assert _action_to_delta(4) == (0, 0)
    assert _action_to_delta(8) == (1, 1)
