import numpy as np
import pytest

pytest.importorskip("gymnasium")

from coverage_sim.rl.grid_coverage_env import GridCoverageGymEnv  # noqa: E402


def test_grid_coverage_gym_reset_step() -> None:
    env = GridCoverageGymEnv(max_steps=5, seed=0)
    obs, _ = env.reset(seed=0)
    assert "visited_flat" in obs
    assert env.num_robots == 3
    pos = env.get_robot_positions_world()
    assert len(pos) == 3
    assert all(len(p) == 2 for p in pos)
    a = np.array([1, 1], dtype=np.int64)
    _obs, _reward, _term, _trunc, _ = env.step(a)
