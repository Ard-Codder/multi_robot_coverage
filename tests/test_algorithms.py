import numpy as np

from coverage_sim.algorithms.random_walk import RandomWalkCoverage
from coverage_sim.robots.robot_state import RobotState


def test_random_walk_target_in_bounds() -> None:
    algo = RandomWalkCoverage(bounds_xy=(-5.0, 5.0, -3.0, 3.0), seed=1)
    state = RobotState(
        name="robot_0",
        position=np.array([0.0, 0.0], dtype=float),
        target=np.array([0.0, 0.0], dtype=float),
        last_position=np.array([0.0, 0.0], dtype=float),
    )
    targets = algo.choose_targets({"robot_0": state})
    x, y = targets["robot_0"]
    assert -5.0 <= float(x) <= 5.0
    assert -3.0 <= float(y) <= 3.0

