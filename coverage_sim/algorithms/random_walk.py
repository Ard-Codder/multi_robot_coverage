from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from coverage_sim.robots.robot_state import RobotState


Bounds = Tuple[float, float, float, float]


@dataclass
class RandomWalkCoverage:
    bounds_xy: Bounds
    reach_threshold_m: float = 0.4
    seed: int | None = 42

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def choose_targets(self, robot_states: Dict[str, RobotState]) -> Dict[str, np.ndarray]:
        targets: Dict[str, np.ndarray] = {}
        for name, state in robot_states.items():
            if self._is_target_reached(state.position, state.target):
                targets[name] = self._sample_random_point()
            else:
                targets[name] = state.target.copy()
        return targets

    def _sample_random_point(self) -> np.ndarray:
        x_min, x_max, y_min, y_max = self.bounds_xy
        return np.array(
            [
                self.rng.uniform(x_min, x_max),
                self.rng.uniform(y_min, y_max),
            ],
            dtype=float,
        )

    def _is_target_reached(self, current_xy: np.ndarray, target_xy: np.ndarray) -> bool:
        return float(np.linalg.norm(target_xy - current_xy)) <= self.reach_threshold_m

