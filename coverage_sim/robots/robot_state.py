from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RobotState:
    name: str
    position: np.ndarray
    target: np.ndarray
    travelled_distance: float = 0.0
    last_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))

    def update_position(self, new_position: np.ndarray) -> None:
        self.travelled_distance += float(np.linalg.norm(new_position - self.position))
        self.last_position = self.position.copy()
        self.position = new_position.copy()

