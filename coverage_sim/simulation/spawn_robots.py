from __future__ import annotations

from typing import List, Tuple

import numpy as np


def create_spawn_points(num_robots: int, bounds_xy: Tuple[float, float, float, float]) -> List[np.ndarray]:
    x_min, x_max, y_min, y_max = bounds_xy
    xs = np.linspace(x_min + 1.0, x_max - 1.0, num=max(2, num_robots))
    y = 0.5 * (y_min + y_max)
    points = [np.array([float(xs[i]), float(y)], dtype=float) for i in range(num_robots)]
    return points

