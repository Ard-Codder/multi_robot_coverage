"""
Упрощённая среда Gymnasium: один «центр масс» роботов на сетке, дискретные смещения цели.
Назначение — воспроизводимый RL-прототип на CPU без Isaac; метрики близки по смыслу к coverage.
"""

from __future__ import annotations

from typing import Any, Optional, SupportsFloat, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError("Установите пакет gymnasium: pip install gymnasium") from e


class GridCoverageGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        grid_size: int = 16,
        bounds_xy: Tuple[float, float, float, float] = (-8.0, 8.0, -8.0, 8.0),
        target_coverage: float = 0.85,
        max_steps: int = 200,
        n_robots: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._bounds = bounds_xy
        self._gx = grid_size
        self._gy = grid_size
        self._target_cov = float(target_coverage)
        self._max_steps = int(max_steps)
        self._n = int(n_robots)
        self._rng = np.random.default_rng(seed)

        self._visited = np.zeros((self._gx, self._gy), dtype=np.float32)
        self._robot_cells: list[Tuple[int, int]] = []
        self._step_count = 0

        # Действие: смещение общей цели на сетке (-1..1 по x и y)
        self.action_space = spaces.MultiDiscrete([3, 3])
        flat = self._gx * self._gy
        self.observation_space = spaces.Dict(
            {
                "visited_flat": spaces.Box(low=0.0, high=1.0, shape=(flat,), dtype=np.float32),
                "goal_ix": spaces.Discrete(flat),
                "coverage": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

        self._goal_cx = self._gx // 2
        self._goal_cy = self._gy // 2

    @property
    def num_robots(self) -> int:
        return self._n

    def get_robot_positions_world(self) -> list[tuple[float, float]]:
        """Текущие позиции роботов в координатах симуляции (x, y) для экспорта / визуализации."""
        return [self._cell_to_world(cx, cy) for cx, cy in self._robot_cells]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._visited.fill(0.0)
        self._step_count = 0
        self._robot_cells = []
        for _ in range(self._n):
            cx = int(self._rng.integers(0, self._gx))
            cy = int(self._rng.integers(0, self._gy))
            self._robot_cells.append((cx, cy))
            self._visited[cx, cy] = 1.0
        self._goal_cx = self._gx // 2
        self._goal_cy = self._gy // 2
        return self._get_obs(), {}

    def _cell_to_world(self, cx: int, cy: int) -> Tuple[float, float]:
        x_min, x_max, y_min, y_max = self._bounds
        fx = (cx + 0.5) / self._gx
        fy = (cy + 0.5) / self._gy
        return x_min + fx * (x_max - x_min), y_min + fy * (y_max - y_min)

    def _move_toward_goal(self) -> None:
        gx, gy = self._goal_cx, self._goal_cy
        for i in range(self._n):
            cx, cy = self._robot_cells[i]
            dx = np.sign(gx - cx)
            dy = np.sign(gy - cy)
            if self._rng.random() < 0.15:
                dx = int(self._rng.integers(-1, 2))
            if self._rng.random() < 0.15:
                dy = int(self._rng.integers(-1, 2))
            nx = int(np.clip(cx + dx, 0, self._gx - 1))
            ny = int(np.clip(cy + dy, 0, self._gy - 1))
            self._robot_cells[i] = (nx, ny)
            self._visited[nx, ny] = 1.0

    def _coverage(self) -> float:
        return float(self._visited.mean())

    def _get_obs(self) -> dict[str, np.ndarray]:
        cov = self._coverage()
        goal_ix = self._goal_cy * self._gx + self._goal_cx
        return {
            "visited_flat": self._visited.reshape(-1).astype(np.float32),
            "goal_ix": np.array(goal_ix, dtype=np.int64),
            "coverage": np.array([cov], dtype=np.float32),
        }

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:
        ax, ay = int(action[0]) - 1, int(action[1]) - 1
        self._goal_cx = int(np.clip(self._goal_cx + ax, 0, self._gx - 1))
        self._goal_cy = int(np.clip(self._goal_cy + ay, 0, self._gy - 1))
        self._move_toward_goal()
        self._step_count += 1
        cov = self._coverage()
        reward = float(cov * 10.0 - 0.01 * self._step_count)
        terminated = cov >= self._target_cov
        truncated = self._step_count >= self._max_steps
        return self._get_obs(), reward, terminated, truncated, {}
