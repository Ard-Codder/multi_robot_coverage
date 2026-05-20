from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np

from coverage_lab.algorithms.baselines import BaselineVoronoi
from coverage_lab.env.grid_world import GridWorld2D
from coverage_lab.types import LabScene

Mode = Literal["pure", "soft_guarded", "guarded"]
Cell = tuple[int, int]


@dataclass
class MLGoalPlanner:
    """Goal-level learned planner.

    The network selects a local goal/frontier cell, while the simulator moves the
    robot toward that target. This is intentionally higher-level than
    N/S/E/W/stop classification.
    """

    model_path: str
    mode: Mode = "soft_guarded"
    stagnation_steps: int = 20
    top_k: int = 16
    use_allocation: bool = True

    def __post_init__(self) -> None:
        import torch

        from coverage_lab.ml_planner.model import GoalCNN

        ckpt = torch.load(self.model_path, map_location="cpu")
        self._torch = torch
        self.window = int(ckpt.get("window", 31))
        self.in_ch = int(ckpt.get("in_ch", 6))
        self._model = GoalCNN(in_ch=self.in_ch, window=self.window)
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()
        self._guardrail = BaselineVoronoi()
        self._last_cov = 0.0
        self._stagnation = 0
        self._total_decisions = 0
        self._model_steps = 0
        self._fallback_steps = 0
        self._unsafe_goals = 0
        self._goal_dist_sum = 0.0
        self._allocated_steps = 0
        self._reserved_conflicts = 0

    def _grid_shape(self, scene: LabScene) -> tuple[int, int]:
        x_min, x_max, y_min, y_max = scene.bounds_xy
        res = float(scene.grid_resolution_m)
        return max(1, int(np.ceil((x_max - x_min) / res))), max(1, int(np.ceil((y_max - y_min) / res)))

    def _to_cell(self, scene: LabScene, p: np.ndarray) -> Cell:
        x_min, _, y_min, _ = scene.bounds_xy
        res = float(scene.grid_resolution_m)
        return int((float(p[0]) - x_min) / res), int((float(p[1]) - y_min) / res)

    def _cell_center(self, scene: LabScene, c: Cell) -> np.ndarray:
        x_min, _, y_min, _ = scene.bounds_xy
        res = float(scene.grid_resolution_m)
        return np.array([x_min + (c[0] + 0.5) * res, y_min + (c[1] + 0.5) * res], dtype=float)

    def _blocked(self, world: GridWorld2D, metrics=None) -> set[Cell]:
        if metrics is not None and hasattr(metrics, "blocked_cells"):
            return set(metrics.blocked_cells)
        return set(world.obstacle_cells())

    def _visited(self, metrics=None) -> set[Cell]:
        if metrics is not None and hasattr(metrics, "visited_cells"):
            return set(metrics.visited_cells)
        return set()

    def _frontier(self, *, visited: set[Cell], blocked: set[Cell], nx: int, ny: int) -> set[Cell]:
        out: set[Cell] = set()
        for cx, cy in visited:
            for nb in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if 0 <= nb[0] < nx and 0 <= nb[1] < ny and nb not in visited and nb not in blocked:
                    out.add(nb)
        return out

    def _obs_for_robot(
        self,
        *,
        robot_cell: Cell,
        other: set[Cell],
        visited: set[Cell],
        blocked: set[Cell],
        frontier: set[Cell],
        nx: int,
        ny: int,
    ) -> np.ndarray:
        w = int(self.window)
        r = w // 2
        obs = np.zeros((1, self.in_ch, w, w), dtype=np.float32)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                cell = (robot_cell[0] + dx, robot_cell[1] + dy)
                if cell[0] < 0 or cell[1] < 0 or cell[0] >= nx or cell[1] >= ny:
                    continue
                px, py = dy + r, dx + r
                if cell in visited:
                    obs[0, 0, px, py] = 1.0
                if self.in_ch >= 2 and cell in blocked:
                    obs[0, 1, px, py] = 1.0
                if self.in_ch >= 3 and cell not in blocked and cell not in visited:
                    obs[0, 2, px, py] = 1.0
                if self.in_ch >= 4 and dx == 0 and dy == 0:
                    obs[0, 3, px, py] = 1.0
                if self.in_ch >= 5 and cell in other:
                    obs[0, 4, px, py] = 1.0
                if self.in_ch >= 6 and cell in frontier:
                    obs[0, 5, px, py] = 1.0
        return obs

    def _safe(self, c: Cell, *, nx: int, ny: int, blocked: set[Cell]) -> bool:
        return 0 <= c[0] < nx and 0 <= c[1] < ny and c not in blocked

    def _local_gain(self, c: Cell, *, visited: set[Cell], blocked: set[Cell], nx: int, ny: int) -> float:
        if not self._safe(c, nx=nx, ny=ny, blocked=blocked):
            return -10.0
        gain = 0.0 if c in visited else 1.0
        for nb in ((c[0] + 1, c[1]), (c[0] - 1, c[1]), (c[0], c[1] + 1), (c[0], c[1] - 1)):
            if 0 <= nb[0] < nx and 0 <= nb[1] < ny and nb not in visited and nb not in blocked:
                gain += 0.25
        return gain

    def _goal_candidates(
        self,
        *,
        logits,
        robot_cell: Cell,
        visited: set[Cell],
        blocked: set[Cell],
        frontier: set[Cell],
        nx: int,
        ny: int,
    ) -> list[tuple[float, Cell]]:
        k = min(int(self.top_k), int(logits.numel()))
        vals, idxs = self._torch.topk(logits.flatten(), k=k)
        radius = int(self.window) // 2
        out: list[tuple[float, Cell]] = []
        for value, idx_t in zip(vals.tolist(), idxs.tolist()):
            rel_y, rel_x = divmod(int(idx_t), int(self.window))
            goal = (robot_cell[0] + rel_x - radius, robot_cell[1] + rel_y - radius)
            dist = abs(goal[0] - robot_cell[0]) + abs(goal[1] - robot_cell[1])
            frontier_bonus = 1.5 if goal in frontier else 0.0
            gain = self._local_gain(goal, visited=visited, blocked=blocked, nx=nx, ny=ny)
            score = float(value) + 1.8 * gain + frontier_bonus - 0.03 * float(dist)
            out.append((score, goal))
        return sorted(out, key=lambda item: item[0], reverse=True)

    def _allocate_greedy(
        self,
        candidates: dict[str, list[tuple[float, Cell]]],
        *,
        nx: int,
        ny: int,
        blocked: set[Cell],
    ) -> tuple[dict[str, Cell], bool]:
        assigned: dict[str, Cell] = {}
        reserved: set[Cell] = set()
        unsafe = False
        for name in sorted(candidates):
            chosen: Cell | None = None
            for _, goal in candidates[name]:
                if not self._safe(goal, nx=nx, ny=ny, blocked=blocked):
                    continue
                if goal in reserved:
                    continue
                chosen = goal
                break
            if chosen is None:
                goal = candidates[name][0][1]
                chosen = goal
                unsafe = True
                self._unsafe_goals += 1
            if chosen in reserved:
                self._reserved_conflicts += 1
            reserved.add(chosen)
            assigned[name] = chosen
        self._allocated_steps += 1
        return assigned, unsafe

    def choose_targets(self, robot_xy: Dict[str, np.ndarray], *, scene: LabScene, world: GridWorld2D, metrics=None, state=None):
        self._total_decisions += 1
        cov = float(metrics.coverage_percent()) if metrics is not None and hasattr(metrics, "coverage_percent") else 0.0
        if cov <= self._last_cov + 1e-6:
            self._stagnation += 1
        else:
            self._stagnation = 0
        self._last_cov = cov

        if self.mode == "guarded" and metrics is not None and cov < float(scene.target_coverage):
            self._fallback_steps += 1
            return self._guardrail.choose_targets(robot_xy, scene=scene, world=world, metrics=metrics)

        nx, ny = self._grid_shape(scene)
        blocked = self._blocked(world, metrics)
        visited = self._visited(metrics)
        frontier = self._frontier(visited=visited, blocked=blocked, nx=nx, ny=ny)
        current = {name: self._to_cell(scene, p) for name, p in robot_xy.items()}
        candidate_map: dict[str, list[tuple[float, Cell]]] = {}
        for name, p in robot_xy.items():
            cell = current[name]
            other = {c for n, c in current.items() if n != name}
            obs = self._obs_for_robot(
                robot_cell=cell,
                other=other,
                visited=visited,
                blocked=blocked,
                frontier=frontier,
                nx=nx,
                ny=ny,
            )
            with self._torch.no_grad():
                logits = self._model(self._torch.from_numpy(obs))[0]
            candidate_map[name] = self._goal_candidates(
                logits=logits,
                robot_cell=cell,
                visited=visited,
                blocked=blocked,
                frontier=frontier,
                nx=nx,
                ny=ny,
            )
        if self.use_allocation:
            assigned, unsafe = self._allocate_greedy(candidate_map, nx=nx, ny=ny, blocked=blocked)
        else:
            assigned = {name: candidates[0][1] for name, candidates in candidate_map.items()}
            unsafe = any(not self._safe(goal, nx=nx, ny=ny, blocked=blocked) for goal in assigned.values())
        out: dict[str, np.ndarray] = {}
        for name, goal in assigned.items():
            cell = current[name]
            self._goal_dist_sum += abs(goal[0] - cell[0]) + abs(goal[1] - cell[1])
            if not self._safe(goal, nx=nx, ny=ny, blocked=blocked):
                unsafe = True
                self._unsafe_goals += 1
            out[name] = self._cell_center(scene, goal)
        self._model_steps += 1
        if self.mode == "soft_guarded" and (unsafe or self._stagnation >= int(self.stagnation_steps)):
            self._fallback_steps += 1
            return self._guardrail.choose_targets(robot_xy, scene=scene, world=world, metrics=metrics)
        return out

    def diagnostics(self) -> dict[str, float | int | str]:
        total = max(int(self._total_decisions), 1)
        robot_decisions = max(int(self._model_steps), 1)
        return {
            "ml_goal_mode": self.mode,
            "ml_goal_total_decisions": int(self._total_decisions),
            "ml_goal_model_steps": int(self._model_steps),
            "ml_goal_fallback_steps": int(self._fallback_steps),
            "ml_goal_model_rate": float(self._model_steps / total),
            "ml_goal_fallback_rate": float(self._fallback_steps / total),
            "ml_goal_unsafe_goals": int(self._unsafe_goals),
            "ml_goal_avg_l1_distance": float(self._goal_dist_sum / robot_decisions),
            "ml_goal_allocated_steps": int(self._allocated_steps),
            "ml_goal_reserved_conflicts": int(self._reserved_conflicts),
        }
