from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Set, Tuple

import numpy as np

from coverage_lab.env.grid_world import GridWorld2D
from coverage_lab.types import LabScene
from coverage_lab.algorithms.baselines import BaselineVoronoi

Mode = Literal["pure", "soft_guarded", "guarded"]


@dataclass
class MLGuidedPlanner:
    """Uses a trained classifier to choose next discrete move from local visited patch."""

    model_path: str
    window: int = 9
    stagnation_steps: int = 16
    mode: Mode = "soft_guarded"

    def __post_init__(self) -> None:
        import torch

        from coverage_lab.ml_planner.model import SmallCNN

        self._torch = torch
        self._device = torch.device("cpu")
        ckpt = torch.load(self.model_path, map_location="cpu")
        self._input_mode = str(ckpt.get("input_mode", "legacy"))
        self._model_variant = str(ckpt.get("model_variant", "small"))
        state_dict = ckpt["state_dict"]
        conv0 = state_dict.get("net.0.weight")
        in_ch = int(ckpt.get("in_ch", conv0.shape[1] if conv0 is not None else 1))
        self._model = SmallCNN(in_ch=in_ch, variant=self._model_variant)
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()

        self._visited = set()
        self._last_cov = 0.0
        self._stagnation = 0
        self._guardrail = BaselineVoronoi()
        self._total_decisions = 0
        self._model_steps = 0
        self._fallback_steps = 0
        self._unsafe_model_targets = 0
        self._action_counts = {i: 0 for i in range(5)}

    def _to_cell(self, scene: LabScene, p: np.ndarray):
        x_min, _, y_min, _ = scene.bounds_xy
        res = float(scene.grid_resolution_m)
        return int((float(p[0]) - x_min) / res), int((float(p[1]) - y_min) / res)

    def _cell_center(self, scene: LabScene, c):
        x_min, _, y_min, _ = scene.bounds_xy
        res = float(scene.grid_resolution_m)
        return np.array([x_min + (c[0] + 0.5) * res, y_min + (c[1] + 0.5) * res], dtype=float)

    def _grid_shape(self, scene: LabScene) -> tuple[int, int]:
        x_min, x_max, y_min, y_max = scene.bounds_xy
        res = float(scene.grid_resolution_m)
        return max(1, int(np.ceil((x_max - x_min) / res))), max(1, int(np.ceil((y_max - y_min) / res)))

    def _blocked_cells(self, world: GridWorld2D, metrics=None) -> set[tuple[int, int]]:
        if metrics is not None and hasattr(metrics, "blocked_cells"):
            return set(metrics.blocked_cells)
        return set(world.obstacle_cells())

    def _visited_cells(self, metrics=None) -> set[tuple[int, int]]:
        if metrics is not None and hasattr(metrics, "visited_cells"):
            return set(metrics.visited_cells)
        return set(self._visited)

    def _is_safe_cell(self, c: tuple[int, int], *, scene: LabScene, world: GridWorld2D, metrics=None) -> bool:
        nx, ny = self._grid_shape(scene)
        if c[0] < 0 or c[1] < 0 or c[0] >= nx or c[1] >= ny:
            return False
        return c not in self._blocked_cells(world, metrics)

    def _nearest_unvisited_targets(
        self,
        *,
        robot_xy: Dict[str, np.ndarray],
        scene: LabScene,
        world: GridWorld2D,
        metrics,
    ) -> Dict[str, np.ndarray]:
        x_min, x_max, y_min, y_max = scene.bounds_xy
        res = float(scene.grid_resolution_m)
        nx = max(1, int(np.ceil((x_max - x_min) / res)))
        ny = max(1, int(np.ceil((y_max - y_min) / res)))
        blocked = set(metrics.blocked_cells) if metrics is not None and hasattr(metrics, "blocked_cells") else set(world.obstacle_cells())
        visited = set(metrics.visited_cells) if metrics is not None and hasattr(metrics, "visited_cells") else set(self._visited)
        unvisited = [(cx, cy) for cx in range(nx) for cy in range(ny) if (cx, cy) not in blocked and (cx, cy) not in visited]
        reserved: Set[Tuple[int, int]] = set()

        out: Dict[str, np.ndarray] = {}
        for name, p in sorted(robot_xy.items()):
            if not unvisited:
                out[name] = p.copy()
                continue
            c0 = self._to_cell(scene, p)
            avail = [c for c in unvisited if c not in reserved]
            if not avail:
                avail = unvisited
            tgt = min(avail, key=lambda c: abs(c[0] - c0[0]) + abs(c[1] - c0[1]))
            reserved.add(tgt)
            out[name] = self._cell_center(scene, tgt)
        return out

    def _model_targets(
        self,
        robot_xy: Dict[str, np.ndarray],
        *,
        scene: LabScene,
        world: GridWorld2D,
        metrics=None,
    ) -> tuple[Dict[str, np.ndarray], bool]:
        out = {}
        w = int(self.window)
        r = w // 2
        blocked = self._blocked_cells(world, metrics)
        visited = self._visited_cells(metrics)
        unsafe = False
        nx, ny = self._grid_shape(scene)
        for name, p in robot_xy.items():
            c0 = self._to_cell(scene, p)
            self._visited.add(c0)
            patch = np.zeros((1, self._model.net[0].in_channels, w, w), dtype=np.float32)
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    cx, cy = c0[0] + dx, c0[1] + dy
                    if cx < 0 or cy < 0 or cx >= nx or cy >= ny:
                        continue
                    px, py = dy + r, dx + r
                    cell = (cx, cy)
                    # Channel 0 is always visited for backwards compatibility.
                    if cell in visited or cell in self._visited:
                        patch[0, 0, px, py] = 1.0
                    if patch.shape[1] >= 2 and cell in blocked:
                        patch[0, 1, px, py] = 1.0
                    if patch.shape[1] >= 3 and cell not in blocked and cell not in visited:
                        patch[0, 2, px, py] = 1.0
                    if patch.shape[1] >= 4 and dx == 0 and dy == 0:
                        patch[0, 3, px, py] = 1.0
            with self._torch.no_grad():
                logits = self._model(self._torch.from_numpy(patch))
                a = int(self._torch.argmax(logits, dim=1).item())
            self._action_counts[a] = self._action_counts.get(a, 0) + 1
            if a == 0:
                dc = (0, 1)
            elif a == 1:
                dc = (0, -1)
            elif a == 2:
                dc = (1, 0)
            elif a == 3:
                dc = (-1, 0)
            else:
                dc = (0, 0)
            tgt_cell = (c0[0] + dc[0], c0[1] + dc[1])
            if not self._is_safe_cell(tgt_cell, scene=scene, world=world, metrics=metrics):
                unsafe = True
                self._unsafe_model_targets += 1
            out[name] = self._cell_center(scene, tgt_cell)
        return out, unsafe

    def choose_targets(self, robot_xy: Dict[str, np.ndarray], *, scene: LabScene, world: GridWorld2D, metrics=None, state=None):
        self._total_decisions += 1
        cov = float(metrics.coverage_percent()) if metrics is not None and hasattr(metrics, "coverage_percent") else 0.0
        if cov <= self._last_cov + 1e-6:
            self._stagnation += 1
        else:
            self._stagnation = 0
        self._last_cov = cov

        if self.mode == "guarded" and metrics is not None and (
            cov < float(scene.target_coverage) or self._stagnation >= int(self.stagnation_steps)
        ):
            self._fallback_steps += 1
            return self._guardrail.choose_targets(robot_xy, scene=scene, world=world, metrics=metrics)

        self._model_steps += 1
        out, unsafe = self._model_targets(robot_xy, scene=scene, world=world, metrics=metrics)
        if self.mode == "soft_guarded" and (unsafe or self._stagnation >= int(self.stagnation_steps)):
            self._fallback_steps += 1
            return self._guardrail.choose_targets(robot_xy, scene=scene, world=world, metrics=metrics)
        return out

    def diagnostics(self) -> dict[str, float | int | str | dict[int, int]]:
        total = max(int(self._total_decisions), 1)
        model = int(self._model_steps)
        fallback = int(self._fallback_steps)
        return {
            "ml_mode": self.mode,
            "ml_input_mode": self._input_mode,
            "ml_model_variant": self._model_variant,
            "ml_total_decisions": int(self._total_decisions),
            "ml_model_steps": model,
            "ml_fallback_steps": fallback,
            "ml_model_step_rate": float(model / total),
            "ml_fallback_rate": float(fallback / total),
            "ml_unsafe_model_targets": int(self._unsafe_model_targets),
            "ml_action_counts": {str(k): int(v) for k, v in sorted(self._action_counts.items())},
        }

