from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Sample:
    obs: np.ndarray  # (H,W) visited mask
    action: int  # 0..4


def _discretize_move(p0: np.ndarray, p1: np.ndarray) -> int:
    d = p1 - p0
    if float(np.linalg.norm(d)) < 1e-6:
        return 4
    if abs(float(d[0])) > abs(float(d[1])):
        return 2 if float(d[0]) > 0 else 3
    return 0 if float(d[1]) > 0 else 1


def build_dataset_from_runs(
    json_paths: List[Path],
    *,
    window: int = 9,
    input_mode: str = "legacy",
) -> Tuple[np.ndarray, np.ndarray]:
    """Supervised dataset from teacher trajectories.

    legacy: one channel with incremental visited cells.
    rich: visited, blocked, free-unvisited, robot-center channels.
    """
    X: List[np.ndarray] = []
    y: List[int] = []
    rich = input_mode == "rich"
    in_ch = 4 if rich else 1
    for jp in json_paths:
        data = json.loads(jp.read_text(encoding="utf-8"))
        bounds = data["bounds_xy"]
        x_min, x_max, y_min, y_max = [float(v) for v in bounds]
        res = float(data["grid_resolution_m"])
        nx = max(1, int(np.ceil((x_max - x_min) / res)))
        ny = max(1, int(np.ceil((y_max - y_min) / res)))
        blocked_cells = set(tuple(c) for c in data.get("obstacle_cells", []))
        robot_paths: Dict[str, List[List[float]]] = data.get("robot_paths", {})
        arrays = {
            name: np.asarray(pts, dtype=float)
            for name, pts in robot_paths.items()
            if pts and len(pts) >= 2
        }
        if not arrays:
            continue
        max_len = max(len(arr) for arr in arrays.values())
        visited_so_far: set[tuple[int, int]] = set()
        for i in range(max_len - 1):
            for arr in arrays.values():
                if i < len(arr):
                    c = (int((arr[i][0] - x_min) / res), int((arr[i][1] - y_min) / res))
                    visited_so_far.add(c)
            for pts in arrays.values():
                if i >= len(pts) - 1:
                    continue
                p0 = pts[i]
                p1 = pts[i + 1]
                cx = int((p0[0] - x_min) / res)
                cy = int((p0[1] - y_min) / res)
                w = int(window)
                r = w // 2
                obs = np.zeros((in_ch, w, w), dtype=np.float32)
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        cell = (cx + dx, cy + dy)
                        if cell[0] < 0 or cell[1] < 0 or cell[0] >= nx or cell[1] >= ny:
                            continue
                        px, py = dy + r, dx + r
                        if cell in visited_so_far:
                            obs[0, px, py] = 1.0
                        if rich:
                            if cell in blocked_cells:
                                obs[1, px, py] = 1.0
                            if cell not in blocked_cells and cell not in visited_so_far:
                                obs[2, px, py] = 1.0
                            if dx == 0 and dy == 0:
                                obs[3, px, py] = 1.0
                X.append(obs)
                y.append(_discretize_move(p0, p1))
    if not X:
        return np.zeros((0, in_ch, window, window), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(X, axis=0), np.asarray(y, dtype=np.int64)

