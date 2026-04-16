from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np


def save_plots(result: Dict[str, Any], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    coverage = result.get("coverage_history") or []
    distance = result.get("distance_history") or []
    if coverage:
        fig = plt.figure(figsize=(8, 4))
        plt.plot(coverage, label="coverage")
        plt.xlabel("Step")
        plt.ylabel("Coverage")
        plt.title(f"Coverage vs Time ({result.get('algorithm')})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        fig.tight_layout()
        fig.savefig(str(output_prefix) + "_coverage_vs_time.png", dpi=140)
        plt.close(fig)

    if coverage and distance:
        fig = plt.figure(figsize=(8, 4))
        plt.plot(distance, coverage, label="distance-coverage")
        plt.xlabel("Distance travelled (m)")
        plt.ylabel("Coverage")
        plt.title(f"Distance vs Coverage ({result.get('algorithm')})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        fig.tight_layout()
        fig.savefig(str(output_prefix) + "_distance_vs_coverage.png", dpi=140)
        plt.close(fig)

    _save_coverage_2d(result, output_prefix)


def _save_coverage_2d(result: Dict[str, Any], output_prefix: Path) -> None:
    bounds = result.get("bounds_xy")
    grid_res = result.get("grid_resolution_m")
    visited_cells = result.get("visited_cells", [])
    robot_paths = result.get("robot_paths", {}) or {}
    obstacles = result.get("obstacles", []) or []
    if not bounds or not grid_res:
        return

    x_min, x_max, y_min, y_max = [float(v) for v in bounds]
    grid_res = float(grid_res)
    nx = max(1, int(np.ceil((x_max - x_min) / grid_res)))
    ny = max(1, int(np.ceil((y_max - y_min) / grid_res)))
    cov_grid = np.zeros((ny, nx), dtype=float)

    for cell in visited_cells:
        cx, cy = int(cell[0]), int(cell[1])
        if 0 <= cx < nx and 0 <= cy < ny:
            cov_grid[cy, cx] = 1.0

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(
        cov_grid,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap="viridis",
        alpha=0.75,
        interpolation="nearest",
    )
    for name, pts in robot_paths.items():
        if not pts:
            continue
        arr = np.array(pts, dtype=float)
        plt.plot(arr[:, 0], arr[:, 1], linewidth=1.8, label=name)
        plt.scatter(arr[0, 0], arr[0, 1], s=28, marker="o")
        plt.scatter(arr[-1, 0], arr[-1, 1], s=34, marker="x")

    if obstacles:
        # Support both legacy disks and new typed obstacles
        for obs in obstacles:
            t = str(obs.get("type", "disk"))
            if t == "rect" and all(k in obs for k in ("x", "y", "w", "h")):
                x = float(obs["x"])
                y = float(obs["y"])
                w = float(obs["w"])
                h = float(obs["h"])
                plt.gca().add_patch(
                    plt.Rectangle((x - w * 0.5, y - h * 0.5), w, h, facecolor="#4a4e69", alpha=0.18, edgecolor="#4a4e69", lw=1.0)
                )
            elif all(k in obs for k in ("x", "y", "r")):
                x = float(obs["x"])
                y = float(obs["y"])
                r = float(obs["r"])
                plt.gca().add_patch(plt.Circle((x, y), r, color="#4a4e69", alpha=0.25, lw=0.8))

    blocked = int(result.get("blocked_moves", 0))
    plt.title(f"2D Coverage Map ({result.get('algorithm')}), blocked={blocked}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True, alpha=0.2)
    if robot_paths:
        plt.legend()
    fig.tight_layout()
    fig.savefig(str(output_prefix) + "_coverage_2d.png", dpi=160)
    plt.close(fig)

