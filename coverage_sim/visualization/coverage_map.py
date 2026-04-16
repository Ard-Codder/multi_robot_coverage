from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def save_plots(result: dict, output_prefix: Path) -> None:
    coverage = result.get("coverage_history", [])
    distance = result.get("distance_history", [])
    if not coverage:
        return

    fig1 = plt.figure(figsize=(8, 4))
    plt.plot(coverage, label="coverage")
    plt.xlabel("Step")
    plt.ylabel("Coverage")
    plt.title(f"Coverage vs Time ({result.get('algorithm')})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig1.tight_layout()
    fig1.savefig(str(output_prefix) + "_coverage_vs_time.png", dpi=120)
    plt.close(fig1)

    if distance:
        fig2 = plt.figure(figsize=(8, 4))
        plt.plot(distance, coverage, label="distance-coverage")
        plt.xlabel("Distance travelled (m)")
        plt.ylabel("Coverage")
        plt.title(f"Distance vs Coverage ({result.get('algorithm')})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        fig2.tight_layout()
        fig2.savefig(str(output_prefix) + "_distance_vs_coverage.png", dpi=120)
        plt.close(fig2)

    _save_coverage_2d(result, output_prefix)


def _save_coverage_2d(result: dict, output_prefix: Path) -> None:
    bounds = result.get("bounds_xy")
    grid_res = result.get("grid_resolution_m")
    visited_cells = result.get("visited_cells", [])
    robot_paths = result.get("robot_paths", {})
    obstacles = result.get("obstacles", [])
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
        plt.scatter(arr[0, 0], arr[0, 1], s=30, marker="o")
        plt.scatter(arr[-1, 0], arr[-1, 1], s=35, marker="x")

    # Overlay occupancy obstacle cells (any scene object except floor).
    if obstacles:
        ox = [float(obs["x"]) for obs in obstacles]
        oy = [float(obs["y"]) for obs in obstacles]
        plt.scatter(ox, oy, s=8, c="red", alpha=0.22, marker="s", label="obstacles")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    blocked_moves = int(result.get("blocked_moves", 0))
    plt.title(f"2D Coverage Map + Paths + Obstacles ({result.get('algorithm')}), blocked={blocked_moves}")
    plt.grid(True, alpha=0.2)
    if robot_paths:
        plt.legend()
    fig.tight_layout()
    fig.savefig(str(output_prefix) + "_coverage_2d.png", dpi=140)
    plt.close(fig)

