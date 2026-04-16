from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter


def _bounds(result: Dict[str, Any]) -> Tuple[float, float, float, float]:
    b = result.get("bounds_xy")
    if not b or len(b) != 4:
        raise ValueError("В JSON нет bounds_xy=[x_min,x_max,y_min,y_max]")
    x_min, x_max, y_min, y_max = [float(v) for v in b]
    return x_min, x_max, y_min, y_max


def _frames_count(result: Dict[str, Any]) -> int:
    n = 0
    for pts in (result.get("robot_paths") or {}).values():
        if pts:
            n = max(n, len(pts))
    for pts in (result.get("pedestrian_paths") or {}).values():
        if pts:
            n = max(n, len(pts))
    return max(n, 1)


def _coverage_at(result: Dict[str, Any], fi: int) -> float | None:
    hist = result.get("coverage_history") or []
    if not hist:
        return None
    return float(hist[min(fi, len(hist) - 1)])


def render_gif(
    result: Dict[str, Any],
    out_gif: Path,
    *,
    fps: int = 18,
    stride: int = 2,
    trail: int = 60,
    show_dynamic_coverage: bool = True,
) -> None:
    x_min, x_max, y_min, y_max = _bounds(result)
    algo = str(result.get("algorithm", "?"))

    robot_paths = result.get("robot_paths") or {}
    ped_paths = result.get("pedestrian_paths") or {}
    obstacles = result.get("obstacles") or []
    ped_enabled = bool(result.get("pedestrians_enabled", False))
    min_ped_clear = result.get("min_pedestrian_clearance_m", None)

    frames = _frames_count(result)
    idxs = list(range(0, frames, max(int(stride), 1)))
    if idxs[-1] != frames - 1:
        idxs.append(frames - 1)

    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.18)

    # Background: dynamic coverage mask (only visited cells are visible).
    grid_res = result.get("grid_resolution_m", None)
    cov_grid: np.ndarray | None = None
    cov_img = None
    res = None
    nx = ny = None
    if grid_res and show_dynamic_coverage:
        try:
            res = float(grid_res)
            nx = max(1, int(np.ceil((x_max - x_min) / res)))
            ny = max(1, int(np.ceil((y_max - y_min) / res)))
            cov_grid = np.zeros((ny, nx), dtype=float)
            cov_cmap = plt.get_cmap("viridis").copy()
            cov_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
            cov_mask = np.ma.masked_where(cov_grid <= 0.0, cov_grid)
            cov_img = ax.imshow(
                cov_mask,
                origin="lower",
                extent=[x_min, x_max, y_min, y_max],
                cmap=cov_cmap,
                alpha=0.55,
                interpolation="nearest",
                zorder=0,
            )
        except Exception:
            cov_grid = None
            cov_img = None

    def _to_cell(p: np.ndarray) -> tuple[int, int] | None:
        if cov_grid is None or res is None or nx is None or ny is None:
            return None
        cx = int((float(p[0]) - x_min) / res)
        cy = int((float(p[1]) - y_min) / res)
        if 0 <= cx < nx and 0 <= cy < ny:
            return (cx, cy)
        return None

    # obstacles (legacy disks + new typed rectangles)
    for o in obstacles:
        t = str(o.get("type", "disk"))
        if t == "rect" and all(k in o for k in ("x", "y", "w", "h")):
            try:
                ox, oy, w, h = float(o["x"]), float(o["y"]), float(o["w"]), float(o["h"])
            except Exception:
                continue
            ax.add_patch(
                plt.Rectangle(
                    (ox - w * 0.5, oy - h * 0.5),
                    w,
                    h,
                    facecolor="#4a4e69",
                    alpha=0.22,
                    edgecolor="#4a4e69",
                    lw=1.0,
                )
            )
        else:
            try:
                ox, oy, r = float(o["x"]), float(o["y"]), float(o["r"])
            except Exception:
                continue
            ax.add_patch(plt.Circle((ox, oy), r, color="#4a4e69", alpha=0.30, lw=0.8))

    robot_names = sorted(robot_paths.keys())
    colors = ["#c8553d", "#3d5a80", "#81b29a", "#f2cc8f", "#9b6b9e"]
    r_scats, r_lines = [], []
    for i, _ in enumerate(robot_names):
        col = colors[i % len(colors)]
        line, = ax.plot([], [], color=col, lw=2.0, alpha=0.55)
        scat = ax.scatter([], [], s=55, c=col, marker="o", edgecolor="black", linewidth=0.3, zorder=5)
        r_lines.append(line)
        r_scats.append(scat)

    ped_names = sorted(ped_paths.keys())
    ped_colors = ["#e5989b", "#b5838d", "#ffc6ff", "#9b6b9e"]
    p_scats, p_lines = [], []
    for i, _ in enumerate(ped_names):
        col = ped_colors[i % len(ped_colors)]
        line, = ax.plot([], [], color=col, lw=1.4, alpha=0.35)
        scat = ax.scatter([], [], s=38, c=col, marker="s", edgecolor="black", linewidth=0.25, zorder=6)
        p_lines.append(line)
        p_scats.append(scat)

    title = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.75, edgecolor="#ddd"),
    )

    def _set_line(line, pts: np.ndarray) -> None:
        line.set_data(pts[:, 0], pts[:, 1])

    def _set_scat(scat, p: np.ndarray) -> None:
        scat.set_offsets(np.array([[p[0], p[1]]], dtype=float))

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=int(fps))
    with writer.saving(fig, str(out_gif), dpi=120):
        # incremental update for coverage grid
        last_j = -1
        visited_set: set[tuple[int, int]] = set()
        for fi in idxs:
            if cov_grid is not None and cov_img is not None:
                # compute max index available for this frame across robots
                jmax = -1
                for pts in robot_paths.values():
                    if pts:
                        jmax = max(jmax, min(fi, len(pts) - 1))
                if jmax > last_j:
                    for jj in range(last_j + 1, jmax + 1):
                        for pts in robot_paths.values():
                            if not pts or jj >= len(pts):
                                continue
                            p = np.asarray(pts[jj], dtype=float)
                            c = _to_cell(p)
                            if c is not None and c not in visited_set:
                                visited_set.add(c)
                                cov_grid[c[1], c[0]] = 1.0
                    last_j = jmax
                    cov_mask = np.ma.masked_where(cov_grid <= 0.0, cov_grid)
                    cov_img.set_data(cov_mask)

            for ri, name in enumerate(robot_names):
                pts = robot_paths.get(name) or []
                if not pts:
                    continue
                arr = np.asarray(pts, dtype=float)
                j = min(fi, len(arr) - 1)
                a0 = max(0, j - int(trail))
                _set_line(r_lines[ri], arr[a0 : j + 1])
                _set_scat(r_scats[ri], arr[j])

            for pi, name in enumerate(ped_names):
                pts = ped_paths.get(name) or []
                if not pts:
                    continue
                arr = np.asarray(pts, dtype=float)
                j = min(fi, len(arr) - 1)
                a0 = max(0, j - int(trail))
                _set_line(p_lines[pi], arr[a0 : j + 1])
                _set_scat(p_scats[pi], arr[j])

            cov = _coverage_at(result, fi)
            cov_txt = "—" if cov is None else f"{cov * 100:.2f}%"
            final_cov = result.get("coverage_percent", None)
            final_txt = "—" if final_cov is None else f"{float(final_cov) * 100:.2f}%"
            ped_txt = "on" if ped_enabled else "off"
            mpc_txt = "—" if min_ped_clear is None else f"{float(min_ped_clear):.2f} m"
            title.set_text(
                f"{algo}\nframe {fi}/{frames - 1} | coverage {cov_txt} | final {final_txt}\n"
                f"pedestrians {ped_txt} | min_clear {mpc_txt}"
            )

            fig.canvas.draw()
            writer.grab_frame()

    plt.close(fig)

