"""
Генерация медиа-артефактов из JSON результата прогона:
- PNG: графики метрик + 2D-карта покрытия (использует coverage_sim.visualization.coverage_map.save_plots)
- GIF: анимация движения роботов/пешеходов + obstacles + текущий coverage

Пример:
  python scripts/render_run_media.py results/batch_benchmark/frontier_seed0.json
  python scripts/render_run_media.py results/batch_dynamic_quick/random_walk_seed0.json --fps 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

ROOT = Path(__file__).resolve().parent.parent
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bounds(result: Dict[str, Any]) -> Tuple[float, float, float, float]:
    b = result.get("bounds_xy")
    if not b or len(b) != 4:
        raise ValueError("В JSON нет bounds_xy=[x_min,x_max,y_min,y_max]")
    x_min, x_max, y_min, y_max = [float(v) for v in b]
    return x_min, x_max, y_min, y_max


def _get_frames_count(result: Dict[str, Any]) -> int:
    paths = result.get("robot_paths") or {}
    n = 0
    for pts in paths.values():
        if pts:
            n = max(n, len(pts))
    peds = result.get("pedestrian_paths") or {}
    for pts in peds.values():
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
) -> None:
    x_min, x_max, y_min, y_max = _bounds(result)
    algo = str(result.get("algorithm", "?"))

    robot_paths = result.get("robot_paths") or {}
    ped_paths = result.get("pedestrian_paths") or {}
    obstacles = result.get("obstacles") or []
    ped_enabled = bool(result.get("pedestrians_enabled", False))
    min_ped_clear = result.get("min_pedestrian_clearance_m", None)

    frames = _get_frames_count(result)
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

    # Obstacles as circles.
    for o in obstacles:
        try:
            ox, oy, r = float(o["x"]), float(o["y"]), float(o["r"])
        except Exception:
            continue
        circ = plt.Circle((ox, oy), r, color="#4a4e69", alpha=0.35, lw=0.8)
        ax.add_patch(circ)

    # Artists per robot
    robot_names = sorted(robot_paths.keys())
    colors = ["#c8553d", "#3d5a80", "#81b29a", "#f2cc8f", "#9b6b9e"]
    r_scats = []
    r_lines = []
    for i, name in enumerate(robot_names):
        col = colors[i % len(colors)]
        line, = ax.plot([], [], color=col, lw=2.0, alpha=0.55)
        scat = ax.scatter([], [], s=55, c=col, marker="o", edgecolor="black", linewidth=0.3, zorder=5)
        r_lines.append(line)
        r_scats.append(scat)

    # Pedestrians
    ped_names = sorted(ped_paths.keys())
    p_scats = []
    p_lines = []
    ped_colors = ["#e5989b", "#b5838d", "#ffc6ff", "#9b6b9e"]
    for i, name in enumerate(ped_names):
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

    writer = PillowWriter(fps=int(fps))
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(out_gif), dpi=120):
        for fi in idxs:
            # robots
            for ri, name in enumerate(robot_names):
                pts = robot_paths.get(name) or []
                if not pts:
                    continue
                arr = np.asarray(pts, dtype=float)
                j = min(fi, len(arr) - 1)
                a0 = max(0, j - int(trail))
                _set_line(r_lines[ri], arr[a0 : j + 1])
                _set_scat(r_scats[ri], arr[j])

            # pedestrians
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
            ped_txt = "on" if ped_enabled else "off"
            mpc_txt = "—" if min_ped_clear is None else f"{float(min_ped_clear):.2f} m"
            title.set_text(
                f"{algo}\n"
                f"frame {fi}/{frames - 1} | coverage {cov_txt}\n"
                f"pedestrians {ped_txt} | min_clear {mpc_txt}"
            )

            fig.canvas.draw()
            writer.grab_frame()

    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("json_path", type=Path, help="Путь к results/.../*.json")
    p.add_argument("--out-dir", type=Path, default=None, help="Куда писать артефакты (по умолчанию рядом с JSON)")
    p.add_argument("--fps", type=int, default=18)
    p.add_argument("--stride", type=int, default=2, help="Шаг кадров (2 = каждый 2-й)")
    p.add_argument("--trail", type=int, default=60, help="Длина хвоста траектории в кадрах")
    args = p.parse_args()

    json_path = args.json_path
    if not json_path.is_absolute():
        json_path = (ROOT / json_path).resolve()
    result = _load_json(json_path)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = json_path.parent
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # PNG plots (reuse existing helper)
    from coverage_sim.visualization.coverage_map import save_plots

    stem = json_path.stem
    save_plots(result, out_dir / stem)

    # GIF
    out_gif = out_dir / f"{stem}.gif"
    render_gif(result, out_gif, fps=args.fps, stride=args.stride, trail=args.trail)

    print(f"OK: {out_gif}")


if __name__ == "__main__":
    main()

