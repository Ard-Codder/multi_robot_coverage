"""
Replay a CoverageLab result JSON in NVIDIA Isaac Sim 5.x (kinematic visualization).

Must be launched with Isaac Sim's Python, e.g.:
  C:\\isaacsim\\python.bat experiments\\isaac5_replay_coverage_lab.py --json results\\lab\\...\\run.json
  or pass the JSON as the first positional argument (same as run_isaac_replay_large_complex.bat).

Coordinate convention: lab (x, y) maps to world X, Y; Z is up (ground at Z=0).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coverage_lab.isaac_export.replay_data import (
    LabReplayData,
    assert_paths_align_for_replay,
    build_lab_replay_data,
    iter_replay_frame_indices,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Isaac Sim 5 replay of CoverageLab JSON trajectories.")
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Path to coverage_lab result JSON (robot_paths, pedestrian_paths, obstacles).",
    )
    p.add_argument(
        "--geometry-yaml",
        type=Path,
        default=None,
        help="Optional scene YAML; if set, static obstacles/bounds come from YAML (paths still from JSON).",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "Use every N-th sample along paths (preview speed). "
            "Default: 1 for interactive replay, 3 with --gif-out (matches 2D batch GIF cadence)."
        ),
    )
    p.add_argument("--max-steps", type=int, default=None, help="Max path index (truncate replay).")
    p.add_argument("--headless", action="store_true", help="Run without a window.")
    p.add_argument(
        "--updates-per-frame",
        type=int,
        default=1,
        help="How many SimulationApp.update() calls per trajectory step (render smoothness).",
    )
    p.add_argument(
        "--experience",
        type=Path,
        default=None,
        help="Optional path to a .kit experience file (default: ISAAC_PATH/apps/isaacsim.exp.base.python.kit).",
    )
    p.add_argument(
        "--compat-renderer",
        action="store_true",
        help=(
            "Windows: force D3D12 (--/app/vulkan=false) to avoid Vulkan+RTX crashes (RTX 50 / Isaac 5.0). "
            "Same as COVERAGE_LAB_ISAAC_COMPAT=1. Does NOT set compatibilityMode (that often breaks RTX Hydra = black viewport)."
        ),
    )
    p.add_argument(
        "--compat-aggressive",
        action="store_true",
        help=(
            "Also set omni.kit.renderer.core compatibilityMode (only if you still crash without it; "
            "may log 'HydraEngine rtx failed' and black viewport — prefer without this flag)."
        ),
    )
    p.add_argument(
        "--replay-speed",
        type=float,
        default=0.0,
        help=">0: pause between keyframes ≈ (stride*dt_sec)/speed so motion is visible (1 ≈ lab realtime). 0 = as fast as possible.",
    )
    p.add_argument(
        "--robot-size",
        type=float,
        default=None,
        help="Robot cube edge length in meters (omit for auto scale from map bounds).",
    )
    p.add_argument(
        "--ped-size",
        type=float,
        default=None,
        help="Pedestrian cube edge length in meters (omit for auto scale from map bounds).",
    )
    p.add_argument(
        "--warmup-frames",
        type=int,
        default=48,
        help="SimulationApp.update() ticks after app start before creating USD (avoids empty stage / wrong sync). Use 0 to skip.",
    )
    p.add_argument(
        "--gif-out",
        type=Path,
        default=None,
        help="Записать вид активного viewport в GIF (как «лабораторная» гифка, но из Isaac). Несовместимо с --headless. Нужен Pillow.",
    )
    p.add_argument("--gif-fps", type=int, default=16, help="Скорость GIF при --gif-out (по умолчанию 16, как 2D batch GIF).")
    p.add_argument(
        "--gif-substeps",
        type=int,
        default=1,
        help="При --gif-out: сколько промежуточных кадров между двумя индексами траектории (линейная интерполяция xy). 1 = как 2D GIF, без доп. кадров.",
    )
    p.add_argument(
        "--gif-post-updates",
        type=int,
        default=10,
        help="Доп. вызовов update() перед каждым кадром захвата (RTX/гидра).",
    )
    p.add_argument(
        "--camera",
        choices=("angled", "topdown", "framed"),
        default="topdown",
        help="topdown — вид сверху (по умолчанию, вся арена в кадре); angled — диагональ; framed — авто-кадр по сцене.",
    )
    p.add_argument(
        "--gif-max-width",
        type=int,
        default=960,
        help="Макс. ширина кадра GIF (0 = без ресайза). Уменьшает размер файла.",
    )
    p.add_argument(
        "--auto-close",
        action="store_true",
        help="После воспроизведения сразу выйти (не ждать закрытия окна). Удобно с --gif-out / автоматизация.",
    )
    p.add_argument(
        "--visual-style",
        choices=("realistic", "lab"),
        default="realistic",
        help=(
            "realistic: асфальт + сетка снега (следы роботов), трава на крышах прямоугольников, "
            "роверы/люди из USD (Nova Carter + metropolis test при наличии), иначе капсулы. "
            "lab — кубики, без снега/асфальта-сцены."
        ),
    )
    p.add_argument(
        "json_file",
        nargs="?",
        type=Path,
        default=None,
        metavar="RESULT.json",
        help="Same as --json; use as first argument when calling via .bat with %%*.",
    )
    ns = p.parse_args()
    json_path = ns.json if ns.json is not None else ns.json_file
    if json_path is None:
        p.error("Specify --json PATH or pass RESULT.json as the first argument.")
    setattr(ns, "json_path", json_path)
    return ns


def _wall_thickness_xy(bounds_xy: Tuple[float, float, float, float]) -> float:
    x_min, x_max, y_min, y_max = bounds_xy
    span_x = max(x_max - x_min, 1e-6)
    span_y = max(y_max - y_min, 1e-6)
    return float(max(0.12, min(1.5, 0.006 * min(span_x, span_y))))


def _outer_arena_xy(bounds_xy: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Внешний прямоугольник с учётом толщины бордов LabBounds (чтобы topdown кадрировал «только арену + стены»)."""
    x_min, x_max, y_min, y_max = bounds_xy
    t = _wall_thickness_xy(bounds_xy)
    return x_min - t, x_max + t, y_min - t, y_max + t


def _xy_on_path_lerp(path: Sequence[Sequence[float]], fi: float) -> Tuple[float, float]:
    """Линейная интерполяция (x,y) по индексу fi в float."""
    n = len(path)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return float(path[0][0]), float(path[0][1])
    fi = max(0.0, min(float(fi), float(n - 1) - 1e-6))
    i0 = int(math.floor(fi))
    i1 = min(i0 + 1, n - 1)
    t = fi - i0
    x0, y0 = float(path[i0][0]), float(path[i0][1])
    x1, y1 = float(path[i1][0]), float(path[i1][1])
    return (1.0 - t) * x0 + t * x1, (1.0 - t) * y0 + t * y1


def _heading_deg_on_path(path: Sequence[Sequence[float]], fi: float, window: float = 0.75) -> float:
    """Курс в градусах (Z-вращение) по локальной разнице точек на траектории."""
    n = len(path)
    if n < 2:
        return 0.0
    lo = max(0.0, fi - window)
    hi = min(float(n - 1), fi + window)
    xa, ya = _xy_on_path_lerp(path, lo)
    xb, yb = _xy_on_path_lerp(path, hi)
    return float(math.degrees(math.atan2(yb - ya, xb - xa)))


def _gif_sample_indices_float(indices: List[int], substeps: int) -> List[float]:
    """Плавный GIF: первый кадр на indices[0], затем по каждому отрезку indices[k]→indices[k+1] равномерно substeps шагов (включая конечную точку)."""
    if len(indices) == 0:
        return []
    if substeps <= 1:
        return [float(i) for i in indices]
    out: List[float] = [float(indices[0])]
    for k in range(len(indices) - 1):
        i0, i1 = int(indices[k]), int(indices[k + 1])
        for s in range(1, substeps + 1):
            out.append(float(i0 + (i1 - i0) * (s / float(substeps))))
    return out


def _set_orient_yaw(stage, orient_path: str, yaw_deg: float) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(orient_path)
    if not prim.IsValid():
        return
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, float(yaw_deg)))


def _try_hide_viewport_grid() -> None:
    """Сетка пола Kit (не USD) — выключить, если настройка есть."""
    try:
        import carb

        s = carb.settings.get_settings()
        for key in (
            "/persistent/exts/omni.kit.viewport.menubar.display/showGrid",
            "/exts/omni.kit.viewport.menubar.display/showGrid",
            "/persistent/app/viewport/showGrid",
        ):
            try:
                if hasattr(s, "set_bool"):
                    s.set_bool(key, False)
                else:
                    s.set(key, False)
            except Exception:
                pass
    except Exception:
        pass


def _set_translate(stage, prim_path: str, xyz: Sequence[float]) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    xf = UsdGeom.Xformable(prim)
    v = Gf.Vec3f(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    attr = prim.GetAttribute("xformOp:translate")
    if attr.IsValid():
        attr.Set(v)
        return
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(v)
            return
    xf.AddTranslateOp().Set(v)


def _rgb01(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    return rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0


# Fallback цвета для Visual* (до бинда материалов)
C_FLOOR = (165, 188, 155)
C_GRASS = (35, 120, 48)
C_TREE = (18, 78, 32)
C_ROBOT = (20, 95, 220)
C_HUMAN = (210, 150, 118)
C_BOUNDS = (160, 42, 42)
C_ASPHALT = (48, 50, 56)
C_SNOW = (235, 240, 248)
C_DIRT = (88, 72, 58)


class _SnowCoverState:
    """Сетка тонких «сугробов» над асфальтом: при проезде робота ячейки скрываются (след)."""

    __slots__ = ("x_min", "y_min", "cell_x", "cell_y", "nx", "ny", "paths", "_hidden", "stamp_radius_m")

    def __init__(
        self,
        x_min: float,
        y_min: float,
        cell_x: float,
        cell_y: float,
        nx: int,
        ny: int,
        paths: Tuple[str, ...],
        stamp_radius_m: float,
    ) -> None:
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.cell_x = float(cell_x)
        self.cell_y = float(cell_y)
        self.nx = int(nx)
        self.ny = int(ny)
        self.paths = paths
        self.stamp_radius_m = float(stamp_radius_m)
        self._hidden = [False] * len(paths)

    def stamp_clear(self, stage, x: float, y: float, radius: float) -> None:
        import math

        from pxr import UsdGeom

        if self.nx <= 0 or self.ny <= 0 or not self.paths:
            return
        cmin = max(min(self.cell_x, self.cell_y), 1e-6)
        dr = int(math.ceil(radius / cmin)) + 1
        ix0 = int((x - self.x_min) / max(self.cell_x, 1e-9))
        iy0 = int((y - self.y_min) / max(self.cell_y, 1e-9))
        for dix in range(-dr, dr + 1):
            for diy in range(-dr, dr + 1):
                ix = ix0 + dix
                iy = iy0 + diy
                if ix < 0 or iy < 0 or ix >= self.nx or iy >= self.ny:
                    continue
                cx = self.x_min + (ix + 0.5) * self.cell_x
                cy = self.y_min + (iy + 0.5) * self.cell_y
                if (cx - x) ** 2 + (cy - y) ** 2 > (radius + 0.32 * cmin) ** 2:
                    continue
                k = iy * self.nx + ix
                if k < 0 or k >= len(self.paths) or self._hidden[k]:
                    continue
                prim = stage.GetPrimAtPath(self.paths[k])
                if not prim.IsValid():
                    continue
                UsdGeom.Imageable(prim).MakeInvisible()
                self._hidden[k] = True


def _agent_sizes_from_bounds(bounds_xy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Кубики агентов: доля меньшей стороны арены, чтобы на больших картах не были точками."""
    x_min, x_max, y_min, y_max = bounds_xy
    span_x = max(x_max - x_min, 1e-6)
    span_y = max(y_max - y_min, 1e-6)
    s = min(span_x, span_y)
    robot = float(max(0.32, min(2.0, 0.024 * s)))
    ped = float(max(0.26, min(1.65, 0.019 * s)))
    return robot, ped


def _env_float_positive(name: str) -> Optional[float]:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        print(f"[isaac5_replay] WARN: {name}={raw!r} is not a number; ignoring.", flush=True)
        return None
    if value <= 0.0:
        print(f"[isaac5_replay] WARN: {name}={raw!r} must be > 0; ignoring.", flush=True)
        return None
    return value


def _resolve_snow_stamp_config(pack: LabReplayData, robot_size: float) -> Tuple[float, Optional[float], str]:
    """
    Resolve snow grid/stamp parameters with 2D parity.

    2D coverage updates one discrete visited cell per robot sample using
    ``grid_resolution_m``; when available we reuse it as snow cell target.
    """
    grid_res = None
    if pack.grid_resolution_m is not None and float(pack.grid_resolution_m) > 0.0:
        grid_res = float(pack.grid_resolution_m)
    fallback_cell = float(max(0.12, robot_size))
    env_cell = _env_float_positive("COVERAGE_LAB_SNOW_CELL_M")
    env_radius = _env_float_positive("COVERAGE_LAB_SNOW_STAMP_RADIUS_M")
    cell_m = float(env_cell if env_cell is not None else (grid_res if grid_res is not None else fallback_cell))
    if grid_res is not None:
        parity = f"grid_resolution_m={grid_res:.3f} -> 1 visited cell ~= {grid_res * grid_res:.3f}m2"
    else:
        parity = (
            f"grid_resolution_m missing -> fallback cell from robot_size={robot_size:.3f}m "
            f"(1 cell ~= {fallback_cell * fallback_cell:.3f}m2)"
        )
    if env_cell is not None and grid_res is not None:
        parity += f"; cell override COVERAGE_LAB_SNOW_CELL_M={env_cell:.3f}"
    if env_radius is not None:
        parity += f"; radius override COVERAGE_LAB_SNOW_STAMP_RADIUS_M={env_radius:.3f}"
    return cell_m, env_radius, parity


def _ensure_replay_materials(stage, mats_root: str = "/World/Looks/ReplayMats") -> Dict[str, str]:
    """UsdPreviewSurface с разными roughness/metallic/specular — газон, дерево, металл робота, кожа пешехода."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    looks = "/World/Looks"
    if not stage.GetPrimAtPath(looks).IsValid():
        UsdGeom.Scope.Define(stage, Sdf.Path(looks))
    if not stage.GetPrimAtPath(mats_root).IsValid():
        UsdGeom.Scope.Define(stage, Sdf.Path(mats_root))

    def one_mat(
        suffix: str,
        rgb: Tuple[int, int, int],
        *,
        roughness: float,
        metallic: float,
        specular_scale: float = 1.0,
        emissive_scale: float = 0.018,
    ) -> str:
        mat_path_str = f"{mats_root}/mat_{suffix}"
        mat_path = Sdf.Path(mat_path_str)
        if stage.GetPrimAtPath(mat_path).IsValid():
            return mat_path_str
        mat = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, mat_path.AppendPath("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        r, g, b = _rgb01(rgb)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(r, g, b))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
        sr = min(1.0, r * 1.15 * specular_scale)
        sg = min(1.0, g * 1.15 * specular_scale)
        sb = min(1.0, b * 1.15 * specular_scale)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(sr, sg, sb))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(r * emissive_scale, g * emissive_scale, b * emissive_scale)
        )
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat_path_str

    return {
        "floor": one_mat("floor", C_FLOOR, roughness=0.72, metallic=0.0, specular_scale=0.35),
        "asphalt": one_mat("asphalt", C_ASPHALT, roughness=0.94, metallic=0.02, specular_scale=0.28, emissive_scale=0.006),
        "snow": one_mat("snow", C_SNOW, roughness=0.78, metallic=0.0, specular_scale=0.55, emissive_scale=0.02),
        "dirt": one_mat("dirt", C_DIRT, roughness=0.88, metallic=0.0, specular_scale=0.35),
        "grass": one_mat("grass", C_GRASS, roughness=0.9, metallic=0.0, specular_scale=0.45),
        "tree": one_mat("tree", C_TREE, roughness=0.88, metallic=0.0, specular_scale=0.3),
        "robot": one_mat("robot", C_ROBOT, roughness=0.28, metallic=0.72, specular_scale=1.0, emissive_scale=0.01),
        "human": one_mat("human", C_HUMAN, roughness=0.52, metallic=0.04, specular_scale=0.55),
        "bounds": one_mat("bounds", C_BOUNDS, roughness=0.55, metallic=0.12, specular_scale=0.5),
    }


def _warmup_simulation_app(simulation_app, *, frames: int) -> None:
    """Прогон кадров после старта Kit: стартовый experience иногда подменяет stage асинхронно."""
    n = max(0, int(frames))
    if n == 0:
        return
    print(f"[isaac5_replay] warmup: {n} update() ticks before USD scene build", flush=True)
    for _ in range(n):
        simulation_app.update()


def _set_default_world_prim(stage) -> None:
    wp = stage.GetPrimAtPath("/World")
    if wp.IsValid():
        stage.SetDefaultPrim(wp)


def _print_world_stage_hint(stage) -> None:
    import os

    w = stage.GetPrimAtPath("/World")
    kids: List[str] = []
    if w.IsValid():
        for c in w.GetChildren():
            kids.append(c.GetName())
    pid = os.getpid()
    head = kids[:18]
    tail = " ..." if len(kids) > 18 else ""
    print(
        f"[isaac5_replay] pid={pid}  дети /World: {len(kids)}  [{', '.join(head)}{tail}]",
        flush=True,
    )
    if len(kids) < 3:
        print(
            "[isaac5_replay] Под /World слишком мало примов — либо Kit ещё не тот stage, "
            "либо открыто другое окно Isaac (не процесс этого python.bat). "
            "Разверните World в Outliner и сравните со списком.",
            flush=True,
        )


def _viewport_api_for_capture(viewport: Any) -> Any:
    """Объект с ``schedule_capture`` — настоящий ViewportAPI для Kit capture.

    ``get_active_viewport()`` уже возвращает ViewportAPI; у некоторых обёрток есть
    поле ``viewport_api``, указывающее на *другой* handle — тогда в файл уходит
    пустой/серый LdrColor (в окне одна сцена, в PNG — «точка»).
    """
    if viewport is None:
        raise ValueError("viewport is None")
    if callable(getattr(viewport, "schedule_capture", None)):
        return viewport
    inner = getattr(viewport, "viewport_api", None)
    if inner is not None and callable(getattr(inner, "schedule_capture", None)):
        return inner
    return viewport


def _normalized_render_product_path(vp_api: Any) -> Optional[str]:
    """Путь RenderProduct для ``capture_viewport_to_file(..., render_product_path=...)`` (см. omni.kit.widget.viewport tests)."""
    rp = getattr(vp_api, "render_product_path", None)
    if rp is None and callable(getattr(vp_api, "get_render_product_path", None)):
        try:
            rp = vp_api.get_render_product_path()
        except Exception:
            rp = None
    if not rp:
        return None
    s = str(rp).strip()
    if not s:
        return None
    if s[0] != "/":
        return f"/Render/RenderProduct_{s}"
    return s


def _png_looks_degenerate_flat(path: Path, *, std_thresh: float = 14.0) -> bool:
    """True если картинка почти однотонная (типичный сбой захвата RTX → серый + точка)."""
    try:
        from PIL import Image

        import numpy as np

        with Image.open(path) as im:
            arr = np.asarray(im.convert("L"), dtype=np.float32)
        if arr.size < 64:
            return True
        return float(arr.std()) < std_thresh
    except Exception:
        return True


def _write_png_from_viewport_capture_buffer(
    path: Path,
    buffer,
    buffer_size: int,
    width: int,
    height: int,
    byte_format: Any,
) -> None:
    """Сохранить кадр из GPU-буфера Hydra через ``omni.kit.renderer_capture.convert_raw_bytes_to_list``.

    В колбэке ``buffer`` часто приходит как **PyCapsule**, а не ``bytes`` — ``frombuffer`` падает.
    Запись через Pillow обходит Kit ``capture_next_frame_rp_resource`` → файл с пустым exe
    (``Can't find a file to execute``).
    """
    import numpy as np
    from PIL import Image

    w, h = int(width), int(height)
    bsz = int(buffer_size)

    import omni.kit.renderer_capture

    flat = omni.kit.renderer_capture.convert_raw_bytes_to_list(buffer, bsz, w, h, byte_format)
    arr = np.asarray(flat, dtype=np.uint8)
    npx = w * h
    if arr.size >= npx * 4:
        rgb = arr[: npx * 4].reshape((h, w, 4))[:, :, :3].copy()
    elif arr.size >= npx * 3:
        rgb = arr[: npx * 3].reshape((h, w, 3)).copy()
    else:
        raise ValueError(f"capture list size {arr.size} vs {w}x{h} fmt={byte_format!r}")
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG")


def _capture_viewport_png_sync(
    viewport: Any,
    path: Path,
    simulation_app,
    *,
    min_png_bytes: int = 8000,
    max_attempts: int = 6,
) -> None:
    """Сохранить один кадр viewport через буфер + Pillow.

    ``capture_viewport_to_file`` на части установок вызывает ``renderer_capture`` с
    записью на диск через внешний процесс с **пустым путём** → лог Kit
    ``Can't find a file to execute:`` и пустой/серый кадр. Буферный путь обходит это.

    Viewport — тот же, что при kick (см. ``_viewport_api_for_capture``).
    """
    from omni.kit.viewport.utility import capture_viewport_to_buffer, get_active_viewport, next_viewport_frame_async

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    def _resolve_viewport_api() -> Any:
        if viewport is not None:
            return _viewport_api_for_capture(viewport)
        vp_live = get_active_viewport()
        if vp_live is None:
            try:
                from omni.kit.viewport.utility import get_viewport_from_window_name

                vp_live = get_viewport_from_window_name("Viewport")
            except Exception:
                vp_live = None
        return _viewport_api_for_capture(vp_live) if vp_live is not None else None

    vp = _resolve_viewport_api()
    if vp is None:
        raise RuntimeError("[isaac5_replay] capture: нет viewport (ожидается окно после kick).")

    last_err: Optional[BaseException] = None
    for attempt in range(max_attempts):
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
        if attempt > 0:
            for _ in range(16 + attempt * 12):
                simulation_app.update()

        cap_err: List[Optional[BaseException]] = [None]

        def _on_buffer(buffer, buffer_size, width, height, byte_format) -> None:
            try:
                _write_png_from_viewport_capture_buffer(
                    path, buffer, buffer_size, width, height, byte_format
                )
            except Exception as e:
                cap_err[0] = e

        try:
            helper = capture_viewport_to_buffer(vp, _on_buffer, is_hdr=False)
        except Exception as e:
            last_err = e
            continue

        async def wait_capture() -> None:
            extra = 8 + attempt * 4
            try:
                await next_viewport_frame_async(vp, n_frames=extra)
            except TypeError:
                try:
                    await next_viewport_frame_async(vp)
                except TypeError:
                    pass
            cf = 64 + attempt * 16
            try:
                await helper.wait_for_result(completion_frames=cf)
            except TypeError:
                await helper.wait_for_result()

        try:
            simulation_app.run_coroutine(wait_capture(), run_until_complete=True)
        except Exception as e:
            last_err = e
            continue
        if cap_err[0] is not None:
            last_err = cap_err[0]
            continue
        if not path.is_file():
            last_err = FileNotFoundError(str(path))
            continue
        try:
            sz = path.stat().st_size
        except OSError:
            sz = 0
        flat = _png_looks_degenerate_flat(path)
        ok_size = sz >= min_png_bytes
        ok_content = ok_size and not flat
        if ok_content:
            return
        if attempt == max_attempts - 1:
            if not ok_size:
                print(
                    f"[isaac5_replay] WARN: PNG {sz} B < {min_png_bytes} B, last attempt kept.",
                    flush=True,
                )
            if flat:
                print(
                    "[isaac5_replay] WARN: flat frame (low L_std) — RTX/viewport; try --gif-post-updates 20, close extra kit.exe.",
                    flush=True,
                )
            return

    if last_err is not None:
        raise last_err
    raise TimeoutError("viewport capture: failed")


def _assemble_gif_from_png_folder(png_dir: Path, out_gif: Path, *, fps: int, max_width: int) -> None:
    try:
        from PIL import Image
    except ImportError as e:
        raise SystemExit("Для --gif-out нужен Pillow (pip install pillow).") from e

    paths = sorted(png_dir.glob("frame_*.png"))
    if not paths:
        raise SystemExit(f"[isaac5_replay] нет PNG в {png_dir}, GIF не собран")
    frames: List[Any] = []
    mw = int(max_width)
    for p in paths:
        with Image.open(p) as src:
            im = src.convert("RGB").copy()
        if mw > 0 and im.width > mw:
            nh = max(1, int(round(im.height * (mw / float(im.width)))))
            im = im.resize((mw, nh), Image.Resampling.LANCZOS).copy()
        frames.append(im)
    out_gif = out_gif.resolve()
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / max(1, int(fps)))
    save_kw: Dict[str, Any] = dict(
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    try:
        save_kw["disposal"] = 2
        frames[0].save(out_gif, **save_kw)
    except TypeError:
        save_kw.pop("disposal", None)
        frames[0].save(out_gif, **save_kw)
    for im in frames:
        im.close()
    print(f"[isaac5_replay] GIF saved: {out_gif} ({len(paths)} frames, {fps} fps)", flush=True)


def _frame_luma_std_mean_edge(rgb) -> Tuple[float, float, float]:
    """Метрики кадра: std/среднее по luminance, средняя величина градиента (контур сцены)."""
    import numpy as np

    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return 0.0, 0.0, 0.0
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    std = float(lum.std())
    mean = float(lum.mean())
    gx = np.abs(np.diff(lum, axis=1))
    gy = np.abs(np.diff(lum, axis=0))
    edge = float(0.5 * (gx.mean() + gy.mean()))
    return std, mean, edge


def _print_gif_pixel_quality_report(out_gif: Path, *, source_png_paths: Optional[List[Path]] = None) -> None:
    """Печать метрик по кадрам итогового GIF (и при желании сырых PNG) — не полагаться на размер файла."""
    try:
        from PIL import Image

        import numpy as np
    except ImportError:
        print("[isaac5_replay] pixel audit: skip (Pillow/numpy missing).", flush=True)
        return

    out_gif = out_gif.resolve()
    if not out_gif.is_file():
        print(f"[isaac5_replay] pixel audit: file not found {out_gif}", flush=True)
        return

    print("[isaac5_replay] --- GIF pixel audit (final GIF, per frame) ---", flush=True)
    stds: List[float] = []
    edges: List[float] = []
    with Image.open(out_gif) as g:
        n_frames = int(getattr(g, "n_frames", 1) or 1)
        for n in range(n_frames):
            g.seek(n)
            im = g.convert("RGB").copy()
            std, mean, edge = _frame_luma_std_mean_edge(im)
            stds.append(std)
            edges.append(edge)
            print(
                f"[isaac5_replay]   gif frame {n}: L_std={std:.2f} L_mean={mean:.1f} edge={edge:.3f} {im.width}x{im.height}",
                flush=True,
            )

    if source_png_paths:
        print("[isaac5_replay] --- raw PNG audit (first/last frame before GIF resize) ---", flush=True)
        for label, pth in (
            ("first", source_png_paths[0]),
            ("last", source_png_paths[-1]),
        ):
            if not pth.is_file():
                continue
            with Image.open(pth) as im:
                im = im.convert("RGB")
                std, mean, edge = _frame_luma_std_mean_edge(im)
                print(
                    f"[isaac5_replay]   png {label}: L_std={std:.2f} L_mean={mean:.1f} edge={edge:.3f} {im.width}x{im.height}",
                    flush=True,
                )

    if not stds:
        print("[isaac5_replay] pixel verdict: FAIL (0 frames in GIF).", flush=True)
        return

    med_std = float(np.median(np.array(stds, dtype=np.float64)))
    med_edge = float(np.median(np.array(edges, dtype=np.float64)))
    bad = sum(1 for s, e in zip(stds, edges) if s < 15.0 and e < 1.0)
    # Degenerate viewport: low luminance std and almost no gradient.
    if med_std < 16.0 and med_edge < 1.0:
        print(
            f"[isaac5_replay] pixel verdict: FAIL - empty/flat frames "
            f"(median L_std={med_std:.2f}, median edge={med_edge:.3f}; {bad}/{len(stds)} frames L_std<15 and edge<1). "
            "Run via Isaac python.bat with JSON args; PowerShell: .\\run_isaac_replay_large_complex.bat ...; "
            "close extra kit.exe; check log for HydraEngine rtx failed / GPU.",
            flush=True,
        )
    else:
        print(
            f"[isaac5_replay] pixel verdict: OK (median L_std={med_std:.2f}, median edge={med_edge:.3f}).",
            flush=True,
        )


def _bind_preview_material(stage, root_prim_path: str, mat_prim_path: str) -> None:
    from pxr import Usd, UsdGeom, UsdShade

    mat_prim = stage.GetPrimAtPath(mat_prim_path)
    root = stage.GetPrimAtPath(root_prim_path)
    if not mat_prim.IsValid() or not root.IsValid():
        return
    mat = UsdShade.Material(mat_prim)
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Cylinder) or prim.IsA(
            UsdGeom.Cone
        ) or prim.IsA(UsdGeom.Capsule):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def _isaac_install_root() -> Path:
    return Path((os.environ.get("ISAAC_PATH") or "").strip().strip('"'))


def _find_usd_filename(filename: str) -> Optional[str]:
    root = _isaac_install_root()
    if not root.is_dir():
        return None
    try:
        for p in root.rglob(filename):
            if p.is_file():
                return str(p.resolve())
    except OSError:
        return None
    return None


def _find_metropolis_test_character_usd() -> Optional[str]:
    root = _isaac_install_root()
    if not root.is_dir():
        return None
    try:
        for p in root.rglob("test.usd"):
            if not p.is_file():
                continue
            s = str(p).replace("\\", "/").lower()
            if "metropolis" in s and "characters" in s:
                return str(p.resolve())
    except OSError:
        return None
    return None


def _find_metropolis_mesh_character_usds(limit: int = 10) -> List[str]:
    """Поиск mesh-персонажей Metropolis под ISAAC_PATH (без skeleton-only/rig служебных USD)."""
    root = _isaac_install_root()
    if not root.is_dir() or limit <= 0:
        return []
    bad_tokens = (
        "skeleton",
        "skel",
        "rig",
        "retarget",
        "anim_graph",
        "animation",
        "locomotion",
        "proxy",
        "physics",
    )
    scored: List[Tuple[int, str]] = []
    try:
        for p in root.rglob("*.usd"):
            if not p.is_file():
                continue
            s = str(p).replace("\\", "/").lower()
            if "metropolis" not in s or "characters" not in s:
                continue
            if any(tok in s for tok in bad_tokens):
                continue
            score = 0
            if any(tok in s for tok in ("people", "pedestrian", "human", "adult", "male", "female")):
                score += 6
            if "mesh" in s or "geo" in s:
                score += 3
            if p.name.lower() == "test.usd":
                score -= 3
            scored.append((score, str(p.resolve())))
    except OSError:
        return []
    scored.sort(key=lambda x: (-x[0], x[1]))
    out: List[str] = []
    seen: set[str] = set()
    for _score, path in scored:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
        if len(out) >= limit:
            break
    return out


def _ensure_grass_blade_proto(stage, mats: Dict[str, str], proto_path: str = "/World/ReplayLooks/GrassBladeProto") -> None:
    from pxr import Gf, Sdf, UsdGeom

    if stage.GetPrimAtPath(proto_path).IsValid():
        return
    xf = UsdGeom.Xform.Define(stage, Sdf.Path(proto_path))
    xf.ClearXformOpOrder()
    xf.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, 0.0))
    cyl_path = f"{proto_path}/Stem"
    cyl = UsdGeom.Cylinder.Define(stage, Sdf.Path(cyl_path))
    cyl.CreateRadiusAttr(0.02)
    cyl.CreateHeightAttr(0.36)
    _bind_preview_material(stage, cyl_path, mats["grass"])


def _add_grass_on_rectangles(stage, pack: LabReplayData, mats: Dict[str, str], rng: random.Random) -> int:
    """Плотная трава только на верхней грани прямоугольных блоков (игровой газон)."""
    from pxr import Gf, Sdf, UsdGeom, Vt

    x_min, x_max, y_min, y_max = pack.bounds_xy
    span = max(x_max - x_min, y_max - y_min, 10.0)
    total = 0
    _ensure_grass_blade_proto(stage, mats)
    proto_path = "/World/ReplayLooks/GrassBladeProto"
    for i, r in enumerate(pack.rectangles):
        hz = float(max(0.4, 0.014 * min(x_max - x_min, y_max - y_min)))
        half_w = float(r.w) * 0.5
        half_h = float(r.h) * 0.5
        z_top = hz + 0.04
        inst_path = f"/World/ReplayGrassRect_{i}"
        if stage.GetPrimAtPath(inst_path).IsValid():
            continue
        inst = UsdGeom.PointInstancer.Define(stage, Sdf.Path(inst_path))
        inst.CreatePrototypesRel().SetTargets([Sdf.Path(proto_path)])
        step = float(max(0.09, min(0.22, 0.0042 * span)))
        ax, bx = float(r.x) - half_w + 0.06, float(r.x) + half_w - 0.06
        ay, by = float(r.y) - half_h + 0.06, float(r.y) + half_h - 0.06
        if bx <= ax or by <= ay:
            continue
        positions: List[Gf.Vec3f] = []
        for gx in _frange(ax, bx, step):
            for gy in _frange(ay, by, step):
                positions.append(
                    Gf.Vec3f(float(gx + rng.uniform(-0.03, 0.03)), float(gy + rng.uniform(-0.03, 0.03)), z_top)
                )
        n = len(positions)
        if n == 0:
            continue
        inst.CreateProtoIndicesAttr(Vt.IntArray([0] * n))
        inst.CreatePositionsAttr(Vt.Vec3fArray(positions))
        scales = [Gf.Vec3f(rng.uniform(0.85, 1.2), rng.uniform(0.9, 1.35), rng.uniform(0.85, 1.15)) for _ in range(n)]
        inst.CreateScalesAttr(Vt.Vec3fArray(scales))
        rots: List[Gf.Quath] = []
        for _ in range(n):
            yaw = rng.uniform(0.0, 360.0)
            qd = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), yaw).GetQuat()
            im = qd.GetImaginary()
            rots.append(Gf.Quath(float(qd.GetReal()), float(im[0]), float(im[1]), float(im[2])))
        inst.CreateOrientationsAttr(Vt.QuathArray(rots))
        total += n
    if total > 0:
        print(f"[isaac5_replay] grass on rects: PointInstancer blades={total}", flush=True)
    return total


def _build_snow_cover(
    stage,
    pack: LabReplayData,
    mats: Dict[str, str],
    *,
    cell: float = 2.35,
    stamp_radius_m: Optional[float] = None,
    parity_note: str = "",
) -> Optional[_SnowCoverState]:
    """Тонкая сетка сугробов над асфальтом; роботы «растаптывают» ячейки (MakeInvisible)."""
    import math
    import numpy as np
    from isaacsim.core.api.objects import VisualCuboid

    x_min, x_max, y_min, y_max = pack.bounds_xy
    span_x = x_max - x_min
    span_y = y_max - y_min
    if span_x <= 1e-3 or span_y <= 1e-3:
        return None
    cell = float(max(cell, 1e-6))
    # Same discretization strategy as 2D coverage overlays: ceil(span / grid_resolution_m).
    nx = max(1, int(math.ceil(span_x / cell)))
    ny = max(1, int(math.ceil(span_y / cell)))
    cell_x = span_x / nx
    cell_y = span_y / ny
    cell_area_m2 = float(max(cell_x * cell_y, 1e-9))
    # Default stamp radius makes circular clear area ~= one 2D visited cell area.
    stamp_radius = float(stamp_radius_m) if stamp_radius_m is not None and stamp_radius_m > 0.0 else float(
        math.sqrt(cell_area_m2 / math.pi)
    )
    snow_z = 0.045
    paths: List[str] = []
    parent = "/World/SnowCover"
    if not stage.GetPrimAtPath(parent).IsValid():
        from pxr import Sdf, UsdGeom

        UsdGeom.Scope.Define(stage, Sdf.Path(parent))
    for iy in range(ny):
        for ix in range(nx):
            cx = x_min + (ix + 0.5) * cell_x
            cy = y_min + (iy + 0.5) * cell_y
            pth = f"{parent}/c_{ix}_{iy}"
            VisualCuboid(
                prim_path=pth,
                name=f"snow_{ix}_{iy}",
                position=np.array([cx, cy, snow_z], dtype=float),
                scale=np.array([cell_x * 0.992, cell_y * 0.992, 0.055], dtype=float),
                size=1.0,
                color=np.array([c / 255.0 for c in C_SNOW], dtype=float),
            )
            _bind_preview_material(stage, pth, mats["snow"])
            paths.append(pth)
    st = _SnowCoverState(x_min, y_min, cell_x, cell_y, nx, ny, tuple(paths), stamp_radius_m=stamp_radius)
    stamp_area = math.pi * stamp_radius * stamp_radius
    parity_tail = parity_note.strip() if parity_note.strip() else f"1 cell ~= {cell_area_m2:.3f}m2"
    print(
        f"[isaac5_replay] snow stamp: cell={cell_x:.3f}x{cell_y:.3f}m radius={stamp_radius:.3f}m "
        f"area~={stamp_area:.3f}m2 (2D parity: {parity_tail})",
        flush=True,
    )
    return st


def _frange(a: float, b: float, step: float) -> List[float]:
    if step <= 1e-9:
        return [a, b]
    out: List[float] = []
    x = a
    while x <= b + 1e-6:
        out.append(float(x))
        x += step
    return out


def _build_static_scene(
    pack: LabReplayData,
    *,
    visual_style: str = "realistic",
    rng: Optional[random.Random] = None,
    snow_cell_m: float = 2.35,
    snow_stamp_radius_m: Optional[float] = None,
    snow_parity_note: str = "",
) -> Tuple[Dict[str, str], Optional[_SnowCoverState]]:
    import numpy as np
    import omni.usd
    from isaacsim.core.api.objects import VisualCuboid, VisualCylinder
    from isaacsim.core.api.objects.ground_plane import GroundPlane
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    GroundPlane(prim_path="/World/GroundPlane", z_position=0.0)

    stage = omni.usd.get_context().get_stage()
    # Дефолтный GroundPlane рисует белую сетку на Z=0 и визуально «съедает» асфальт/снег сверху.
    if visual_style == "realistic":
        gp = stage.GetPrimAtPath("/World/GroundPlane")
        if gp.IsValid():
            UsdGeom.Imageable(gp).MakeInvisible()
            print(
                "[isaac5_replay] GroundPlane hidden (realistic): иначе белая сетка Isaac перекрывает асфальт и снег.",
                flush=True,
            )
    mats = _ensure_replay_materials(stage)
    snow_out: Optional[_SnowCoverState] = None

    span_xy = max(
        pack.bounds_xy[1] - pack.bounds_xy[0],
        pack.bounds_xy[3] - pack.bounds_xy[2],
        10.0,
    )
    # Камера GIF часто top-down: при «солнце сверху» кубы/деревья читаются плоско.
    # Ключ — низкий угол над XY (лучи вдоль плоскости + умеренный подъём по Z) + тени;
    # dome ослаблен, чтобы не забивать контраст.
    key = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
    key.CreateIntensityAttr(float(5200.0 + 48.0 * span_xy**0.5))
    key.CreateAngleAttr(0.52)
    prim_key = key.GetPrim()
    key_xf = UsdGeom.Xformable(prim_key)
    key_xf.ClearXformOpOrder()
    key_xf.AddRotateXYZOp().Set(Gf.Vec3f(-22.0, 58.0, 0.0))
    for attr_name in ("inputs:enableShadows", "inputs:shadow:enabled"):
        a = prim_key.GetAttribute(attr_name)
        if not a:
            try:
                prim_key.CreateAttribute(attr_name, Sdf.ValueTypeNames.Bool).Set(True)
            except Exception:
                pass
        else:
            try:
                a.Set(True)
            except Exception:
                pass
    fill = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantFill"))
    fill.CreateIntensityAttr(float(520.0 + 7.5 * span_xy**0.5))
    fill.CreateAngleAttr(1.35)
    fill_xf = UsdGeom.Xformable(fill.GetPrim())
    fill_xf.ClearXformOpOrder()
    fill_xf.AddRotateXYZOp().Set(Gf.Vec3f(-18.0, -128.0, 0.0))
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/ReplayDomeFill"))
    dome.CreateIntensityAttr(float(88.0 + 0.55 * span_xy))
    dome.CreateColorAttr(Gf.Vec3f(0.82, 0.88, 1.0))

    x_min, x_max, y_min, y_max = pack.bounds_xy
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    span = max(x_max - x_min, y_max - y_min, 10.0)
    floor_h = max(0.06, 0.004 * min(x_max - x_min, y_max - y_min, span))
    VisualCuboid(
        prim_path="/World/LabFloorTint",
        name="lab_floor",
        position=np.array([cx, cy, -floor_h * 0.5]),
        scale=np.array([span * 1.05, span * 1.05, floor_h]),
        size=1.0,
        color=np.array([c / 255.0 for c in C_ASPHALT], dtype=float),
    )
    _bind_preview_material(stage, "/World/LabFloorTint", mats["asphalt"])

    for i, d in enumerate(pack.disks):
        path = f"/World/Obstacles/disk_{i}"
        h = max(0.35, 0.012 * min(x_max - x_min, y_max - y_min))
        VisualCylinder(
            prim_path=path,
            name=f"disk_{i}",
            position=np.array([d.x, d.y, h * 0.5]),
            radius=float(d.r),
            height=h,
            color=np.array(C_TREE, dtype=float),
        )
        _bind_preview_material(stage, path, mats["tree"])

    for i, r in enumerate(pack.rectangles):
        path = f"/World/Obstacles/rect_{i}"
        hz = max(0.4, 0.014 * min(x_max - x_min, y_max - y_min))
        VisualCuboid(
            prim_path=path,
            name=f"rect_{i}",
            position=np.array([r.x, r.y, hz * 0.5]),
            scale=np.array([r.w, r.h, hz]),
            size=1.0,
            color=np.array([c / 255.0 for c in C_DIRT], dtype=float),
        )
        _bind_preview_material(stage, path, mats["dirt"])

    span_x = x_max - x_min
    span_y = y_max - y_min
    thick = float(max(0.12, min(1.5, 0.006 * min(span_x, span_y))))
    wall_h = float(max(0.75, min(4.0, 0.045 * min(span_x, span_y))))
    hz = wall_h * 0.5
    # Четыре грани рабочей области (как «борта» карты)
    walls = [
        ("north", np.array([cx, y_max + thick * 0.5, hz]), np.array([span_x + 2 * thick, thick, wall_h])),
        ("south", np.array([cx, y_min - thick * 0.5, hz]), np.array([span_x + 2 * thick, thick, wall_h])),
        ("east", np.array([x_max + thick * 0.5, cy, hz]), np.array([thick, span_y + 2 * thick, wall_h])),
        ("west", np.array([x_min - thick * 0.5, cy, hz]), np.array([thick, span_y + 2 * thick, wall_h])),
    ]
    for tag, pos, scale in walls:
        pth = f"/World/LabBounds/{tag}"
        VisualCuboid(
            prim_path=pth,
            name=f"bounds_{tag}",
            position=pos,
            scale=scale,
            size=1.0,
            color=np.array(C_BOUNDS, dtype=float),
        )
        _bind_preview_material(stage, pth, mats["bounds"])

    if visual_style == "realistic" and rng is not None:
        try:
            snow_out = _build_snow_cover(
                stage,
                pack,
                mats,
                cell=snow_cell_m,
                stamp_radius_m=snow_stamp_radius_m,
                parity_note=snow_parity_note,
            )
        except Exception as e:
            print(f"[isaac5_replay] WARN: snow cover failed: {e!r}", flush=True)
            snow_out = None
        try:
            _add_grass_on_rectangles(stage, pack, mats, rng)
        except Exception as e:
            print(f"[isaac5_replay] WARN: rect grass failed: {e!r}", flush=True)

    return mats, snow_out


def _spawn_agents(
    pack: LabReplayData,
    mats: Dict[str, str],
    *,
    robot_size: float = 0.42,
    ped_size: float = 0.34,
    visual_style: str = "realistic",
    rng: Optional[random.Random] = None,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, float], Dict[str, float]]:
    """Возвращает пути корневых примов агентов и мировую Z центра (для _set_translate)."""
    import numpy as np
    import omni.usd
    from isaacsim.core.api.objects import VisualCapsule, VisualCuboid

    stage = omni.usd.get_context().get_stage()
    robot_paths: Dict[str, str] = {}
    ped_paths: Dict[str, str] = {}
    robot_z: Dict[str, float] = {}
    ped_z: Dict[str, float] = {}

    if visual_style == "lab" or rng is None:
        z_r = robot_size * 0.5
        ped_foot = max(0.12, ped_size * 0.4)
        ped_height = max(0.55, ped_size * 2.05)
        pz = float(ped_height * 0.5)
        for name in sorted(pack.robot_paths.keys()):
            path = f"/World/Robots/{name}"
            VisualCuboid(
                prim_path=path,
                name=name,
                position=np.array([0.0, 0.0, z_r]),
                size=float(robot_size),
                color=np.array(C_ROBOT, dtype=float),
            )
            _bind_preview_material(stage, path, mats["robot"])
            robot_paths[name] = path
            robot_z[name] = float(z_r)
        for name in sorted(pack.pedestrian_paths.keys()):
            path = f"/World/Pedestrians/{name}"
            VisualCuboid(
                prim_path=path,
                name=name,
                position=np.array([0.0, 0.0, pz]),
                scale=np.array([ped_foot, ped_foot, ped_height]),
                size=1.0,
                color=np.array(C_HUMAN, dtype=float),
            )
            _bind_preview_material(stage, path, mats["human"])
            ped_paths[name] = path
            ped_z[name] = float(pz)
        return robot_paths, ped_paths, robot_z, ped_z

    rover_usd = (os.environ.get("COVERAGE_LAB_REPLAY_ROVER_USD") or "").strip() or _find_usd_filename("nova_carter.usd")
    ped_env_usd = (os.environ.get("COVERAGE_LAB_REPLAY_PED_USD") or "").strip()

    def _env_float(name: str, default: float = 0.0) -> float:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print(f"[isaac5_replay] WARN: {name}='{raw}' is not float; using {default}.", flush=True)
            return float(default)

    rover_extra_rot = (
        _env_float("COVERAGE_LAB_REPLAY_ROVER_EXTRA_RX", 0.0),
        _env_float("COVERAGE_LAB_REPLAY_ROVER_EXTRA_RY", 0.0),
        _env_float("COVERAGE_LAB_REPLAY_ROVER_EXTRA_RZ", 0.0),
    )
    ped_extra_rot = (
        _env_float("COVERAGE_LAB_REPLAY_PED_EXTRA_RX", 0.0),
        _env_float("COVERAGE_LAB_REPLAY_PED_EXTRA_RY", 0.0),
        _env_float("COVERAGE_LAB_REPLAY_PED_EXTRA_RZ", 0.0),
    )
    rover_scale_mult = max(0.25, _env_float("COVERAGE_LAB_REPLAY_ROVER_SCALE_MULT", 1.0))

    if rover_usd:
        print(f"[isaac5_replay] rover USD candidate: {rover_usd}", flush=True)
    else:
        print(
            "[isaac5_replay] rover USD candidate: not found (set COVERAGE_LAB_REPLAY_ROVER_USD to force a model).",
            flush=True,
        )
    if any(abs(v) > 1e-6 for v in rover_extra_rot):
        print(f"[isaac5_replay] rover extra rotate XYZ(deg): {rover_extra_rot}", flush=True)
    if any(abs(v) > 1e-6 for v in ped_extra_rot):
        print(f"[isaac5_replay] pedestrian extra rotate XYZ(deg): {ped_extra_rot}", flush=True)
    if abs(rover_scale_mult - 1.0) > 1e-6:
        print(f"[isaac5_replay] rover scale multiplier from env: {rover_scale_mult:.3f}", flush=True)

    ped_candidates_raw: List[str] = []
    if ped_env_usd:
        ped_candidates_raw.append(ped_env_usd)
    ped_candidates_raw.extend(_find_metropolis_mesh_character_usds(limit=10))
    maybe_test = _find_metropolis_test_character_usd()
    if maybe_test:
        ped_candidates_raw.append(maybe_test)
    maybe_skeleton = _find_usd_filename("human_skeleton.usd")
    if maybe_skeleton:
        ped_candidates_raw.append(maybe_skeleton)

    ped_candidates: List[str] = []
    seen_ped: set[str] = set()
    for p in ped_candidates_raw:
        key = str(p).strip()
        if not key:
            continue
        norm = key.replace("\\", "/").lower()
        if norm in seen_ped:
            continue
        seen_ped.add(norm)
        ped_candidates.append(key)

    if ped_candidates:
        print(f"[isaac5_replay] pedestrian USD candidates (priority, top {len(ped_candidates)}):", flush=True)
        for i, p in enumerate(ped_candidates[:8], start=1):
            print(f"[isaac5_replay]   {i:02d}) {p}", flush=True)
        if len(ped_candidates) > 8:
            print(f"[isaac5_replay]   ... +{len(ped_candidates) - 8} more", flush=True)
    else:
        print(
            "[isaac5_replay] pedestrian USD candidate: not found (set COVERAGE_LAB_REPLAY_PED_USD to force a full character).",
            flush=True,
        )
    if ped_candidates and "human_skeleton.usd" in ped_candidates[0].replace("\\", "/").lower():
        print(
            "[isaac5_replay] WARN: pedestrian USD points to human_skeleton.usd; this can be skeleton-only/poorly visible. "
            "Prefer COVERAGE_LAB_REPLAY_PED_USD with a full character mesh.",
            flush=True,
        )

    def _clear_prim_tree(path: str) -> None:
        from pxr import Sdf

        try:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                stage.RemovePrim(Sdf.Path(path))
        except Exception:
            pass

    def _try_usd(
        root: str,
        usd_path: str,
        *,
        scale: float,
        z_world: float,
        yaw: float,
        extra_rot_xyz: Tuple[float, float, float],
        agent_kind: str,
        agent_name: str,
        require_mesh: bool = False,
        min_bbox_diag: float = 0.0,
        candidate_rank: int = 1,
        total_candidates: int = 1,
    ) -> bool:
        from pxr import Gf, Sdf, Usd, UsdGeom

        _clear_prim_tree(root)
        if not usd_path:
            print(
                f"[isaac5_replay] {agent_kind} '{agent_name}': USD path is empty -> fallback primitive.",
                flush=True,
            )
            return False
        usd_file = Path(usd_path)
        if not usd_file.is_file():
            print(
                f"[isaac5_replay] {agent_kind} '{agent_name}': USD not found at '{usd_path}' -> fallback primitive.",
                flush=True,
            )
            return False
        try:
            usd_ref = str(usd_file.resolve())
            UsdGeom.Xform.Define(stage, Sdf.Path(root))
            orient_path = f"{root}/orient"
            UsdGeom.Xform.Define(stage, Sdf.Path(orient_path))
            ori = UsdGeom.Xformable(stage.GetPrimAtPath(orient_path))
            ori.ClearXformOpOrder()
            ori.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, float(yaw)))
            payload_path = f"{orient_path}/payload"
            pp = stage.DefinePrim(Sdf.Path(payload_path), "Xform")
            pp.GetReferences().AddReference(usd_ref)
            pxf = UsdGeom.Xformable(pp)
            pxf.ClearXformOpOrder()
            if any(abs(v) > 1e-6 for v in extra_rot_xyz):
                rv = Gf.Vec3d(float(extra_rot_xyz[0]), float(extra_rot_xyz[1]), float(extra_rot_xyz[2]))
                rot_attr = pp.GetAttribute("xformOp:rotateXYZ")
                if rot_attr.IsValid():
                    rot_attr.Set(rv)
                else:
                    set_rot = False
                    for op in pxf.GetOrderedXformOps():
                        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                            op.Set(rv)
                            set_rot = True
                            break
                    if not set_rot:
                        pxf.AddRotateXYZOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(rv)
            scale_v = Gf.Vec3d(float(scale), float(scale), float(scale))
            scale_attr = pp.GetAttribute("xformOp:scale")
            if scale_attr.IsValid():
                scale_attr.Set(scale_v)
            else:
                set_scale = False
                for op in pxf.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                        op.Set(scale_v)
                        set_scale = True
                        break
                if not set_scale:
                    pxf.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(scale_v)
            _set_translate(stage, root, (0.0, 0.0, float(z_world)))

            mesh_count = 0
            for prim in Usd.PrimRange(pp):
                if prim.IsA(UsdGeom.Mesh):
                    mesh_count += 1
            diag = -1.0
            try:
                bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_])
                bbox = bbox_cache.ComputeWorldBound(pp).GetRange()
                if not bbox.IsEmpty():
                    sz = bbox.GetSize()
                    diag = float(math.sqrt(float(sz[0]) ** 2 + float(sz[1]) ** 2 + float(sz[2]) ** 2))
            except Exception:
                diag = -1.0
            print(
                f"[isaac5_replay] {agent_kind} '{agent_name}': USD load OK '{usd_ref}' "
                f"(cand={candidate_rank}/{total_candidates}, scale={scale:.3f}, z={z_world:.3f}, "
                f"yaw={yaw:.1f}, meshes={mesh_count}, bbox_diag={diag:.3f}).",
                flush=True,
            )
            if mesh_count == 0:
                extra_hint = ""
                if agent_kind == "pedestrian":
                    extra_hint = " Set COVERAGE_LAB_REPLAY_PED_USD to a full-body mesh character USD."
                print(
                    f"[isaac5_replay] WARN: {agent_kind} '{agent_name}' USD has no Mesh prims (skeleton-only/invisible possible)."
                    f"{extra_hint}",
                    flush=True,
                )
                if require_mesh:
                    _clear_prim_tree(root)
                    return False
            if 0.0 < diag < 0.12:
                print(
                    f"[isaac5_replay] WARN: {agent_kind} '{agent_name}' bbox_diag={diag:.3f}m looks tiny; "
                    "increase scale or set COVERAGE_LAB_REPLAY_*_EXTRA_RX/RY/RZ if axis is wrong.",
                    flush=True,
                )
            if min_bbox_diag > 0.0 and 0.0 < diag < float(min_bbox_diag):
                print(
                    f"[isaac5_replay] WARN: {agent_kind} '{agent_name}' bbox_diag={diag:.3f}m < {min_bbox_diag:.3f}m; "
                    "trying next USD candidate.",
                    flush=True,
                )
                _clear_prim_tree(root)
                return False
            return True
        except Exception as e:
            print(
                f"[isaac5_replay] {agent_kind} '{agent_name}': USD load FAILED '{usd_path}' with {e!r} -> fallback primitive.",
                flush=True,
            )
            _clear_prim_tree(root)
            return False

    # Капсулы «стоя» вдоль Z: локальная ось капсулы Y → world Z (поворот +90° вокруг X).
    qx90 = (0.7071067811865476, 0.7071067811865476, 0.0, 0.0)

    for _i, name in enumerate(sorted(pack.robot_paths.keys())):
        path = f"/World/Robots/{name}"
        # Nova Carter в метрах ~1 м; на большой карте scale 0.2 давал «точку» сверху.
        base = max(0.72, min(3.4, (robot_size * 1.9) / 0.95))
        sc = float(base * rover_scale_mult * rng.uniform(0.94, 1.1))
        yaw = float(rng.uniform(0.0, 360.0))
        rz = float(max(0.06, robot_size * 0.22))
        if _try_usd(
            path,
            rover_usd or "",
            scale=sc,
            z_world=rz,
            yaw=yaw,
            extra_rot_xyz=rover_extra_rot,
            agent_kind="robot",
            agent_name=name,
            require_mesh=True,
            min_bbox_diag=0.18,
        ):
            robot_paths[name] = path
            robot_z[name] = rz
            continue
        h_body = float(max(0.14, robot_size * (0.55 + 0.55 * rng.random())))
        rad = float(max(0.11, robot_size * (0.38 + 0.35 * rng.random())))
        tint = np.array(
            [
                min(1.0, (C_ROBOT[0] / 255.0) * rng.uniform(0.82, 1.12)),
                min(1.0, (C_ROBOT[1] / 255.0) * rng.uniform(0.88, 1.08)),
                min(1.0, (C_ROBOT[2] / 255.0) * rng.uniform(0.9, 1.15)),
            ],
            dtype=float,
        )
        VisualCapsule(
            prim_path=path,
            name=name,
            position=np.array([0.0, 0.0, rad + 0.5 * h_body]),
            orientation=np.array(qx90, dtype=float),
            radius=rad,
            height=h_body,
            color=tint,
        )
        _bind_preview_material(stage, path, mats["robot"])
        robot_paths[name] = path
        robot_z[name] = float(rad + 0.5 * h_body)

    for _j, name in enumerate(sorted(pack.pedestrian_paths.keys())):
        path = f"/World/Pedestrians/{name}"
        base = max(0.65, min(2.8, ped_size * 4.2))
        sc = float(base * rng.uniform(0.94, 1.06))
        yaw = float(rng.uniform(0.0, 360.0))
        pz = float(max(0.03, ped_size * 0.16))
        ped_ok = False
        for cand_i, ped_usd in enumerate(ped_candidates, start=1):
            if not _try_usd(
                path,
                ped_usd,
                scale=sc,
                z_world=pz,
                yaw=yaw,
                extra_rot_xyz=ped_extra_rot,
                agent_kind="pedestrian",
                agent_name=name,
                require_mesh=True,
                min_bbox_diag=0.22,
                candidate_rank=cand_i,
                total_candidates=max(1, len(ped_candidates)),
            ):
                continue
            ped_ok = True
            break
        if ped_ok:
            ped_paths[name] = path
            ped_z[name] = pz
            continue
        rad = float(max(0.16, ped_size * (0.48 + 0.25 * rng.random())))
        h_body = float(max(0.62, ped_size * (1.55 + 0.85 * rng.random())))
        tint = np.array(
            [
                min(1.0, (C_HUMAN[0] / 255.0) * rng.uniform(0.85, 1.18)),
                min(1.0, (C_HUMAN[1] / 255.0) * rng.uniform(0.88, 1.12)),
                min(1.0, (C_HUMAN[2] / 255.0) * rng.uniform(0.9, 1.1)),
            ],
            dtype=float,
        )
        VisualCapsule(
            prim_path=path,
            name=name,
            position=np.array([0.0, 0.0, rad + 0.5 * h_body]),
            orientation=np.array(qx90, dtype=float),
            radius=rad,
            height=h_body,
            color=tint,
        )
        _bind_preview_material(stage, path, mats["human"])
        ped_paths[name] = path
        ped_z[name] = float(rad + 0.5 * h_body)

    return robot_paths, ped_paths, robot_z, ped_z


def _print_replay_scene_summary(
    pack: LabReplayData,
    robot_prims: Dict[str, str],
    ped_prims: Dict[str, str],
    *,
    visual_style: str = "realistic",
) -> None:
    """Сколько статики/агентов реально заведено в USD (без догадок по Outliner)."""
    grass = " + grass on rect tops (PointInstancer)" if visual_style == "realistic" else ""
    snow = " + snow grid (tracks)" if visual_style == "realistic" else ""
    if visual_style == "realistic":
        ag = (
            f"agents: {len(robot_prims)} robots, {len(ped_prims)} pedestrians "
            "(USD refs when found under ISAAC_PATH, else capsules)."
        )
    else:
        ag = (
            f"agents: {len(robot_prims)} robot cuboids, {len(ped_prims)} pedestrian cuboids "
            "(PreviewSurface on Mesh/Cube/Cylinder under each prim)."
        )
    print(
        "[isaac5_replay] USD content: "
        f"asphalt slab + {len(pack.disks)} cylinders (trees) + {len(pack.rectangles)} dirt blocks "
        f"+ 4 bounds walls + lights{grass}{snow}; "
        f"{ag}",
        flush=True,
    )


def _replay_loop(
    pack: LabReplayData,
    indices: List[int],
    robot_prims: Dict[str, str],
    ped_prims: Dict[str, str],
    *,
    robot_z: Dict[str, float],
    ped_z: Dict[str, float],
    simulation_app,
    updates_per_frame: int,
    stride: int,
    replay_speed: float,
    gif_tmp_dir: Optional[Path] = None,
    gif_viewport: Any = None,
    gif_camera_prim: Optional[str] = None,
    gif_post_updates: int = 6,
    snow_state: Optional[_SnowCoverState] = None,
    snow_stamp_radius: Optional[float] = None,
    gif_substeps: int = 1,
) -> None:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    n_idx = len(indices)
    step_dt = float(pack.dt_sec) * max(1, int(stride))
    sleep_s = (step_dt / replay_speed) if replay_speed > 0.0 else 0.0
    record_gif = gif_tmp_dir is not None and gif_viewport is not None
    gs = max(1, int(gif_substeps)) if record_gif else 1
    fi_list = _gif_sample_indices_float(indices, gs) if record_gif else [float(i) for i in indices]
    n_play = len(fi_list)
    sleep_per = (
        (sleep_s / max(1, gs)) if (record_gif and gs > 1 and sleep_s > 0.0) else sleep_s
    )
    log_every = max(1, n_play // 12) if n_play > 12 else 1
    active_snow_radius = (
        float(snow_stamp_radius)
        if snow_stamp_radius is not None
        else (float(snow_state.stamp_radius_m) if snow_state is not None else 0.0)
    )

    if record_gif and gs > 1:
        print(
            f"[isaac5_replay] GIF substeps={gs} -> {n_play} capture frames (trajectory keys={n_idx})",
            flush=True,
        )

    for gif_k, fi_f in enumerate(fi_list):
        for name, ppath in robot_prims.items():
            path_pts = pack.robot_paths[name]
            x, y = _xy_on_path_lerp(path_pts, fi_f)
            rz = float(robot_z.get(name, 0.0))
            _set_translate(stage, ppath, (float(x), float(y), rz))
            op = f"{ppath}/orient"
            if stage.GetPrimAtPath(op).IsValid():
                ydeg = _heading_deg_on_path(path_pts, fi_f)
                _set_orient_yaw(stage, op, ydeg)
        for name, ppath in ped_prims.items():
            pts = pack.pedestrian_paths.get(name) or []
            if not pts:
                continue
            x, y = _xy_on_path_lerp(pts, fi_f)
            pz = float(ped_z.get(name, 0.0))
            _set_translate(stage, ppath, (float(x), float(y), pz))
            op = f"{ppath}/orient"
            if stage.GetPrimAtPath(op).IsValid():
                ydeg = _heading_deg_on_path(pts, fi_f)
                _set_orient_yaw(stage, op, ydeg)
        if snow_state is not None and active_snow_radius > 0.0:
            for name, _ppath in robot_prims.items():
                path_pts = pack.robot_paths[name]
                rx, ry = _xy_on_path_lerp(path_pts, fi_f)
                snow_state.stamp_clear(stage, float(rx), float(ry), active_snow_radius)
        for _ in range(max(1, int(updates_per_frame))):
            simulation_app.update()
        if record_gif:
            for _ in range(max(0, int(gif_post_updates))):
                simulation_app.update()
            if gif_camera_prim:
                _set_viewport_active_camera(gif_viewport, gif_camera_prim, simulation_app)
            png_path = gif_tmp_dir / f"frame_{gif_k:05d}.png"
            try:
                _capture_viewport_png_sync(gif_viewport, png_path, simulation_app)
            except Exception as e:
                print(f"[isaac5_replay] WARN: GIF frame {gif_k + 1}/{n_play}: {e!r}", flush=True)
        if sleep_per > 0.0:
            time.sleep(sleep_per)
        if gif_k % log_every == 0 or gif_k == n_play - 1:
            print(f"[isaac5_replay] кадр {gif_k + 1}/{n_play} (path_index~{fi_f:.2f})", flush=True)


def _env_compat_mode() -> str:
    """
    COVERAGE_LAB_ISAAC_COMPAT (если переменная не задана — режим выключен):
      0 / off / false — выключено
      1 / on / true — только D3D12 (рекомендуется; так задаёт run_isaac_replay_large_complex.bat)
      2 / aggressive / soft / full — D3D12 + compatibilityMode (только если без него всё ещё AV;
          часто даёт 'HydraEngine rtx failed' и чёрный viewport)
    """
    raw = (os.environ.get("COVERAGE_LAB_ISAAC_COMPAT") or "").strip()
    if not raw:
        return "off"
    v = raw.lower()
    if v in ("0", "false", "no", "off", "disable", "none"):
        return "off"
    if v in ("2", "aggressive", "soft", "full", "compat", "legacy"):
        return "aggressive"
    return "d3d12"


def _compat_extra_args(*, aggressive: bool) -> List[str]:
    """Kit CLI: Vulkan off; optional compatibilityMode (часто ломает RTX viewport — только по флагу)."""
    out = ["--/app/vulkan=false"]
    if aggressive:
        out.append("--/exts/omni.kit.renderer.core/compatibilityMode=true")
    return out


def _kick_viewport_and_frame(
    pack: LabReplayData,
    simulation_app,
    stage,
    *,
    camera: str = "angled",
    warmup: int = 12,
) -> Tuple[Optional[Any], Optional[str]]:
    """Камера: angled (диагональ), topdown, или framed (auto-frame по примам).

    Возвращает (viewport_обёртка, путь_камеры_для_rebind). Для framed путь None —
    используется встроенная камера после frame_viewport_prims.
    """
    cam_path: Optional[str] = None
    vp: Optional[Any] = None
    for _ in range(max(1, warmup)):
        simulation_app.update()
    try:
        from omni.kit.viewport.utility import frame_viewport_prims, get_active_viewport_and_window

        vp, vp_window = get_active_viewport_and_window()
        if not vp:
            return None, None
        if vp_window is not None and callable(getattr(vp_window, "focus", None)):
            try:
                vp_window.focus()
            except Exception:
                pass
        if camera == "framed":
            prims = [
                "/World/LabFloorTint",
                "/World/LabBounds/north",
                "/World/LabBounds/south",
                "/World/LabBounds/east",
                "/World/LabBounds/west",
            ]
            frame_viewport_prims(vp, [p for p in prims if stage.GetPrimAtPath(p).IsValid()])
        elif camera == "topdown":
            cp = _create_top_down_gif_camera_rig(stage, pack)
            _set_viewport_active_camera(vp, cp, simulation_app)
            cam_path = cp
        else:
            cp = _create_angled_camera_rig(stage, pack)
            _set_viewport_active_camera(vp, cp, simulation_app)
            cam_path = cp
    except Exception:
        vp = None
        cam_path = None
    # RTX/Hydra часто даёт «пустой» или однотонный первый кадр, пока не сдвинут viewport;
    # вращение мышью форсирует перерисовку — дублируем тем же лишними update().
    if vp is not None:
        _try_hide_viewport_grid()
        for _ in range(36):
            simulation_app.update()
    return vp, cam_path


def _create_angled_camera_rig(stage, pack: LabReplayData) -> str:
    """Камера с диагонали: центр карты в кадре, запас по краям (меньше «кривой» обрезки)."""
    import math

    import numpy as np
    from pxr import Gf, Sdf, UsdGeom

    x_min, x_max, y_min, y_max = pack.bounds_xy
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    span_x = max(x_max - x_min, 1e-6)
    span_y = max(y_max - y_min, 1e-6)
    span = max(span_x, span_y, 5.0)
    # Чуть выше и менее «вдоль пола», иначе без вращения viewport часто видна почти одна плоскость.
    eye = np.array([cx + span * 0.52, cy - span * 0.48, span * 1.05 + 12.0], dtype=float)
    target = np.array([cx, cy, 0.0], dtype=float)
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    f = target - eye
    fn = float(np.linalg.norm(f))
    if fn < 1e-9:
        eye = np.array([cx + span * 0.5, cy - span * 0.5, span + 5.0], dtype=float)
        f = target - eye
        fn = float(np.linalg.norm(f))
    f = f / fn
    r = np.cross(up, f)
    rn = float(np.linalg.norm(r))
    if rn < 1e-6:
        up = np.array([0.0, 1.0, 0.0], dtype=float)
        r = np.cross(up, f)
        rn = float(np.linalg.norm(r))
    r = r / rn
    u = np.cross(f, r)
    M = Gf.Matrix4d(
        r[0],
        u[0],
        -f[0],
        0.0,
        r[1],
        u[1],
        -f[1],
        0.0,
        r[2],
        u[2],
        -f[2],
        0.0,
        eye[0],
        eye[1],
        eye[2],
        1.0,
    )
    rig_path = "/World/ReplaySceneCamRig"
    cam_prim_path = f"{rig_path}/Camera"
    xf = UsdGeom.Xform.Define(stage, Sdf.Path(rig_path))
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(M)
    cam = UsdGeom.Camera.Define(stage, Sdf.Path(cam_prim_path))
    d = float(np.linalg.norm(target - eye))
    margin = 1.22
    half_extent = 0.55 * max(span_x, span_y)
    vfov = 2.0 * math.atan(margin * half_extent / max(d, 1e-6))
    vfov = max(0.32, min(vfov, 1.38))
    fl = 18.0
    v_ap = 2.0 * fl * math.tan(vfov * 0.5)
    h_ap = v_ap * (16.0 / 9.0)
    cam.CreateFocalLengthAttr(fl)
    cam.CreateVerticalApertureAttr(v_ap)
    cam.CreateHorizontalApertureAttr(h_ap)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, d * 6.0))
    return cam_prim_path


def _create_top_down_gif_camera_rig(stage, pack: LabReplayData) -> str:
    """Ортогональный вид сверху: ось взгляда камеры = −world Z, плоскость кадра || XY (перпендикулярно карте).

    Кадр по **внешнему** контуру арены (bounds + толщина бордов), с маленьким margin — по краям в основном
    только коралловые стены, без лишней серой сетки за пределами.
    """
    from pxr import Gf, Sdf, UsdGeom

    ox0, ox1, oy0, oy1 = _outer_arena_xy(pack.bounds_xy)
    cx = 0.5 * (ox0 + ox1)
    cy = 0.5 * (oy0 + oy1)
    span_xo = max(ox1 - ox0, 1e-6)
    span_yo = max(oy1 - oy0, 1e-6)
    half_diag = 0.5 * math.sqrt(span_xo * span_xo + span_yo * span_yo)
    viewport_aspect = 16.0 / 9.0
    margin = 1.006
    ground_half_vertical = max(0.5 * span_yo, 0.5 * span_xo / viewport_aspect) * margin
    z_hi = float(max(ground_half_vertical * 2.35 + 5.0, half_diag * 1.02 + 6.0))

    rig_path = "/World/ReplayCamTopRig"
    cam_prim_path = f"{rig_path}/Camera"
    xf = UsdGeom.Xform.Define(stage, Sdf.Path(rig_path))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(cx), float(cy), float(z_hi)))

    cam = UsdGeom.Camera.Define(stage, Sdf.Path(cam_prim_path))
    d = float(z_hi)
    fl = 18.0
    vfov = 2.0 * math.atan(ground_half_vertical / max(d, 1e-6))
    vfov = max(0.42, min(vfov, 1.45))
    v_ap = 2.0 * fl * math.tan(vfov * 0.5)
    h_ap = v_ap * viewport_aspect
    cam.CreateFocalLengthAttr(fl)
    cam.CreateVerticalApertureAttr(v_ap)
    cam.CreateHorizontalApertureAttr(h_ap)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.08, max(d * 8.0, z_hi * 4.0)))
    print(
        f"[isaac5_replay] topdown camera: outer span {span_xo:.1f} x {span_yo:.1f} m, "
        f"margin={margin}, aspect={viewport_aspect:.3f}, z={z_hi:.1f}",
        flush=True,
    )
    return cam_prim_path


def _set_viewport_active_camera(viewport: Any, camera_prim_path: str, simulation_app) -> bool:
    """Переключить активную камеру viewport (разные версии Kit дают разные методы)."""
    ok = False
    if viewport is None:
        return False
    setters: List[Any] = [getattr(viewport, "set_active_camera", None)]
    vapi = getattr(viewport, "viewport_api", None)
    if vapi is not None:
        setters.append(getattr(vapi, "set_active_camera", None))
    for setter in setters:
        if setter is None:
            continue
        for args in ((camera_prim_path,), (camera_prim_path, ""), (camera_prim_path, None)):
            try:
                setter(*args)
                ok = True
                break
            except TypeError:
                continue
            except Exception:
                break
        if ok:
            break
    for _ in range(12):
        simulation_app.update()
    return ok


def _default_experience_kit() -> Path | None:
    """Первый существующий .kit из ISAAC_PATH (base.python → base → full)."""
    raw = (os.environ.get("ISAAC_PATH") or "").strip().strip('"')
    if not raw:
        return None
    root = Path(raw)
    for name in ("isaacsim.exp.base.python.kit", "isaacsim.exp.base.kit", "isaacsim.exp.full.kit"):
        cand = root / "apps" / name
        if cand.is_file():
            return cand
    return None


def main() -> None:
    args = _parse_args()
    _src = Path(__file__).resolve()
    try:
        _mt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_src.stat().st_mtime))
    except OSError:
        _mt = "?"
    print(
        f"[isaac5_replay] script={_src} mtime_local={_mt} visual_style={args.visual_style}",
        flush=True,
    )
    if args.gif_out is not None and args.headless:
        raise SystemExit("--gif-out несовместим с --headless: нужен видимый viewport для захвата.")
    json_path = args.json_path
    if not json_path.is_file():
        raise SystemExit(f"JSON not found: {json_path}")

    result = json.loads(json_path.read_text(encoding="utf-8"))
    pack = build_lab_replay_data(result, geometry_yaml=args.geometry_yaml)
    stride = max(1, int(args.stride)) if args.stride is not None else (3 if args.gif_out is not None else 1)
    if args.stride is None and args.gif_out is not None:
        print(
            "[isaac5_replay] --stride not set: using stride=3 for --gif-out "
            "(matches 2D coverage_lab render_gif cadence).",
            flush=True,
        )
    n = assert_paths_align_for_replay(pack, stride=stride, max_steps=args.max_steps)
    indices = iter_replay_frame_indices(n, stride=stride, max_steps=args.max_steps)
    if args.gif_out is not None:
        gif_substeps = max(1, min(20, int(args.gif_substeps)))
        gif_frames = 1 + max(0, (len(indices) - 1) * gif_substeps)
        gif_secs = gif_frames / max(1, int(args.gif_fps))
        print(
            f"[isaac5_replay] GIF timing plan: keyframes={len(indices)} stride={stride} "
            f"substeps={gif_substeps} fps={int(args.gif_fps)} -> frames={gif_frames}, ~{gif_secs:.1f}s playback.",
            flush=True,
        )

    try:
        from isaacsim.simulation_app import SimulationApp
    except ImportError as e:
        raise SystemExit(
            "Isaac Sim Python is required (import isaacsim failed). "
            "Run this script with Isaac Sim's python.bat, not the system Python.\n"
            f"Detail: {e}"
        ) from e

    auto_robot, auto_ped = _agent_sizes_from_bounds(pack.bounds_xy)
    robot_size = float(args.robot_size) if args.robot_size is not None else auto_robot
    ped_size = float(args.ped_size) if args.ped_size is not None else auto_ped
    snow_cell_m, snow_stamp_radius_m, snow_parity_note = _resolve_snow_stamp_config(pack, robot_size)
    seed_raw = result.get("seed", 0)
    try:
        seed_i = int(seed_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        seed_i = 0
    rng = random.Random(seed_i ^ 0x9E3779B9)

    config: dict = {
        "headless": bool(args.headless),
        "width": 1280,
        "height": 720,
        "active_gpu": 0,
        "multi_gpu": False,
        "renderer": "RaytracedLighting",
    }
    # Headless + default viewport sync can crash RTX on some drivers; skip viewport when no window.
    if args.headless:
        config["disable_viewport_updates"] = True
        config["hide_ui"] = True
    exp_path = args.experience if args.experience is not None else _default_experience_kit()
    experience_arg = str(Path(exp_path).resolve()) if exp_path is not None and Path(exp_path).is_file() else ""
    merged_extra: List[str] = []
    mode = _env_compat_mode()
    want_compat = args.compat_renderer or mode in ("d3d12", "aggressive")
    aggressive = args.compat_aggressive or mode == "aggressive"
    if want_compat:
        merged_extra.extend(_compat_extra_args(aggressive=aggressive))
        msg = "[isaac5_replay] compat: D3D12 (vulkan=false)"
        if aggressive:
            msg += " + renderer.core.compatibilityMode (aggressive — может быть чёрный RTX viewport)"
        print(msg, flush=True)
    extra = os.environ.get("ISAAC_EXTRA_ARGS", "").strip()
    if extra:
        merged_extra.extend(shlex.split(extra, posix=os.name != "nt"))
    if merged_extra:
        config["extra_args"] = merged_extra
    print(
        "\n[isaac5_replay] Сцена создаётся в ЭТОМ окне Kit (процесс python.bat). "
        "Отдельный Isaac Sim из ярлыка — другой Stage: там карты не будет. "
        "При 'kvdb lock' закройте другие kit.exe в диспетчере задач.\n",
        flush=True,
    )
    if exp_path is not None:
        print(f"[isaac5_replay] kit experience: {exp_path}", flush=True)
    else:
        print("[isaac5_replay] kit experience: (нет .kit на диске — встроенный режим SimulationApp)", flush=True)
    simulation_app = SimulationApp(config, experience=experience_arg)
    print("[isaac5_replay] SimulationApp started; entering scene setup", flush=True)
    gif_tmp: Optional[Path] = None
    gif_vp: Any = None
    try:
        _warmup_simulation_app(simulation_app, frames=args.warmup_frames)
        mats, snow_state = _build_static_scene(
            pack,
            visual_style=args.visual_style,
            rng=rng,
            snow_cell_m=snow_cell_m,
            snow_stamp_radius_m=snow_stamp_radius_m,
            snow_parity_note=snow_parity_note,
        )
        robot_prims, ped_prims, robot_z_map, ped_z_map = _spawn_agents(
            pack,
            mats,
            robot_size=robot_size,
            ped_size=ped_size,
            visual_style=args.visual_style,
            rng=rng,
        )
        _print_replay_scene_summary(pack, robot_prims, ped_prims, visual_style=args.visual_style)
        import omni.usd

        st = omni.usd.get_context().get_stage()
        _set_default_world_prim(st)
        _print_world_stage_hint(st)
        gif_cam_path: Optional[str] = None
        if not args.headless:
            gif_vp, gif_cam_path = _kick_viewport_and_frame(pack, simulation_app, st, camera=args.camera)
            if args.gif_out is not None:
                if gif_vp is None:
                    print("[isaac5_replay] WARN: viewport после kick=None — GIF не пишется.", flush=True)
                else:
                    gif_tmp = Path(tempfile.mkdtemp(prefix="isaac5_replay_gif_"))
                    print(
                        f"[isaac5_replay] GIF capture, camera={args.camera}, temp={gif_tmp} -> {args.gif_out.resolve()}",
                        flush=True,
                    )
                    if gif_cam_path:
                        print(f"[isaac5_replay] GIF viewport camera prim: {gif_cam_path}", flush=True)
                    try:
                        rp_dbg = _normalized_render_product_path(_viewport_api_for_capture(gif_vp))
                        if rp_dbg:
                            print(f"[isaac5_replay] GIF render_product_path: {rp_dbg}", flush=True)
                    except Exception:
                        pass
        print(
            f"[isaac5_replay] scene={pack.scene_name} frames={len(indices)} "
            f"path_len={n} dt={pack.dt_sec}s stride={stride} robots={len(robot_prims)} peds={len(ped_prims)} "
            f"robot~{robot_size:.2f}m ped~{ped_size:.2f}m replay_speed={args.replay_speed}",
            flush=True,
        )
        if args.replay_speed <= 0.0:
            print(
                "[isaac5_replay] Воспроизведение без паузы между кадрами (очень быстро). "
                "Чтобы видеть движение, добавьте например --replay-speed 1",
                flush=True,
            )
        else:
            print("[isaac5_replay] Старт воспроизведения (пауза по dt*stride/speed)...", flush=True)
        _replay_loop(
            pack,
            indices,
            robot_prims,
            ped_prims,
            robot_z=robot_z_map,
            ped_z=ped_z_map,
            simulation_app=simulation_app,
            updates_per_frame=args.updates_per_frame,
            stride=stride,
            replay_speed=args.replay_speed,
            gif_tmp_dir=gif_tmp,
            gif_viewport=gif_vp,
            gif_camera_prim=gif_cam_path,
            gif_post_updates=args.gif_post_updates,
            snow_state=snow_state if args.visual_style == "realistic" else None,
            snow_stamp_radius=snow_stamp_radius_m,
            gif_substeps=max(1, min(20, int(args.gif_substeps))),
        )
        if args.gif_out is not None and gif_tmp is not None and gif_vp is not None:
            png_sorted = sorted(gif_tmp.glob("frame_*.png"))
            _assemble_gif_from_png_folder(
                gif_tmp, args.gif_out, fps=args.gif_fps, max_width=args.gif_max_width
            )
            _print_gif_pixel_quality_report(args.gif_out.resolve(), source_png_paths=png_sorted or None)
        if args.headless:
            print("[isaac5_replay] headless replay done; exiting.")
        elif args.auto_close:
            print("[isaac5_replay] auto-close: выход без ожидания окна.", flush=True)
        else:
            print("[isaac5_replay] trajectory finished; close the window or press Stop to exit.")
            while simulation_app.is_running():
                simulation_app.update()
    finally:
        if gif_tmp is not None:
            shutil.rmtree(gif_tmp, ignore_errors=True)
        simulation_app.close()


if __name__ == "__main__":
    main()
