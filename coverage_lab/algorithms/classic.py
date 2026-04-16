from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from coverage_lab.env.grid_world import GridWorld2D
from coverage_lab.types import LabScene

STAGNATION_EPS = 1e-6
STAGNATION_STEPS = 16


def _grid_dims(scene: LabScene) -> Tuple[int, int, float, float]:
    x_min, x_max, y_min, y_max = scene.bounds_xy
    res = float(scene.grid_resolution_m)
    nx = max(1, int(math.ceil((x_max - x_min) / res)))
    ny = max(1, int(math.ceil((y_max - y_min) / res)))
    return nx, ny, x_min, y_min


def _cell_center(scene: LabScene, cx: int, cy: int) -> np.ndarray:
    x_min, _, y_min, _ = scene.bounds_xy
    res = float(scene.grid_resolution_m)
    return np.array([x_min + (cx + 0.5) * res, y_min + (cy + 0.5) * res], dtype=float)


def _to_cell(scene: LabScene, p: np.ndarray) -> Tuple[int, int]:
    x_min, _, y_min, _ = scene.bounds_xy
    res = float(scene.grid_resolution_m)
    return int((float(p[0]) - x_min) / res), int((float(p[1]) - y_min) / res)


def _nearest_unvisited_cell(
    *,
    scene: LabScene,
    blocked: set[Tuple[int, int]],
    visited: set[Tuple[int, int]],
    start: Tuple[int, int],
    candidates: set[Tuple[int, int]] | None = None,
    forbidden: set[Tuple[int, int]] | None = None,
) -> Tuple[int, int] | None:
    nx, ny, _, _ = _grid_dims(scene)
    avoid = forbidden or set()
    best = None
    bestd = 1e9
    if candidates is None:
        iterable = (
            (cx, cy)
            for cx in range(nx)
            for cy in range(ny)
            if (cx, cy) not in blocked and (cx, cy) not in visited and (cx, cy) not in avoid
        )
    else:
        iterable = (
            c
            for c in candidates
            if c not in blocked and c not in visited and c not in avoid
        )
    for c in iterable:
        d = abs(c[0] - start[0]) + abs(c[1] - start[1])
        if d < bestd:
            bestd = d
            best = c
    return best


class Boustrophedon:
    """Boustrophedon coverage по свободным клеткам (skip visited/blocked)."""

    def __init__(self) -> None:
        self._lists: Dict[str, List[Tuple[int, int]]] = {}
        self._idx: Dict[str, int] = {}
        self._scene_key: Tuple[str, float, Tuple[float, float, float, float]] | None = None
        self._last_cov = 0.0
        self._stagnation = 0

    def choose_targets(self, robot_xy: Dict[str, np.ndarray], *, scene: LabScene, world: GridWorld2D, metrics=None):
        names = sorted(robot_xy.keys())
        nx, ny, _, _ = _grid_dims(scene)
        scene_key = (scene.name, float(scene.grid_resolution_m), tuple(scene.bounds_xy))
        blocked = set(world.obstacle_cells())

        # Build per-robot ordered lists once per scene (balanced round-robin)
        if self._scene_key != scene_key or set(self._lists.keys()) != set(names):
            free: List[Tuple[int, int]] = []
            for cy in range(ny):
                xs = range(nx) if (cy % 2 == 0) else range(nx - 1, -1, -1)
                for cx in xs:
                    if (cx, cy) not in blocked:
                        free.append((cx, cy))
            self._lists = {n: [] for n in names}
            for i, c in enumerate(free):
                self._lists[names[i % len(names)]].append(c)
            self._idx = {n: 0 for n in names}
            self._scene_key = scene_key

        visited = set(getattr(metrics, "visited_cells", set())) if metrics is not None else set()
        if metrics is not None and hasattr(metrics, "coverage_percent"):
            cov = float(metrics.coverage_percent())
            if cov <= self._last_cov + STAGNATION_EPS:
                self._stagnation += 1
            else:
                self._stagnation = 0
            self._last_cov = cov
        force_escape = self._stagnation >= STAGNATION_STEPS
        reserved: set[Tuple[int, int]] = set()

        # Optional: if we already covered almost everything, just hold
        out: Dict[str, np.ndarray] = {}
        for name in names:
            lst = self._lists.get(name, [])
            i = int(self._idx.get(name, 0))
            while i < len(lst) and lst[i] in visited:
                i += 1
            self._idx[name] = i
            if i < len(lst) and not force_escape:
                cx, cy = lst[i]
                c = (cx, cy)
                if c in reserved:
                    cur = _to_cell(scene, robot_xy[name])
                    alt = _nearest_unvisited_cell(
                        scene=scene,
                        blocked=blocked,
                        visited=visited,
                        start=cur,
                        forbidden=reserved,
                    )
                    c = alt if alt is not None else c
                reserved.add(c)
                out[name] = _cell_center(scene, c[0], c[1])
            else:
                # Fallback: nearest unvisited free cell (global)
                # (keeps progress even if round-robin partition got exhausted unevenly)
                cur = _to_cell(scene, robot_xy[name])
                best = _nearest_unvisited_cell(
                    scene=scene,
                    blocked=blocked,
                    visited=visited,
                    start=cur,
                    forbidden=reserved,
                )
                if best is not None:
                    reserved.add(best)
                    out[name] = _cell_center(scene, best[0], best[1])
                else:
                    out[name] = robot_xy[name].copy()
        return out


class STC:
    """STC-like: обход остовного дерева на grid (упрощение: DFS по свободным клеткам)."""

    def __init__(self) -> None:
        self._route: Dict[str, List[Tuple[int, int]]] = {}
        self._idx: Dict[str, int] = {}
        self._last_cov = 0.0
        self._stagnation = 0

    def _build_route(self, scene: LabScene, world: GridWorld2D, start: Tuple[int, int]) -> List[Tuple[int, int]]:
        nx, ny, _, _ = _grid_dims(scene)
        blocked = set(world.obstacle_cells())

        def neigh(c: Tuple[int, int]):
            x, y = c
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx2, ny2 = x + dx, y + dy
                if 0 <= nx2 < nx and 0 <= ny2 < ny and (nx2, ny2) not in blocked:
                    yield (nx2, ny2)

        # DFS spanning tree edges via parent map
        stack = [start]
        parent = {start: None}
        order: List[Tuple[int, int]] = []
        while stack:
            c = stack.pop()
            order.append(c)
            for n in neigh(c):
                if n not in parent:
                    parent[n] = c
                    stack.append(n)

        # Euler tour on tree (approx): walk order, connecting via parents
        route: List[Tuple[int, int]] = []
        cur = start
        route.append(cur)
        for node in order[1:]:
            # climb to LCA (naive via parent chain)
            path_a = set()
            a = cur
            while a is not None:
                path_a.add(a)
                a = parent[a]  # type: ignore[index]
            b = node
            chain_b = []
            while b not in path_a:
                chain_b.append(b)
                b = parent[b]  # type: ignore[index]
                if b is None:
                    break
            lca = b if b is not None else start
            # move cur up to lca
            a = cur
            while a != lca:
                a = parent[a]  # type: ignore[index]
                if a is None:
                    break
                route.append(a)
            # move down to node
            for step in reversed(chain_b):
                route.append(step)
            cur = node
        return route

    def choose_targets(self, robot_xy: Dict[str, np.ndarray], *, scene: LabScene, world: GridWorld2D, metrics=None):
        visited = set(getattr(metrics, "visited_cells", set())) if metrics is not None else set()
        blocked = set(world.obstacle_cells())
        if metrics is not None and hasattr(metrics, "coverage_percent"):
            cov = float(metrics.coverage_percent())
            if cov <= self._last_cov + STAGNATION_EPS:
                self._stagnation += 1
            else:
                self._stagnation = 0
            self._last_cov = cov
        if self._stagnation >= STAGNATION_STEPS:
            # Rebuild routes from current positions when stuck.
            self._route.clear()
            self._idx.clear()
            self._stagnation = 0

        reserved: set[Tuple[int, int]] = set()
        out = {}
        for name, p in robot_xy.items():
            if name not in self._route:
                start = _to_cell(scene, p)
                self._route[name] = self._build_route(scene, world, start)
                self._idx[name] = 0
            idx = int(self._idx.get(name, 0))
            route = self._route[name]
            # Skip already visited cells along the route to avoid wasting steps.
            while idx < len(route) and route[idx] in visited:
                idx += 1
            if idx >= len(route):
                # If we exhausted the route but still haven't reached target coverage, rebuild from current cell.
                start = _to_cell(scene, p)
                self._route[name] = self._build_route(scene, world, start)
                route = self._route[name]
                idx = 0
                while idx < len(route) and route[idx] in visited:
                    idx += 1
            idx = min(idx, len(route) - 1)
            tgt = route[idx]
            if tgt in reserved or tgt in visited:
                cur = _to_cell(scene, p)
                alt = _nearest_unvisited_cell(
                    scene=scene,
                    blocked=blocked,
                    visited=visited,
                    start=cur,
                    forbidden=reserved,
                )
                if alt is not None:
                    tgt = alt
            reserved.add(tgt)
            out[name] = _cell_center(scene, tgt[0], tgt[1])
            self._idx[name] = idx + 1 if idx + 1 < len(route) else idx
        return out


@dataclass
class DARP:
    """DARP-like: разбиение клеток по ближайшему роботу с балансировкой (упрощённо)."""

    inner: str = "boustro"  # boustro|stc

    def __post_init__(self) -> None:
        self._regions: Dict[str, set[Tuple[int, int]]] = {}
        self._planners = {}
        self._recent_targets: Dict[str, List[Tuple[int, int]]] = {}
        self._last_cov = 0.0
        self._stagnation = 0
        self._tick = 0

    def _compute_regions(self, robot_xy: Dict[str, np.ndarray], scene: LabScene, world: GridWorld2D) -> None:
        nx, ny, _, _ = _grid_dims(scene)
        blocked = set(world.obstacle_cells())
        names = sorted(robot_xy.keys())
        # initial assignment by nearest robot
        cells = [(x, y) for x in range(nx) for y in range(ny) if (x, y) not in blocked]
        centers = {n: _to_cell(scene, robot_xy[n]) for n in names}

        assign: Dict[Tuple[int, int], str] = {}
        for c in cells:
            best = None
            bestd = 1e9
            for n in names:
                cx, cy = centers[n]
                d = abs(c[0] - cx) + abs(c[1] - cy)
                if d < bestd:
                    bestd = d
                    best = n
            assign[c] = str(best)

        # simple balancing: move boundary cells from largest region to smallest
        reg: Dict[str, set[Tuple[int, int]]] = {n: set() for n in names}
        for c, n in assign.items():
            reg[n].add(c)

        target = int(math.ceil(len(cells) / max(1, len(names))))
        for _ in range(2000):
            largest = max(names, key=lambda n: len(reg[n]))
            smallest = min(names, key=lambda n: len(reg[n]))
            if len(reg[largest]) <= target and len(reg[smallest]) >= target - 1:
                break
            # pick a cell in largest that is close to smallest centroid
            if not reg[largest]:
                break
            sx, sy = centers[smallest]
            cand = min(reg[largest], key=lambda c: abs(c[0] - sx) + abs(c[1] - sy))
            reg[largest].remove(cand)
            reg[smallest].add(cand)

        self._regions = reg

    def choose_targets(self, robot_xy: Dict[str, np.ndarray], *, scene: LabScene, world: GridWorld2D, metrics=None):
        self._tick += 1
        blocked = set(world.obstacle_cells())
        visited = set(getattr(metrics, "visited_cells", set())) if metrics is not None else set()
        if metrics is not None and hasattr(metrics, "coverage_percent"):
            cov = float(metrics.coverage_percent())
            if cov <= self._last_cov + STAGNATION_EPS:
                self._stagnation += 1
            else:
                self._stagnation = 0
            self._last_cov = cov

        need_repartition = (not self._regions) or (self._stagnation >= STAGNATION_STEPS and (self._tick % 8 == 0))
        if need_repartition:
            self._compute_regions(robot_xy, scene, world)
            self._planners = {}
            for name in robot_xy.keys():
                if self.inner == "stc":
                    self._planners[name] = STC()
                else:
                    self._planners[name] = Boustrophedon()

        # For now we reuse inner planner but force targets to stay inside region
        out = {}
        for name, p in robot_xy.items():
            planner = self._planners[name]
            t = planner.choose_targets({name: p}, scene=scene, world=world, metrics=metrics)[name]
            cx, cy = _to_cell(scene, t)
            if (cx, cy) not in self._regions.get(name, set()):
                # pull target back to nearest region cell
                region = list(self._regions.get(name, set()))
                if region:
                    nearest = min(region, key=lambda c: abs(c[0] - cx) + abs(c[1] - cy))
                    t = _cell_center(scene, nearest[0], nearest[1])
                    cx, cy = nearest

            # Penalize short loops: avoid recently targeted cells inside region.
            recent = self._recent_targets.setdefault(name, [])
            region_set = self._regions.get(name, set())
            if (cx, cy) in recent:
                alt = _nearest_unvisited_cell(
                    scene=scene,
                    blocked=blocked,
                    visited=visited,
                    start=_to_cell(scene, p),
                    candidates=region_set,
                    forbidden=set(recent),
                )
                if alt is not None:
                    t = _cell_center(scene, alt[0], alt[1])
                    cx, cy = alt
            recent.append((cx, cy))
            if len(recent) > 12:
                del recent[:-12]
            out[name] = t
        return out

