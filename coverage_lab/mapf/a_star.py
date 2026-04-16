from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple


Cell = Tuple[int, int]


@dataclass(frozen=True)
class Constraint:
    agent: str
    t: int
    kind: str  # "vertex" | "edge"
    cell: Cell | None = None
    edge: Tuple[Cell, Cell] | None = None


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(c: Cell) -> Iterable[Cell]:
    x, y = c
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)
    yield (x, y)  # wait


def violates(
    agent: str,
    t: int,
    prev: Cell,
    cur: Cell,
    constraints: List[Constraint],
) -> bool:
    for con in constraints:
        if con.agent != agent or con.t != t:
            continue
        if con.kind == "vertex" and con.cell == cur:
            return True
        if con.kind == "edge" and con.edge == (prev, cur):
            return True
    return False


def a_star_time_expanded(
    *,
    agent: str,
    start: Cell,
    goal: Cell,
    grid_size: Tuple[int, int],
    blocked: Set[Cell],
    constraints: List[Constraint],
    max_t: int,
) -> Optional[List[Cell]]:
    nx, ny = grid_size

    def in_bounds(c: Cell) -> bool:
        return 0 <= c[0] < nx and 0 <= c[1] < ny

    start_state = (start, 0)
    g: Dict[Tuple[Cell, int], int] = {start_state: 0}
    came: Dict[Tuple[Cell, int], Tuple[Cell, int]] = {}

    pq: List[Tuple[int, int, Cell, int]] = []
    heapq.heappush(pq, (manhattan(start, goal), 0, start, 0))

    while pq:
        f, gcost, cell, t = heapq.heappop(pq)
        if t > max_t:
            continue
        if cell == goal:
            # reconstruct
            st = (cell, t)
            path: List[Cell] = [cell]
            while st in came:
                st = came[st]
                path.append(st[0])
            path.reverse()
            return path

        for nb in neighbors(cell):
            if not in_bounds(nb) or nb in blocked:
                continue
            nt = t + 1
            if violates(agent, nt, cell, nb, constraints):
                continue
            key = (nb, nt)
            ng = gcost + 1
            if ng < g.get(key, 10**9):
                g[key] = ng
                came[key] = (cell, t)
                h = manhattan(nb, goal)
                heapq.heappush(pq, (ng + h, ng, nb, nt))

    return None

