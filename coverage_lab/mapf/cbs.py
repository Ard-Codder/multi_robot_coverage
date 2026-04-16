from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from coverage_lab.mapf.a_star import Constraint, Cell, a_star_time_expanded


@dataclass
class Plan:
    paths: Dict[str, List[Cell]]
    cost: int


@dataclass
class CBSNode:
    constraints: List[Constraint] = field(default_factory=list)
    paths: Dict[str, List[Cell]] = field(default_factory=dict)
    cost: int = 0


def _path_at(path: List[Cell], t: int) -> Cell:
    if not path:
        raise ValueError("empty path")
    return path[min(t, len(path) - 1)]


def _first_conflict(paths: Dict[str, List[Cell]]) -> Optional[Tuple[str, str, int, str, Cell, Tuple[Cell, Cell]]]:
    agents = sorted(paths.keys())
    if not agents:
        return None
    horizon = max(len(p) for p in paths.values())
    for t in range(horizon):
        # vertex conflicts
        seen: Dict[Cell, str] = {}
        for a in agents:
            c = _path_at(paths[a], t)
            if c in seen and seen[c] != a:
                return (seen[c], a, t, "vertex", c, (c, c))
            seen[c] = a
        # edge conflicts
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                a0 = _path_at(paths[a], t)
                a1 = _path_at(paths[a], t + 1)
                b0 = _path_at(paths[b], t)
                b1 = _path_at(paths[b], t + 1)
                if a0 == b1 and a1 == b0:
                    return (a, b, t + 1, "edge", a1, (a0, a1))
    return None


def _compute_cost(paths: Dict[str, List[Cell]]) -> int:
    return sum(max(0, len(p) - 1) for p in paths.values())


def _replan_one(
    *,
    agent: str,
    start: Cell,
    goal: Cell,
    grid_size: Tuple[int, int],
    blocked: Set[Cell],
    constraints: List[Constraint],
    max_t: int,
) -> Optional[List[Cell]]:
    return a_star_time_expanded(
        agent=agent,
        start=start,
        goal=goal,
        grid_size=grid_size,
        blocked=blocked,
        constraints=constraints,
        max_t=max_t,
    )


def cbs_solve(
    *,
    starts: Dict[str, Cell],
    goals: Dict[str, Cell],
    grid_size: Tuple[int, int],
    blocked: Set[Cell],
    max_t: int = 256,
    max_nodes: int = 2500,
    max_sec: float = 2.0,
) -> Optional[Plan]:
    t0 = time.monotonic()
    root = CBSNode()
    for a in sorted(starts.keys()):
        p = _replan_one(
            agent=a,
            start=starts[a],
            goal=goals[a],
            grid_size=grid_size,
            blocked=blocked,
            constraints=root.constraints,
            max_t=max_t,
        )
        if p is None:
            return None
        root.paths[a] = p
    root.cost = _compute_cost(root.paths)

    pq: List[Tuple[int, int, CBSNode]] = []
    tie = 0
    heapq.heappush(pq, (root.cost, tie, root))

    while pq:
        if (time.monotonic() - t0) > float(max_sec):
            return None
        if tie >= int(max_nodes):
            return None
        _, _, node = heapq.heappop(pq)
        conflict = _first_conflict(node.paths)
        if conflict is None:
            return Plan(paths=node.paths, cost=node.cost)

        a1, a2, t, kind, cell, edge = conflict
        # branch: add constraint for a1 and for a2
        for agent in (a1, a2):
            child = CBSNode(constraints=list(node.constraints), paths=dict(node.paths))
            if kind == "vertex":
                child.constraints.append(Constraint(agent=agent, t=t, kind="vertex", cell=cell))
            else:
                # forbid edge traversal
                child.constraints.append(Constraint(agent=agent, t=t, kind="edge", edge=edge))

            new_path = _replan_one(
                agent=agent,
                start=starts[agent],
                goal=goals[agent],
                grid_size=grid_size,
                blocked=blocked,
                constraints=child.constraints,
                max_t=max_t,
            )
            if new_path is None:
                continue
            child.paths[agent] = new_path
            child.cost = _compute_cost(child.paths)
            tie += 1
            heapq.heappush(pq, (child.cost, tie, child))

    return None

