"""Опциональные обёртки Gymnasium для обучения с подкреплением (сравнение с классикой в batch)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coverage_sim.rl.grid_coverage_env import GridCoverageGymEnv as GridCoverageGymEnv

__all__ = ["GridCoverageGymEnv"]


def __getattr__(name: str):
    if name == "GridCoverageGymEnv":
        from coverage_sim.rl.grid_coverage_env import GridCoverageGymEnv as G

        return G
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
