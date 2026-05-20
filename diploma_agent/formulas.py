"""Formula helpers for thesis sections."""

from __future__ import annotations

FORMULA_SNIPPETS = {
    "coverage": r"""$$
coverage = |C| / |F|
$$

где \(C\) — множество покрытых свободных клеток, \(F\) — множество всех свободных клеток карты.""",
    "efficiency": r"""$$
efficiency = coverage / distance
$$

где \(distance\) — суммарная длина траекторий всех роботов.""",
    "load_balance": r"""$$
CV = sigma(d_1, d_2, ..., d_n) / mu(d_1, d_2, ..., d_n)
$$

где \(d_i\) — длина траектории i-го робота, \(\sigma\) — стандартное отклонение, \(\mu\) — среднее значение.""",
    "pedestrian_clearance": r"""$$
d_min = min_{r,p,t} ||x_r(t) - x_p(t)||
$$

где \(x_r(t)\) и \(x_p(t)\) — положения робота и пешехода в момент времени \(t\).""",
}


def recommended_formula_block() -> str:
    return "\n\n".join(FORMULA_SNIPPETS.values())
