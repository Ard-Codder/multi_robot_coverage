"""Схемы классификации (без заголовков — заголовок на слайде PPTX).

1) classification_layers.png — 5 слоёв
2) classification_algorithms.png — 3 семейства алгоритмов (метрики — в слое 5, не семейство)

Запуск: python scripts/build_classification_schematic.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "diploma_drafts" / "assets"
OUT_LAYERS = ASSETS / "classification_layers.png"
OUT_ALGOS = ASSETS / "classification_algorithms.png"

INK = "#222222"
LINE = "#CCCCCC"
BG = "#FFFFFF"
LABEL_FILL = "#F5F5F5"
LABEL_EDGE = "#BBBBBB"
TOKEN_FILL = "#FFFFFF"
TOKEN_EDGE = "#CCCCCC"

LAYERS: list[dict] = [
    {"name": "1. Покрытие", "tokens": [
        "baseline_random_walk", "baseline_grid", "baseline_voronoi",
        "baseline_frontier", "boustrophedon", "stc",
    ]},
    {"name": "2. Среда", "tokens": [
        "static_A_long", "dynamic_B_long", "large_complex_dynamic",
    ]},
    {"name": "3. Координация", "tokens": [
        "voronoi (зоны)", "darp_boustro", "darp_stc",
        "cbs_boustrophedon", "cbs_darp_boustro",
    ]},
    {"name": "4. Обучение", "tokens": [
        "ppo_policy", "ml_guided × 3", "ml_goal × 6",
    ]},
    {"name": "5. Проверка (метрики)", "tokens": [
        "покрытие %", "заблок. ходы", "наруш. у пешеходов",
        "мин. дистанция", "равномерность нагрузки",
    ]},
]

# Три семейства алгоритмов (не метрики)
FAMILIES: list[dict] = [
    {
        "name": "Классические",
        "tokens": [
            "baseline_random_walk", "baseline_grid", "baseline_voronoi",
            "baseline_frontier", "boustrophedon", "stc",
            "darp_boustro", "darp_stc",
        ],
    },
    {
        "name": "Координация",
        "tokens": [
            "cbs_boustrophedon", "cbs_darp_boustro",
            "CBSDeconflictWrapper",
        ],
    },
    {
        "name": "Обучаемые",
        "tokens": [
            "ppo_policy",
            "ml_guided (3 режима)",
            "ml_goal (6 режимов)",
        ],
    },
]


def _rbox(ax, x, y, w, h, fc=TOKEN_FILL, ec=TOKEN_EDGE, lw=0.8, radius=0.03):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.01,rounding_size={radius}",
            linewidth=lw, edgecolor=ec, facecolor=fc,
        )
    )


def _text(ax, x, y, s, *, size=10, color=INK, weight="normal", ha="left", va="center"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va)


def _token_row(ax, x0, y, x_max, tokens, *, token_h=0.34, gap=0.10, char_w=0.078, pad=0.18):
    x = x0
    y_t = y
    for token in tokens:
        w = max(0.9, len(token) * char_w + pad * 2)
        if x + w > x_max:
            x = x0
            y_t -= token_h + 0.05
        _rbox(ax, x, y_t - token_h / 2, w, token_h)
        _text(ax, x + w / 2, y_t, token, size=9.5, ha="center")
        x += w + gap


def build_layers() -> Path:
    fig = plt.figure(figsize=(13.5, 7.8), dpi=170)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 7.8)
    ax.set_axis_off()
    fig.patch.set_facecolor(BG)

    top, bottom = 7.45, 0.35
    row_h = (top - bottom) / len(LAYERS)
    for i, item in enumerate(LAYERS):
        y_top = top - i * row_h
        y_mid = y_top - row_h / 2
        if i > 0:
            ax.add_line(Line2D([0.5, 13.0], [y_top, y_top], color=LINE, lw=0.5))
        _rbox(ax, 0.5, y_mid - 0.30, 3.2, 0.60, fc=LABEL_FILL, ec=LABEL_EDGE)
        _text(ax, 0.65, y_mid, item["name"], size=12, weight="bold")
        _token_row(ax, 3.85, y_mid, 13.0, item["tokens"])

    fig.savefig(OUT_LAYERS, dpi=170, bbox_inches="tight", pad_inches=0.08, facecolor=BG)
    plt.close(fig)
    return OUT_LAYERS


def build_algos() -> Path:
    """Три семейства в один ряд."""
    fig = plt.figure(figsize=(13.5, 4.8), dpi=170)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 4.8)
    ax.set_axis_off()
    fig.patch.set_facecolor(BG)

    w = 4.15
    h = 4.0
    gap = 0.25
    for i, fam in enumerate(FAMILIES):
        x = 0.5 + i * (w + gap)
        y = 0.35
        _rbox(ax, x, y, w, h, fc=LABEL_FILL, ec=LABEL_EDGE, lw=0.9, radius=0.04)
        _text(ax, x + 0.25, y + h - 0.40, fam["name"], size=14, weight="bold")
        _token_row(ax, x + 0.25, y + h - 0.95, x + w - 0.25, fam["tokens"])

    fig.savefig(OUT_ALGOS, dpi=170, bbox_inches="tight", pad_inches=0.08, facecolor=BG)
    plt.close(fig)
    return OUT_ALGOS


if __name__ == "__main__":
    print(f"Saved: {build_layers()}")
    print(f"Saved: {build_algos()}")
