from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _median(rows: List[Dict[str, str]], key: str) -> float | None:
    vals: List[float] = []
    for r in rows:
        v = r.get(key)
        if v is None or v == "":
            continue
        try:
            vals.append(float(v))
        except Exception:
            continue
    return None if not vals else float(statistics.median(vals))


def _group_by(rows: List[Dict[str, str]], field: str) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        out.setdefault(str(r.get(field, "")), []).append(r)
    return out


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def render_2x2_panel(
    *,
    out_png: Path,
    title: str,
    metrics: List[Tuple[str, str, str]],
    algo_stats: Dict[str, Dict[str, float | None]],
    algos: List[str],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.0))
    axes = axes.reshape(2, 2)
    fig.suptitle(title)

    for ax, (mkey, label, fmt) in zip(axes.flat, metrics):
        vals = []
        for a in algos:
            v = algo_stats.get(a, {}).get(mkey)
            vals.append(0.0 if v is None else float(v))
        ax.bar(range(len(algos)), vals)
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algos, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(label)
        # annotate bars
        for i, v in enumerate(vals):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=8, rotation=0)
        ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", type=Path, default=Path("results/lab/presentation_static/summary.csv"))
    ap.add_argument("--dynamic", type=Path, default=Path("results/lab/presentation_dynamic/summary.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/lab/presentation_report"))
    ap.add_argument("--seed", type=int, default=0, help="Seed for the seed0 panel")
    args = ap.parse_args()

    static_csv = (ROOT / args.static).resolve()
    dynamic_csv = (ROOT / args.dynamic).resolve()
    out_dir = (ROOT / args.out_dir).resolve()
    _ensure_dir(out_dir)

    srows = _load_csv(static_csv)
    drows = _load_csv(dynamic_csv)

    # build cards (median across seeds)
    s_by_algo = _group_by(srows, "algorithm")
    d_by_algo = _group_by(drows, "algorithm")
    algos = sorted(set(s_by_algo) | set(d_by_algo))

    cards: List[Dict[str, str]] = []
    algo_stats: Dict[str, Dict[str, float | None]] = {}
    for a in algos:
        sr = s_by_algo.get(a, [])
        dr = d_by_algo.get(a, [])
        st_cov = _median(sr, "coverage_percent")
        dy_cov = _median(dr, "coverage_percent")
        st_dist = _median(sr, "distance_travelled_m")
        dy_dist = _median(dr, "distance_travelled_m")
        st_blk = _median(sr, "blocked_moves")
        dy_blk = _median(dr, "blocked_moves")
        dy_viol = _median(dr, "robot_ped_violations")
        algo_stats[a] = {
            "st_cov": st_cov,
            "dy_cov": dy_cov,
            "st_dist": st_dist,
            "dy_dist": dy_dist,
            "st_blk": st_blk,
            "dy_blk": dy_blk,
            "dy_viol": dy_viol,
        }
        cards.append(
            {
                "algorithm": a,
                "static_cov_median": "" if st_cov is None else f"{st_cov:.4f}",
                "dynamic_cov_median": "" if dy_cov is None else f"{dy_cov:.4f}",
                "static_dist_median_m": "" if st_dist is None else f"{st_dist:.1f}",
                "dynamic_dist_median_m": "" if dy_dist is None else f"{dy_dist:.1f}",
                "static_blocked_median": "" if st_blk is None else f"{st_blk:.0f}",
                "dynamic_blocked_median": "" if dy_blk is None else f"{dy_blk:.0f}",
                "dynamic_ped_viol_median": "" if dy_viol is None else f"{dy_viol:.0f}",
            }
        )

    cards_csv = out_dir / "report_cards.csv"
    with cards_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cards[0].keys()))
        w.writeheader()
        w.writerows(cards)

    cards_md = out_dir / "report_cards.md"
    lines = [
        "## Карточки (медианы по seed)",
        "",
        f"- static: `{args.static.as_posix()}`",
        f"- dynamic: `{args.dynamic.as_posix()}`",
        "",
        "Файл CSV: `report_cards.csv`",
        "",
        "| algorithm | static_cov_median | dynamic_cov_median | static_dist_median_m | dynamic_dist_median_m | dynamic_ped_viol_median |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in cards:
        lines.append(
            f"| `{c['algorithm']}` | {c['static_cov_median']} | {c['dynamic_cov_median']} | {c['static_dist_median_m']} | {c['dynamic_dist_median_m']} | {c['dynamic_ped_viol_median']} |"
        )
    cards_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 2x2 panels (median)
    render_2x2_panel(
        out_png=out_dir / "panel_median_static.png",
        title="Static (median over seeds): coverage/dist/blocked",
        metrics=[
            ("st_cov", "coverage (median)", "{:.2f}"),
            ("st_dist", "distance (m, median)", "{:.0f}"),
            ("st_blk", "blocked_moves (median)", "{:.0f}"),
            ("dy_viol", "dynamic ped violations (median)", "{:.0f}"),
        ],
        algo_stats=algo_stats,
        algos=algos,
    )

    render_2x2_panel(
        out_png=out_dir / "panel_median_dynamic.png",
        title="Dynamic (median over seeds): coverage/dist/blocked/violations",
        metrics=[
            ("dy_cov", "coverage (median)", "{:.2f}"),
            ("dy_dist", "distance (m, median)", "{:.0f}"),
            ("dy_blk", "blocked_moves (median)", "{:.0f}"),
            ("dy_viol", "ped violations (median)", "{:.0f}"),
        ],
        algo_stats=algo_stats,
        algos=algos,
    )

    # seed-specific panel for quick slide (static+dynamic coverage, distance)
    seed = int(args.seed)
    s_seed = [r for r in srows if int(r.get("seed") or 0) == seed]
    d_seed = [r for r in drows if int(r.get("seed") or 0) == seed]
    ss = {r["algorithm"]: r for r in s_seed}
    dd = {r["algorithm"]: r for r in d_seed}
    algo_stats_seed: Dict[str, Dict[str, float | None]] = {}
    for a in algos:
        sr = ss.get(a, {})
        dr = dd.get(a, {})
        def f(row, k):
            try:
                v = row.get(k)
                return None if v in (None, "") else float(v)
            except Exception:
                return None
        algo_stats_seed[a] = {
            "st_cov": f(sr, "coverage_percent"),
            "dy_cov": f(dr, "coverage_percent"),
            "st_dist": f(sr, "distance_travelled_m"),
            "dy_dist": f(dr, "distance_travelled_m"),
        }

    render_2x2_panel(
        out_png=out_dir / f"panel_seed{seed}_static_dynamic.png",
        title=f"Seed {seed}: static vs dynamic (coverage+distance)",
        metrics=[
            ("st_cov", "static coverage", "{:.2f}"),
            ("dy_cov", "dynamic coverage", "{:.2f}"),
            ("st_dist", "static distance (m)", "{:.0f}"),
            ("dy_dist", "dynamic distance (m)", "{:.0f}"),
        ],
        algo_stats=algo_stats_seed,
        algos=algos,
    )

    print(f"OK: wrote {out_dir}")


if __name__ == "__main__":
    main()

