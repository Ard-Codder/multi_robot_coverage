from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

METRICS = [
    ("coverage_percent", "coverage", "{:.3f}"),
    ("time_to_coverage_sec", "time_s", "{:.1f}"),
    ("distance_travelled_m", "distance_m", "{:.0f}"),
    ("blocked_moves", "blocked", "{:.0f}"),
    ("robot_ped_violations", "ped_viol", "{:.0f}"),
    ("ml_model_step_rate", "model_rate", "{:.2f}"),
    ("ml_fallback_rate", "fallback_rate", "{:.2f}"),
    ("ml_unsafe_model_targets", "unsafe_targets", "{:.0f}"),
]


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _vals(rows: list[dict[str, str]], key: str) -> list[float]:
    vals: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            vals.append(float(value))
        except ValueError:
            pass
    return vals


def _median(rows: list[dict[str, str]], key: str) -> float | None:
    vals = _vals(rows, key)
    return None if not vals else float(statistics.median(vals))


def _fmt(value: float | None, fmt: str) -> str:
    return "" if value is None else fmt.format(float(value))


def _group(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("algorithm", "")), []).append(row)
    return grouped


def build_diagnosis(summary_csv: Path, out_dir: Path, *, sync_docs: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load(summary_csv)
    grouped = _group(rows)
    algos = sorted(grouped)

    lines = [
        "# Диагностика ML-guided",
        "",
        "Цель проверки — отделить вклад нейросетевой части от fallback-логики. "
        "Для этого сравниваются `ml_guided_pure`, `ml_guided_soft_guarded`, `ml_guided_guarded` и `baseline_voronoi`.",
        "",
        "## Медианные показатели",
        "",
        "| algorithm | coverage | time_s | distance_m | blocked | ped_viol | model_rate | fallback_rate | unsafe_targets |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    med: dict[str, dict[str, float | None]] = {}
    for algo in algos:
        med[algo] = {key: _median(grouped[algo], key) for key, _, _ in METRICS}
        lines.append(
            f"| `{algo}` | "
            f"{_fmt(med[algo]['coverage_percent'], '{:.3f}')} | "
            f"{_fmt(med[algo]['time_to_coverage_sec'], '{:.1f}')} | "
            f"{_fmt(med[algo]['distance_travelled_m'], '{:.0f}')} | "
            f"{_fmt(med[algo]['blocked_moves'], '{:.0f}')} | "
            f"{_fmt(med[algo]['robot_ped_violations'], '{:.0f}')} | "
            f"{_fmt(med[algo]['ml_model_step_rate'], '{:.2f}')} | "
            f"{_fmt(med[algo]['ml_fallback_rate'], '{:.2f}')} | "
            f"{_fmt(med[algo]['ml_unsafe_model_targets'], '{:.0f}')} |"
        )

    pure = med.get("ml_guided_pure")
    soft = med.get("ml_guided_soft_guarded")
    guarded = med.get("ml_guided_guarded")
    base = med.get("baseline_voronoi")

    lines.extend(["", "## Вывод", ""])
    if pure and soft and guarded and base:
        pure_cov = pure.get("coverage_percent") or 0.0
        soft_cov = soft.get("coverage_percent") or 0.0
        guarded_cov = guarded.get("coverage_percent") or 0.0
        base_cov = base.get("coverage_percent") or 0.0
        lines.append(
            f"`ml_guided_pure` показывает вклад самой модели без страховки: медианное покрытие {pure_cov:.3f}. "
            f"`ml_guided_soft_guarded` показывает, сколько качества возвращает fallback при небезопасных или стагнирующих шагах: "
            f"медианное покрытие {soft_cov:.3f}. `ml_guided_guarded` воспроизводит прежнюю схему, где до целевого покрытия "
            f"работает baseline-guardrail, поэтому его результат ожидаемо близок к `baseline_voronoi` ({guarded_cov:.3f} против {base_cov:.3f})."
        )
        if pure_cov + 1e-9 < base_cov:
            lines.append(
                "Следовательно, текущая обученная модель сама по себе пока не превосходит сильный Voronoi-baseline. "
                "Главный результат диагностики — не провал ML-направления, а обнаружение того, что прежний финальный отчёт измерял "
                "в основном guardrail, а не самостоятельную нейросетевую политику."
            )
        else:
            lines.append(
                "В этой конфигурации pure-режим не хуже baseline по покрытию; дальнейший анализ должен смотреть на дистанцию, безопасность и устойчивость."
            )
    else:
        lines.append("Недостаточно строк ablation-summary для полного вывода.")

    lines.extend(
        [
            "",
            "## Что это значит для диплома",
            "",
            "Корректная формулировка: реализованная финальная система является hybrid-решением с обучаемым компонентом и проверяемым guardrail. "
            "Ablation показывает реальный вклад каждого слоя: pure-режим оценивает ML-модель, soft-режим — практическую страховку, "
            "guarded-режим — устойчивый демонстрационный baseline. Для дальнейшего усиления ML требуется переход от копирования действий учителей "
            "к обучению по целевой функции покрытия, дистанции и безопасности.",
            "",
        ]
    )

    out_path = out_dir / "ML_GUIDED_DIAGNOSIS.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    if sync_docs:
        docs_path = ROOT / "docs" / "ML_GUIDED_DIAGNOSIS.md"
        docs_path.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("results/lab/ml_guided_ablation/summary.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/lab/ml_guided_ablation"))
    parser.add_argument("--no-sync-docs", action="store_true")
    args = parser.parse_args()
    out = build_diagnosis((ROOT / args.summary).resolve(), (ROOT / args.out_dir).resolve(), sync_docs=not args.no_sync_docs)
    print(f"OK: ML diagnosis -> {out}")


if __name__ == "__main__":
    main()
