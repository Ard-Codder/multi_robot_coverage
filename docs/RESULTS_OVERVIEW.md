# Results Overview

Ниже перечислены основные артефакты, которые стоит открыть в первую очередь.

## 1. Итоговые таблицы

- [`results/lab/presentation_report/report_cards.csv`](../results/lab/presentation_report/report_cards.csv)
- [`results/lab/presentation_report/report_cards.md`](../results/lab/presentation_report/report_cards.md)

Это основной компактный summary по static и dynamic сценам.

## 2. Панели для презентации

- [`results/lab/presentation_report/panel_median_static.png`](../results/lab/presentation_report/panel_median_static.png)
- [`results/lab/presentation_report/panel_median_dynamic.png`](../results/lab/presentation_report/panel_median_dynamic.png)
- [`results/lab/presentation_report/panel_seed0_static_dynamic.png`](../results/lab/presentation_report/panel_seed0_static_dynamic.png)

Если нужен быстрый визуальный обзор, обычно достаточно этих трех PNG.

## 3. GIF с динамической закраской покрытия

- [`results/lab/presentation_report/GIF_INDEX.md`](../results/lab/presentation_report/GIF_INDEX.md)

Через этот индекс можно открыть:
- classical dynamic GIF,
- RL-guided GIF,
- ML-guided hybrid GIF.

## 4. Основные директории с summary

- `results/lab/presentation_static/summary.csv`
- `results/lab/presentation_dynamic/summary.csv`
- `results/lab/presentation_dynamic_rl/summary.csv`
- `results/lab/presentation_dynamic_ml_rl/summary.csv`
- `results/lab/presentation_dynamic_plus_rl_ml/summary.csv`

## 5. Материалы под доклад

- [`docs/METHOD_GROUPS.md`](METHOD_GROUPS.md)
- [`docs/EXPERIMENTS_OVERVIEW.md`](EXPERIMENTS_OVERVIEW.md)
- [`docs/DIPLOMA_RESEARCH_PACKAGE.md`](DIPLOMA_RESEARCH_PACKAGE.md)
- [`docs/RESULTS_VALIDATION.md`](RESULTS_VALIDATION.md)
- private talk drafts intentionally stay outside the public-facing path

## 6. Материалы под диплом

Главный вход для текста диплома:

- [`DIPLOMA_RESEARCH_PACKAGE.md`](DIPLOMA_RESEARCH_PACKAGE.md) — единая исследовательская линия, прикладная классификация методов, экспериментальный протокол, VLA-позиция и каркас глав.

Для диплома результаты лучше интерпретировать через trade-off, а не через один
показатель coverage: время достижения порога, пройденная дистанция, blocked
moves, баланс нагрузки и pedestrian safety вместе показывают практическую
пригодность метода.

