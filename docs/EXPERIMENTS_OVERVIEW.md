# Experiments Overview

Этот файл нужен как краткая карта экспериментов и результатов для внешнего читателя.

## Основные директории

### Static / Dynamic core benchmarks
- `results/lab/presentation_static/summary.csv`
- `results/lab/presentation_dynamic/summary.csv`

### Dynamic with RL comparison
- `results/lab/presentation_dynamic_rl/summary.csv`

### Dynamic with RL + ML comparison
- `results/lab/presentation_dynamic_ml_rl/summary.csv`
- агрегированная сводка: `results/lab/presentation_dynamic_plus_rl_ml/summary.csv`

### Final public-facing artifacts
- `results/lab/presentation_report/report_cards.csv`
- `results/lab/presentation_report/report_cards.md`
- `results/lab/presentation_report/panel_median_static.png`
- `results/lab/presentation_report/panel_median_dynamic.png`
- `results/lab/presentation_report/panel_seed0_static_dynamic.png`
- `results/lab/presentation_report/GIF_INDEX.md`

## Ключевые batch-конфиги

### Основные сравнения
- `experiments_lab/batch_presentation_static.yaml`
- `experiments_lab/batch_presentation_dynamic.yaml`

### RL comparison
- `experiments_lab/batch_presentation_dynamic_rl.yaml`

### RL + ML comparison
- `experiments_lab/batch_presentation_dynamic_ml_rl.yaml`

## Legacy / Isaac-oriented branch

Ниже находится более ранняя и дополнительная ветка проекта:

- `coverage_sim/`
- `experiments/`

Эти папки сохраняются в репозитории, но для внешнего читателя их стоит считать:
- legacy,
- optional,
- не обязательными для воспроизведения основных public research results.

## Как воспроизвести основные public-results

### 1. Static comparison
```bash
python -m experiments_lab.run_batch --config experiments_lab/batch_presentation_static.yaml
```

### 2. Dynamic comparison
```bash
python -m experiments_lab.run_batch --config experiments_lab/batch_presentation_dynamic.yaml
```

### 3. Dynamic RL comparison
```bash
python experiments_lab/train_ppo.py --mode smoke --scene experiments_lab/scenes/dynamic_B_long.yaml
python -m experiments_lab.run_batch --config experiments_lab/batch_presentation_dynamic_rl.yaml
```

### 4. Dynamic ML + RL comparison
```bash
python -m experiments_lab.run_batch --config experiments_lab/batch_presentation_dynamic_ml_rl.yaml
```

### 5. Пересобрать публичный отчет
```bash
python scripts/build_presentation_report.py --static results/lab/presentation_static/summary.csv --dynamic results/lab/presentation_dynamic_plus_rl_ml/summary.csv --seed 0
```

## Что смотреть в первую очередь

1. `report_cards.md` — компактное сравнение алгоритмов.
2. `panel_median_dynamic.png` — итог по динамической сцене.
3. `GIF_INDEX.md` — какие GIF открывать для демонстрации.
4. `docs/DIPLOMA_RESEARCH_PACKAGE.md` — как связать результаты с дипломной классификацией, VLA-блоком и текстом глав.
5. `docs/DIPLOMA_EXPERIMENTS_AND_METRICS.md` — как описывать метрики и экспериментальный раздел.

## Как использовать результаты в дипломе

Для static-сцены основной вопрос: какие методы быстрее и стабильнее достигают
высокого покрытия при разумной длине траекторий. Для dynamic-сцены одного
coverage недостаточно: обязательно обсуждать `min_pedestrian_clearance_m`,
`robot_ped_violations`, `blocked_moves` и `load_balance_cv`.

Рекомендуемый порядок анализа:

1. Сначала показать медианы по seed из `presentation_report/report_cards.md`.
2. Затем разобрать static и dynamic отдельно, потому что у них разные target coverage и ограничения.
3. После этого показать learning-based блок (`ml_guided`, `ppo_policy`) как practical/hybrid comparison, а не как окончательный SOTA-вывод.
4. MAPF/CBS описывать как слой координации и де-конфликта поверх coverage planner, а не как самостоятельный алгоритм покрытия.

