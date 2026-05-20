# Results Validation Status

Дата проверки: 2026-04-28.

## Проверенные артефакты

Проверены ключевые CSV и public-facing media artifacts, которые используются в
README, `docs/RESULTS_OVERVIEW.md` и дипломной линии.

| Артефакт | Статус | Содержание |
|---|---:|---|
| `results/lab/presentation_static/summary.csv` | OK | 25 строк, 5 алгоритмов, seeds 0..4 |
| `results/lab/presentation_dynamic/summary.csv` | OK | 25 строк, 5 алгоритмов, seeds 0..4 |
| `results/lab/presentation_dynamic_ml_rl/summary.csv` | OK | 12 строк, 4 алгоритма, seeds 0..2 |
| `results/lab/presentation_dynamic_plus_rl_ml/summary.csv` | OK | 31 строк, 7 алгоритмов |
| `results/lab/presentation_report/report_cards.csv` | OK | 7 агрегированных строк |
| `results/lab/presentation_report/panel_median_static.png` | OK | public panel exists |
| `results/lab/presentation_report/panel_median_dynamic.png` | OK | public panel exists |
| `results/lab/presentation_report/panel_seed0_static_dynamic.png` | OK | public panel exists |
| `results/lab/presentation_dynamic/*.gif` и др. | опционально | GIF не обязаны лежать в git: см. `GIF_INDEX.md`; создать `python -m experiments_lab.run_batch ... --render` |

Панели PNG в `presentation_report/` пересобираются скриптом и должны быть в репозитории.
GIF тяжёлые — в чистом checkout их может не быть до локального прогона с `--render`.

## Проверка воспроизводимости

Пересборка public report (основная пара CSV: static + dynamic):

```bash
python scripts/build_presentation_report.py --seed 0
```

Вариант с расширенной dynamic-таблицей (RL/ML строки):

```bash
python scripts/build_presentation_report.py --dynamic results/lab/presentation_dynamic_plus_rl_ml/summary.csv --seed 0
```

Перед сдачей имеет смысл: `.venv`, `pip install -r requirements.txt`, команды из
`docs/EXPERIMENTS_OVERVIEW.md`, при необходимости GIF — `run_batch` с флагом `--render`.

## Что можно уверенно использовать в дипломе

- Multi-seed summary по static и dynamic сценам.
- Агрегированные report cards по медианам.
- Панели PNG для презентации и текста.
- GIF для демонстрации траекторий и динамической закраски покрытия.
- Сравнение по coverage, distance, time, blocked moves, load balance и pedestrian safety.

## Статус исходного кода CoverageLab

Ранее в репозитории отсутствовал модуль `coverage_lab/env/grid_world.py`, из-за чего
не работали импорты `coverage_lab.sim`, `experiments_lab.run_batch` и обучение в
`coverage_lab/rl`. Модуль восстановлен:

- `coverage_lab/env/grid_world.py` — `GridWorld2D`, `WorldState`, пешеходы, шаг
  симуляции, `obstacle_cells`, `sample_spawn_points`;
- `coverage_lab/env/__init__.py` — реэкспорт для удобства.

Локальная проверка после установки Python:

```bash
python -c "from coverage_lab.env.grid_world import GridWorld2D, WorldState; print('ok')"
python -m pytest tests -q
```

Короткая проверка пайплайна batch (без GIF):

```bash
python -m experiments_lab.run_batch --config experiments_lab/batch_plan_verify_smoke.yaml
python -m experiments_lab.run_batch --config experiments_lab/batch_plan_verify_dynamic_smoke.yaml
```

