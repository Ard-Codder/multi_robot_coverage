# Матрица экспериментов (шаблон для ВКР)

Заполняйте таблицу под фактические прогоны; источник данных — `summary.csv` после `run_batch.py`.

| Алгоритм   | Seed | max_steps | target_coverage | coverage_% | time_to_coverage_s | load_balance_cv | blocked_moves | use_isaac_sim |
| ---------- | ---- | --------- | ----------------- | ---------- | ------------------ | --------------- | ------------- | ------------- |
| random_walk| 0    | 1200      | 0.95              |            |                    |                 |               | false         |
| grid       | 0    | 1200      | 0.95              |            |                    |                 |               | false         |
| …          |      |           |                   |            |                    |                 |               |               |

Примечания:

- Для сравнения алгоритмов держите одинаковые `max_steps`, `target_coverage`, число роботов (`coverage_sim/configs/robots.yaml`) и границы мира (`world.yaml`).
- Несколько seed снижают дисперсию случайных методов (`random_walk`, `frontier`, `zonal`).
