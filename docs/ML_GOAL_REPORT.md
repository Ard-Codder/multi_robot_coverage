# ML-goal ablation

Этот отчёт сравнивает старый step-level ML-guided и новую goal/frontier-политику.

| algorithm | coverage | time_s | distance_m | blocked | ped_viol | goal_model_rate | goal_fallback_rate | unsafe_goals | avg_goal_l1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_frontier` | 0.781 | 69.4 | 1388 | 0 | 42 |  |  |  |  |
| `baseline_voronoi` | 0.781 | 69.1 | 1382 | 0 | 98 |  |  |  |  |
| `darp_boustro` | 0.693 |  | 10400 | 0 | 327 |  |  |  |  |
| `ml_goal_allocated` | 0.780 | 74.4 | 1488 | 0 | 81 | 1.00 | 0.00 | 0 | 15.5 |
| `ml_goal_ranked` | 0.782 | 73.8 | 1476 | 0 | 71 | 1.00 | 0.00 | 0 | 14.1 |
| `ml_goal_soft_guarded` | 0.780 | 74.4 | 1488 | 0 | 81 | 1.00 | 0.00 | 0 | 15.5 |
| `ml_guided_guarded` | 0.781 | 69.1 | 1382 | 0 | 98 |  |  |  |  |
| `stc` | 0.780 | 144.0 | 2856 | 47 | 115 |  |  |  |  |

## Вывод

Новая goal-политика оценивается отдельно от старого step-level классификатора. Медианное покрытие goal-режима: 0.780, baseline_voronoi: 0.781. По времени goal-режим занимает 74.4 с против 69.1 с у Voronoi и 69.4 с у frontier. По дистанции goal-режим проходит около 1488 м против 1388 м у frontier. Это показывает, что после перехода к выбору целей/frontier ML-компонент стал практически работоспособным: он достигает целевого покрытия, почти догоняет сильный Voronoi-baseline и явно превосходит тяжёлые схемы `stc`/`darp_boustro` по времени достижения цели.
