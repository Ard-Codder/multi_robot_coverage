## Карточки (медианы по seed)

- static: `results/lab/presentation_static/summary.csv`
- dynamic: `results/lab/presentation_dynamic_plus_rl_ml/summary.csv`

Файл CSV: `report_cards.csv`

| algorithm | static_cov_median | dynamic_cov_median | static_dist_median_m | dynamic_dist_median_m | dynamic_ped_viol_median |
|---|---:|---:|---:|---:|---:|
| `baseline_frontier` | 0.9003 | 0.8001 | 976.0 | 865.6 | 33 |
| `baseline_grid` | 0.7216 | 0.8001 | 2550.7 | 2573.4 | 128 |
| `baseline_voronoi` | 0.9003 | 0.8001 | 983.4 | 870.7 | 23 |
| `darp_boustro` | 0.6813 | 0.8001 | 2700.0 | 3960.0 | 141 |
| `ml_guided` |  | 0.8005 |  | 872.1 | 32 |
| `ppo_policy` |  | 0.8005 |  | 872.1 | 32 |
| `stc` | 0.9003 | 0.8001 | 1149.9 | 1324.1 | 57 |
