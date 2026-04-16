## GIF с динамической закраской покрытия

Эти GIF сгенерированы рендерером `coverage_lab/render/animate.py` с `show_dynamic_coverage=True`.
То есть **закраска появляется по мере прохождения роботов**, а не «залита сразу».

### Dynamic (scene `dynamic_B_long`, seed0)

- `results/lab/presentation_dynamic/dynamic_B_long__baseline_grid__seed0.gif`
- `results/lab/presentation_dynamic/dynamic_B_long__baseline_voronoi__seed0.gif`
- `results/lab/presentation_dynamic/dynamic_B_long__baseline_frontier__seed0.gif`
- `results/lab/presentation_dynamic/dynamic_B_long__darp_boustro__seed0.gif`
- `results/lab/presentation_dynamic/dynamic_B_long__stc__seed0.gif`

### RL-guided (scene `dynamic_B_long`, seed0)

- `results/lab/presentation_dynamic_rl/dynamic_B_long__ppo_policy__seed0.gif`

### ML-guided hybrid (scene `dynamic_B_long`, seed0)

- `results/lab/presentation_dynamic_ml_rl/dynamic_B_long__ml_guided__seed0.gif`

### Где лежат рядом PNG

Для каждого прогона рядом лежат PNG:

- `*_coverage_vs_time.png`
- `*_distance_vs_coverage.png`
- `*_coverage_2d.png`

