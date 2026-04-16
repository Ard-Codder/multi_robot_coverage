# Method Groups In This Repository

## 1. Classical Coverage Methods
Цель группы: покрытие территории за счет явного разбиения пространства и заранее определенной логики обхода.

Что есть в репозитории:
- `baseline_grid`
- `baseline_voronoi`
- `baseline_frontier`
- `boustrophedon`
- `stc`
- `darp_boustro`

Где смотреть:
- `coverage_lab/algorithms/baselines.py`
- `coverage_lab/algorithms/classic.py`

Что говорить в докладе:
- это основной и наиболее доведенный блок,
- именно он дал главный объем воспроизводимых результатов.

## 2. MAPF-Oriented Layer
Цель группы: уменьшить конфликты между роботами при сохранении основной стратегии покрытия.

Что есть в репозитории:
- `cbs_boustrophedon`
- `cbs_darp_boustro`
- `CBSDeconflictWrapper`

Где смотреть:
- `coverage_lab/algorithms/mapf_wrapped.py`
- `coverage_lab/mapf/cbs.py`

Что говорить в докладе:
- это не отдельный coverage-алгоритм, а слой де-конфликта поверх основной политики.

## 3. RL / Learning Through Interaction
Цель группы: научить политику действовать в среде через награду и взаимодействие.

Что есть в репозитории:
- Gym-среда `CoveragePedEnv`
- PPO-training pipeline
- `ppo_policy` для batch-инференса

Где смотреть:
- `coverage_lab/rl/env_gym.py`
- `experiments_lab/train_ppo.py`
- `coverage_lab/rl/ppo_policy.py`

Что говорить в докладе:
- RL в проекте доведен до рабочего докладного уровня,
- в текущем виде он используется как practical guided policy.

## 4. ML Planners
Цель группы: использовать обучаемую модель как направляющий планировщик.

Что есть в репозитории:
- `ml_guided`
- `SmallCNN`
- hybrid fallback для стабильного покрытия

Где смотреть:
- `coverage_lab/ml_planner/guided_algo.py`
- `coverage_lab/ml_planner/model.py`

Что говорить в докладе:
- чистый ML-guided вариант был слабым,
- для докладной версии используется `ml_guided hybrid`, то есть модель + guardrail/fallback.

## 5. VLA (Vision-Language-Action)
В рамках этого репозитория VLA не реализованы.

Что говорить в докладе:
- VLA рассматриваются как перспективное направление,
- в текущей работе это обзорный и стратегический блок, а не экспериментальный результат.

