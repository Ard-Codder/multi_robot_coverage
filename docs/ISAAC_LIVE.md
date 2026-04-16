# Запуск Isaac Sim (live-скрипты)

Цель: один воспроизводимый сценарий визуального прогона с `require_isaac=True`.

## Быстрый запуск из корня репозитория (Windows)

В проекте есть обёртки, которые вызывают **Isaac `python.bat`**, а не системный Python:

- `run_live_isaac.bat` → `experiments/run_random_walk_live.py`
- `run_live_zonal_isaac.bat` → `experiments/run_zonal_live.py`

При необходимости поправьте в `.bat` путь `ISAAC_PYTHON=` под вашу установку (сейчас `C:\isaacsim\python.bat`).

## Предпосылки

- Установленный **NVIDIA Isaac Sim** (версию укажите в ВКР).
- В сцене USD ожидаются примитивы/роботы с путями вроде `/World/Robots/robot_0` … `robot_N` (см. сообщения в `experiments/run_random_walk_live.py`).

## Рецепт запуска (Windows)

1. Откройте **Isaac Sim Python** (часто `python.bat` в каталоге установки, например `.../isaac-sim/python.bat`).
2. В той же сессии перейдите в корень проекта и установите зависимости при необходимости (`pip install -r requirements.txt`).
3. Запуск:

```text
"path\to\isaac-sim\python.bat" experiments/run_random_walk_live.py
```

или зональный сценарий:

```text
"path\to\isaac-sim\python.bat" experiments/run_zonal_live.py
```

4. Конфиг сцены и границ: `coverage_sim/configs/world.yaml` (`scene_name`, `bounds_xy`, `use_isaac_sim: true`). После ручного крафта сцены сохраните USD (см. `coverage_sim/assets/README.md`) и согласуйте `bounds_xy` и препятствия с фактической геометрией.

## Артефакты для отчёта

- JSON и графики в `results/` (имена задаются в скрипте live).
- Скриншоты окна симулятора и при необходимости экспорт метаданных: `python scripts/capture_run_metadata.py`.
