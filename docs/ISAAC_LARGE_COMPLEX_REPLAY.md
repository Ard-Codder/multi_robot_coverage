# Isaac Sim 5: визуальный replay сцены CoverageLab

Скрипт [`experiments/isaac5_replay_coverage_lab.py`](../experiments/isaac5_replay_coverage_lab.py) проигрывает **уже сохранённый** JSON результата `coverage_lab` (траектории роботов и пешеходов, препятствия) в **NVIDIA Isaac Sim 5.x**. Это кинематическая марионетка: физика покрытия не пересчитывается, картинка совпадает с данными дипломного прогона.

## Предпосылки

- Установленный **Isaac Sim 5.x** и запуск через **`python.bat`** из каталога установки (не системный Python). Пример пути: `C:\Users\kirar\isaac-sim-standalone-5.0.0-windows-x86_64\python.bat` (в `run_isaac_replay_large_complex.bat` уже выставлен путь под эту установку; при другом каталоге поправьте `ISAAC_PYTHON`).
- Скрипт `python.bat` задаёт **`ISAAC_PATH`** и **`EXP_PATH`**; по умолчанию используется experience `apps\isaacsim.exp.base.python.kit` (через второй аргумент `SimulationApp`).
- В корне проекта доступен пакет `coverage_lab` (скрипт добавляет корень репозитория в `sys.path`).

## Соглашение по осям

- В **coverage_lab** координаты на плоскости: **x, y** (метры).
- В сцене Isaac: **X = x, Y = y, Z вверх**; пол по умолчанию **Z = 0**, роботы и пешеходы слегка приподняты по Z, чтобы не пересекаться с `GroundPlane`.

## Откуда взять JSON

После прогона, например:

```text
python experiments_lab/run_batch.py --config experiments_lab/batch_ideal_coverage_large_complex.yaml
```

в каталоге результатов (например `results/lab/ideal_coverage_demo/`) появятся файлы вида `*_seed*.json` с полями `robot_paths`, `pedestrian_paths`, `obstacles`, `bounds_xy`, `dt_sec` (см. [`coverage_lab/result_contract.md`](../coverage_lab/result_contract.md)).

## Запуск (Windows)

Обёртка из корня репозитория (при необходимости поправьте `ISAAC_PYTHON`).

В **cmd.exe**:

```text
run_isaac_replay_large_complex.bat results\lab\ideal_coverage_demo\YOUR_RUN.json
```

В **PowerShell** команды из текущей папки не ищутся по имени без префикса — используйте `.\`:

```text
.\run_isaac_replay_large_complex.bat results\lab\ideal_coverage_demo\YOUR_RUN.json
```

С опциональной геометрией из YAML (статические препятствия и границы из файла сцены; **траектории всё равно из JSON** — не смешивайте несогласованные прогоны):

```text
run_isaac_replay_large_complex.bat results\lab\ideal_coverage_demo\YOUR_RUN.json --geometry-yaml experiments_lab\scenes\large_complex_dynamic.yaml
```

Напрямую:

```text
C:\Users\kirar\isaac-sim-standalone-5.0.0-windows-x86_64\python.bat experiments\isaac5_replay_coverage_lab.py --json results\lab\ideal_coverage_demo\YOUR_RUN.json
```

### Полезные флаги

| Флаг | Назначение |
|------|------------|
| `--stride N` | Брать каждую N-ю точку пути (быстрее для длинных JSON). |
| `--max-steps M` | Обрезать длину пути по индексу. |
| `--headless` | Без окна; для стабильности включается `disable_viewport_updates`. |
| `--updates-per-frame K` | Сколько `SimulationApp.update()` на один шаг траектории. |
| `--experience ПУТЬ.kit` | Явный experience-файл вместо авто (`ISAAC_PATH\apps\isaacsim.exp.base.python.kit`). |
| `--compat-renderer` | Принудительно D3D12 (как `COVERAGE_LAB_ISAAC_COMPAT=1`). |
| `--compat-aggressive` | Дополнительно `compatibilityMode` (только если без него краш; часто чёрный RTX). |

### Переменные окружения

- **`ISAAC_EXTRA_ARGS`** — дополнительные аргументы Kit (строка в кавычках для Windows), например обход известных проблем рендера. Пример синтаксиса: `set ISAAC_EXTRA_ARGS=--/app/foo=bar` перед вызовом `python.bat`.

## Устранение неполадок (краш при старте)

Если в логе после строки `app ready` появляется **`Windows fatal exception: access violation`** и упоминаются **`rtx.scenedb.plugin.dll`** / **`carb.scenerenderer-rtx.plugin.dll`** (часто **RTX 5090 + Vulkan + Isaac Sim 5.0.0**):

### 1. Режим совместимости рендера (сначала попробуйте это)

По умолчанию достаточно **отключить Vulkan** (`--/app/vulkan=false` → на Windows обычно **D3D12**). Это устраняет типичный `access violation` в `rtx.scenedb` на RTX 50.

**Не включайте** `omni.kit.renderer.core/compatibilityMode`, если нет крайней необходимости: с ним часто в логе появляется `HydraEngine rtx failed creating scene renderer`, viewport остаётся **чёрным**, Stage может казаться «пустым» для RTX.

```text
...python.bat experiments\isaac5_replay_coverage_lab.py --compat-renderer --json ...
```

Переменная **`COVERAGE_LAB_ISAAC_COMPAT`** (как в `run_isaac_replay_large_complex.bat`): `0` — выкл., `1` — только D3D12, `2` или `aggressive` — D3D12 + compatibilityMode (крайний случай).

### Где смотреть сцену «в лайве»

Карта и роботы создаются **в том же окне**, которое открывает **`python.bat … isaac5_replay_coverage_lab.py`** (отдельный процесс Kit). Окно **Isaac Sim из ярлыка** на рабочем столе — **другой** USD Stage: туда скрипт ничего не загружает. Закройте лишние процессы **`kit.exe`**, если видите предупреждение про **kvdb lock**.

### 2. Остальное

1. Запуск из **обычного `cmd.exe` / PowerShell**, не из встроенного терминала IDE.
2. Обновить **драйвер NVIDIA** и при возможности **Isaac Sim** до более свежего патча под ваш GPU.
3. Закрыть другие тяжёлые 3D-приложения на том же GPU.
4. Для отладки визуала — без `--headless`.

### Про предупреждения в логе

- **`omni.isaac.core` deprecated → `isaacsim.core.api`** — это подгрузка **чужих** расширений Omniverse из experience `isaacsim.exp.base*`, не вызовы из `isaac5_replay_coverage_lab.py`. Их можно игнорировать для нашего replay.
- Сообщение **`HydraEngine rtx failed creating scene renderer`** чаще всего связано с **`compatibilityMode`** (`COVERAGE_LAB_ISAAC_COMPAT=2` или `--compat-aggressive`). Для нормальной картинки держите **`COVERAGE_LAB_ISAAC_COMPAT=1`** или только **`--compat-renderer`**.

## Ограничения

- Нет повторного решения задачи в PhysX: только **визуализация** записанных позиций.
- Ветка `coverage_sim` + `require_isaac=True` в старых live-скриптах **не** подключает реальный Omniverse (см. [`ISAAC_LIVE.md`](ISAAC_LIVE.md)); для финальной сцены ВКР используйте этот replay.

## См. также

- Сцена `large_complex_dynamic`: [`experiments_lab/scenes/large_complex_dynamic.yaml`](../experiments_lab/scenes/large_complex_dynamic.yaml)
- Парсинг данных без Isaac: [`coverage_lab/isaac_export/`](../coverage_lab/isaac_export/)
