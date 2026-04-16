# CoverageLab: контракт JSON результата

Каждый прогон сохраняется в один JSON. Этот формат используется:
- `experiments_lab/run_batch.py` для сводки `summary.csv`
- `coverage_lab/render/*` для PNG/GIF
- для сравнения классики/MAPF/RL/ML на одной постановке

## Минимальные поля

### Метаданные
- `algorithm`: строка (имя метода)
- `seed`: число
- `scene`: строка (имя сцены)
- `bounds_xy`: `[x_min, x_max, y_min, y_max]`
- `dt_sec`: число
- `steps`: число фактически выполненных шагов
- `max_steps`: максимум шагов
- `grid_resolution_m`: число

### Геометрия и динамика
- `obstacles`: список дисков `{x,y,r}`
- `robot_paths`: объект `{robot_i: [[x,y], ...]}`
- `pedestrians_enabled`: bool
- `pedestrian_paths`: объект `{ped_id: [[x,y], ...]}` (может быть пустым)

### Истории метрик
- `coverage_history`: список долей \([0..1]\)
- `distance_history`: список суммарной дистанции \(м\)

### Итоговые метрики (скаляры)
- `coverage_percent`: доля покрытия \([0..1]\)
- `time_to_coverage_sec`: число или `null`
- `distance_travelled_m`: число
- `efficiency`: coverage/distance
- `load_balance_cv`: коэффициент вариации длин путей по роботам

### Safety (для динамики)
- `min_pedestrian_clearance_m`: число или `null`
- `robot_robot_collisions`: число (по желанию)
- `robot_ped_violations`: число (по желанию)

