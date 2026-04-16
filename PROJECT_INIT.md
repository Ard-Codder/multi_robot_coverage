# Проект диплома

## Мультиагентное покрытие территории роботами в симуляции Isaac Sim

Автор: дипломный проект
Цель: исследование алгоритмов **мультиагентного покрытия территории** с визуализацией в **NVIDIA Isaac Sim**.

---

# 1. Цель проекта

Создать **простую экспериментальную платформу**, где несколько мобильных роботов:

* перемещаются по sidewalk сцене
* покрывают территорию
* используют разные алгоритмы распределения
* визуализируются в Isaac Sim

Основная задача:

**сравнить алгоритмы мультиагентного покрытия территории.**

---

# 2. Основные ограничения проекта

Для упрощения диплома:

НЕ используем

* ROS2
* сложные middleware
* Nav2
* SLAM

Используем:

* Python
* Isaac Sim API
* numpy
* простые контроллеры движения

Все алгоритмы работают **как Python скрипты**, которые управляют роботами в симуляции.

---

# 3. Роботы

Используем:

**3 робота**

Тип:

Nova Carter (или любой мобильный робот Isaac Sim)

Каждый робот:

* имеет позицию (x,y)
* получает целевую точку
* двигается к ней

Контроллер движения:

```
simple proportional controller
```

---

# 4. Сцена

Используем сцену:

```
Outdoor Sidewalk
```

или аналогичную городскую среду.

Цель:

роботы должны **покрыть всю доступную область**.

---

# 5. Алгоритмы для исследования

Нужно реализовать **несколько алгоритмов покрытия**.

## 1 Random Walk

Самый простой baseline.

Алгоритм:

робот случайно выбирает точку.

Плюсы:

* легко реализовать
* baseline

Минусы:

* медленно покрывает область

---

## 2 Grid Coverage

Простая сетка.

Алгоритм:

1 разбить территорию на клетки
2 роботы получают разные клетки
3 двигаются по ним

Плюсы:

* простой
* предсказуемый

---

## 3 Voronoi Coverage

Алгоритм:

1 роботы делят пространство на области
2 каждый покрывает свою

Используется:

```
Voronoi partitioning
```

Плюсы:

* более оптимально
* часто используется в исследованиях

---

## 4 Frontier Based Coverage

Роботы выбирают:

границу между исследованной и неисследованной областью.

Это приближает систему к **exploration алгоритмам**.

---

# 6. Архитектура проекта

Структура:

```
coverage_sim/

configs/
    robots.yaml
    world.yaml

env/
    isaac_env.py

robots/
    robot_controller.py
    robot_state.py

algorithms/
    random_walk.py
    grid_coverage.py
    voronoi_coverage.py
    frontier_coverage.py

simulation/
    spawn_robots.py
    simulation_loop.py

metrics/
    coverage_metrics.py

visualization/
    coverage_map.py

experiments/
    run_random_walk.py
    run_grid.py
    run_voronoi.py

tests/
    test_algorithms.py

README.md
requirements.txt
```

---

# 7. Основной цикл симуляции

Simulation loop:

```
while simulation_running:

    получить позиции роботов

    обновить карту покрытия

    вызвать алгоритм покрытия

    получить новые цели

    отправить команды роботам

    шаг симуляции
```

---

# 8. Метрики для диплома

Нужно измерять:

### Coverage %

процент покрытой территории

---

### Time to Coverage

время полного покрытия

---

### Distance travelled

сколько прошли роботы

---

### Efficiency

```
coverage / distance
```

---

# 9. Эксперименты

Для диплома провести эксперименты:

| алгоритм | роботы | покрытие | время |
| -------- | ------ | -------- | ----- |
| random   | 3      | ?        | ?     |
| grid     | 3      | ?        | ?     |
| voronoi  | 3      | ?        | ?     |
| frontier | 3      | ?        | ?     |

---

# 10. Интеграция с Isaac Sim

Python скрипт должен:

1 запускать Isaac Sim
2 загружать сцену
3 спавнить роботов
4 запускать алгоритм

Пример структуры:

```
isaac_env.py

class IsaacEnvironment:

    load_world()

    spawn_robot()

    get_robot_pose()

    set_robot_target()

    step_simulation()
```

---

# 11. Автоматические эксперименты

Cursor должен уметь запускать:

```
python experiments/run_random_walk.py
python experiments/run_grid.py
python experiments/run_voronoi.py
```

Каждый запуск:

* стартует симуляцию
* запускает алгоритм
* сохраняет метрики

---

# 12. Выходные данные

После эксперимента сохранить:

```
results/

random_walk.json
grid.json
voronoi.json
```

---

# 13. Визуализация

Построить графики:

```
coverage vs time
distance vs coverage
```

Использовать:

```
matplotlib
```

---

# 14. Основные научные статьи

### Multi-Robot Coverage

Choset H.

Coverage for robotics – A survey.

---

### Multi-Robot Coverage Path Planning

Galceran E., Carreras M.

A survey on coverage path planning for robotics.

---

### Voronoi Coverage

Cortes J.

Coverage control for mobile sensing networks.

---

### Frontier Exploration

Yamauchi B.

Frontier-based exploration using multiple robots.

---

# 15. Репозитории для анализа

MRTA:

https://github.com/robotarium/robotarium-python-simulator

Multi-robot exploration:

https://github.com/ethz-asl/multi_robot_exploration

Coverage planning:

https://github.com/atb033/multi_agent_path_planning

---

# 16. Задачи для Cursor

Cursor должен:

1 создать структуру проекта
2 написать простой Isaac environment
3 реализовать random walk
4 реализовать grid coverage
5 добавить метрики
6 сделать запуск экспериментов

---

# 17. Минимальный результат диплома

Минимально должно быть:

✔ 3 робота
✔ sidewalk сцена
✔ 3 алгоритма
✔ метрики
✔ графики
✔ демонстрация симуляции

---

# 18. Потенциал для статьи

В статье можно:

* сравнить алгоритмы
* показать масштабируемость
* показать влияние числа роботов
