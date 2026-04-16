# Publication Policy

Этот файл нужен как практический чеклист перед публикацией репозитория в открытый Git.

## 1. Что является основным public research core

Публичное ядро проекта:

- `coverage_lab/`
- `experiments_lab/`
- `scripts/`
- `docs/`
- `results/lab/presentation_report/`
- curated summary CSV и демонстрационные GIF/PNG

Именно эти части отражают основной исследовательский вклад:
- постановка задачи,
- реализация методов,
- batch-эксперименты,
- итоговые сравнения,
- докладные артефакты.

## 2. Что считается legacy / optional

Следующие части репозитория полезны, но не являются основным public demo:

- `coverage_sim/` — legacy Isaac Sim oriented track
- `experiments/` — legacy batch/scripts track, связанный с `coverage_sim/`
- `agent/` — optional tools
- `panel/` — optional local UI
- `viz/` — вспомогательная визуализация

Их можно оставлять в репозитории, но в README они должны быть помечены как:
- `legacy`
- `optional`
- `not required for reproducing the main research results`

## 3. Что не стоит публиковать

### Точно не публиковать
- личные рабочие PDF и подборки литературы, например:
  - `Классические алгоритмы покрытия территории.pdf`
- локальные временные YAML/черновики:
  - `experiments_lab/_tmp_*.yaml`
- локальные служебные директории:
  - `.cursor/`
  - `agent-tools/`
- виртуальные окружения, кэши, IDE-файлы
- локальные логи и выгрузки агента/панели

### Публиковать только выборочно
- большие GIF и PNG
- сырые JSON всех прогонов
- тяжелые RL артефакты и чекпоинты

Подход:
- в Git оставлять только curated demo/results,
- все слишком тяжелое и вспомогательное — либо не публиковать, либо хранить отдельно.

## 4. Что желательно оставить в публичной версии

- `README.md`
- `.gitignore`
- `docs/METHOD_GROUPS.md`
- `docs/EXPERIMENTS_OVERVIEW.md`
- `docs/RESULTS_OVERVIEW.md`
- `results/lab/presentation_report/report_cards.md`
- `results/lab/presentation_report/GIF_INDEX.md`
- ключевые панели PNG
- ограниченный набор demo GIF

Материалы для личного выступления, speaker notes, thesis-checklists и рабочие обзоры
лучше держать вне публичной версии или исключать через `.gitignore`.

## 5. Практический чеклист перед переносом в Git

1. Убедиться, что в репозитории нет личных PDF и черновиков.
2. Проверить, что `.gitignore` покрывает локальные кэши, IDE-файлы и RL transient outputs.
3. Оставить только public-facing результаты, которые реально нужны читателю.
4. Проверить README: main track наверху, legacy/optional ниже.
5. Проверить, что ссылки на `report_cards`, панели и GIF-индекс актуальны.

