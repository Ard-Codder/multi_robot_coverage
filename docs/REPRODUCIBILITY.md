# Воспроизводимость экспериментов

## Версии окружения

- Зафиксируйте в тексте ВКР: версию Python (`python --version`), ОС, при использовании Isaac Sim — версию пакета и путь к `python.bat`.
- Хэш коммита Git: `git rev-parse HEAD` или поле `environment.git_head` в `batch_meta.json` после пакетного прогона.
- Зависимости: `requirements.txt` (приложение к работе или репозиторий).

## Снимок метаданных

Из корня репозитория:

```bash
python scripts/capture_run_metadata.py > results/run_metadata.json
```

В выводе: UTC-время, версия Python, платформа, путь к интерпретатору, `git_head` (если каталог — git-репозиторий).

## Матрица экспериментов

См. `experiments/MATRIX.md` и конфиги `experiments/batch_default.yaml`, `experiments/batch_quick.yaml`. Измерения: алгоритм × seed × `max_steps` × `target_coverage` (через `settings` в YAML).

## Пакетный прогон

`python experiments/run_batch.py` записывает для каждого прогона JSON, сводку `summary.csv` (включая `load_balance_cv`, `blocked_moves`, `use_isaac_sim`) и `batch_meta.json` с метаданными окружения.
