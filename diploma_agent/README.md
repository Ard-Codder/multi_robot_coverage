# Local Thesis Agent

Локальная панель для итеративного написания диплома через LM Studio.

## Запуск

1. Запусти LM Studio Server на `http://127.0.0.1:1234/v1`.
2. При необходимости укажи модель:

```powershell
$env:DIPLOMA_MODEL="google/gemma-4-26b-a4b"
$env:DIPLOMA_EMBEDDING_MODEL="text-embedding-nomic-embed-text-v1.5"
```

3. Открой панель:

```powershell
python -m streamlit run diploma_agent/app.py
```

Runtime лежит в `thesis_workspace/` и исключен из git.

Основной рабочий экспорт для ручной правки:

```powershell
python -m diploma_agent.cli /docx
```

Файл появится в `thesis_workspace/build/diploma.docx`.

## Команды

- `/plan` — обновить план диплома.
- `/write 2.1` — написать раздел.
- `/rewrite 2.1 комментарий` — переписать раздел.
- `/review 2.1` — отредактировать раздел.
- `/sources 2.1` — показать источники раздела.
- `/quality 2.1` — локально оценить стиль, повторы, обрывы и сложность текста.
- `/bibliography` — обновить ГОСТ-подобный список использованных источников.
- `/index_code` — построить embedding-индекс summaries кода через LM Studio.
- `/render` или `/pdf` — собрать Typst/PDF.
- `/docx` или `/export_docx` — собрать Word-документ для ручной правки.

Во вкладке `Полный прогон` можно запустить генерацию по всему плану с прогрессом,
preview текста, локальными отчетами качества, библиографией и экспортом DOCX/PDF.
Внешние проверки уникальности подключаются только через официальные API или через
ручной импорт ссылок/метрик отчета.
