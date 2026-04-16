"""
Локальная панель агента (Streamlit): задача, ссылка, фото, вывод.

Запуск из корня репозитория:
  python -m streamlit run panel/app.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UPLOAD_DIR = ROOT / "results" / "panel_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SESSION_LOG = ROOT / "results" / "agent_runs" / "sessions.jsonl"


def _load_recent_sessions(limit: int = 30) -> list[dict]:
    if not SESSION_LOG.exists():
        return []
    lines = SESSION_LOG.read_text(encoding="utf-8").strip().splitlines()
    rows: list[dict] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(rows))


def _fetch_url_snippet(url: str, max_chars: int = 2000) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; CoveragePanel/1.0)"})
    with urlopen(req, timeout=15) as resp:
        raw = resp.read(max_chars * 4)
    text = raw.decode("utf-8", errors="replace")
    # грубо убираем теги для читаемости в промпте
    text = re.sub(r"<script[^>]*>[\s\S]*?</script>", " ", text, flags=re.I)
    text = re.sub(r"<style[^>]*>[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _build_task(
    user_text: str,
    link: str,
    fetch_link: bool,
    image_note: str | None,
) -> str:
    parts = [user_text.strip()]
    if link.strip():
        parts.append(f"\n---\nСсылка: {link.strip()}")
        if fetch_link:
            try:
                snippet = _fetch_url_snippet(link.strip())
                parts.append(f"\nФрагмент страницы:\n{snippet}")
            except (HTTPError, URLError, TimeoutError, OSError) as e:
                parts.append(f"\n(не удалось скачать: {e})")
    if image_note:
        parts.append(f"\n---\n{image_note}")
    return "\n".join(parts).strip()


def main() -> None:
    st.set_page_config(page_title="Coverage Agent", layout="wide", initial_sidebar_state="expanded")
    st.title("Агент покрытия")
    st.caption("Локально: план → запуск симуляции → анализ. LM Studio должен быть запущен.")

    with st.sidebar:
        st.subheader("Подключение")
        base_url = st.text_input(
            "LM Studio API (OpenAI-compatible)",
            value=os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
            help="Обычно …/v1 в конце",
        )
        max_iter = st.number_input("Макс. итераций", min_value=1, max_value=10, value=2)
        task_mode = st.selectbox(
            "Режим задачи (префикс к промпту)",
            (
                "обычный",
                "только batch / симуляция",
                "только литература",
            ),
            help="Направляет агента без смены кода графа: уточняет приоритет инструментов.",
        )
        st.divider()
        st.markdown("**Подсказка:** текст модели не заменяет проверку JSON в `results/agent_runs/`.")
        with st.expander("История сессий (sessions.jsonl)"):
            for row in _load_recent_sessions(12):
                ts = row.get("ts_utc", "?")
                st.caption(ts[:19] if isinstance(ts, str) else ts)
                st.text((row.get("task_excerpt") or "")[:280])

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        task = st.text_area(
            "Задача для агента",
            height=160,
            placeholder="Например: Сравни voronoi и grid, seed 0, max_steps 400",
        )
        link = st.text_input("Ссылка (опционально)", placeholder="https://...")
        fetch_link = st.checkbox("Подтянуть текст со страницы (первые ~2000 символов)", value=False)

        img_note: str | None = None
        up = st.file_uploader("Фото / скрин (опционально)", type=["png", "jpg", "jpeg", "webp", "gif"])
        if up is not None:
            ext = Path(up.name).suffix or ".png"
            fname = f"{uuid.uuid4().hex}{ext}"
            dest = UPLOAD_DIR / fname
            dest.write_bytes(up.getvalue())
            st.image(up, caption=up.name, use_container_width=True)
            rel = str(dest.relative_to(ROOT)).replace("\\", "/")
            img_note = (
                f"Пользователь прикрепил изображение (файл в проекте): `{rel}`. "
                "Текущая связка с LM — текстовая: опиши в задаче, что на картинке, если это важно для эксперимента."
            )

    with col_right:
        st.subheader("Последний запуск")
        if "last_result" not in st.session_state:
            st.session_state.last_result = None

        if st.button("Запустить агента", type="primary", use_container_width=True):
            if not task.strip():
                st.error("Введите задачу.")
            else:
                prefixes = {
                    "обычный": "",
                    "только batch / симуляция": (
                        "[Режим: приоритет run_experiment и batch; обзор статей только по явной просьбе.]\n\n"
                    ),
                    "только литература": (
                        "[Режим: приоритет поиска статей arXiv/Semantic Scholar; симуляцию не запускать без явного запроса.]\n\n"
                    ),
                }
                full_task = prefixes[task_mode] + _build_task(task, link, fetch_link, img_note)
                os.environ["LMSTUDIO_BASE_URL"] = base_url.rstrip("/")
                with st.spinner("Идёт цикл plan → engineer → execute → analyze…"):
                    from agent.graph import run_agent_loop
                    from agent.session_log import append_agent_session_log

                    try:
                        st.session_state.last_result = run_agent_loop(
                            task=full_task,
                            max_iterations=int(max_iter),
                        )
                        try:
                            append_agent_session_log(ROOT, full_task, st.session_state.last_result)
                        except OSError:
                            pass
                    except Exception as e:
                        st.session_state.last_result = {"_error": str(e)}
                        st.error(f"Ошибка: {e}")

        res = st.session_state.last_result
        if res:
            if "_error" in res:
                st.warning(res["_error"])
            else:
                t1, t2, t3, t4 = st.tabs(["План", "Инструменты", "Анализ", "JSON"])
                with t1:
                    st.markdown(res.get("plan") or "_пусто_")
                with t2:
                    st.code(res.get("tool_output") or "_пусто_", language="json")
                with t3:
                    st.markdown(res.get("analysis") or "_пусто_")
                with t4:
                    show = {k: v for k, v in res.items() if not str(k).startswith("_")}
                    st.code(json.dumps(show, indent=2, ensure_ascii=False), language="json")

                st.caption(f"Итераций: {res.get('iteration', '?')}, завершено: {res.get('done', '?')}")


if __name__ == "__main__":
    main()
