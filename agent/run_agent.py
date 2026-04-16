"""
Запуск исследовательского цикла (LM Studio / OpenAI-compatible).

Пример (PowerShell):
  $env:LMSTUDIO_BASE_URL="http://127.0.0.1:1234/v1"
  python agent/run_agent.py --task "Сравни voronoi и frontier на seed 0, max_steps 500"

Удалённый сервер:
  $env:LMSTUDIO_BASE_URL="http://26.148.188.237:1234/v1"
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.graph import run_agent_loop  # noqa: E402
from agent.session_log import append_agent_session_log  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="LangGraph агент + LM Studio")
    p.add_argument(
        "--task",
        type=str,
        default="Запусти короткое сравнение random_walk и grid (один seed), оцени coverage и load_balance_cv.",
        help="Формулировка задачи для агента",
    )
    p.add_argument("--max-iter", type=int, default=3, dest="max_iter", help="Макс. итераций engineer→execute→analyze")
    p.add_argument("--json", action="store_true", help="Печать финального состояния JSON в stdout")
    args = p.parse_args()

    result = run_agent_loop(task=args.task, max_iterations=args.max_iter)
    try:
        append_agent_session_log(ROOT, args.task, result)
    except OSError:
        pass
    show_keys = ("task", "plan", "engineer_output", "tool_output", "analysis", "iteration", "done", "max_iterations")
    safe = {k: v for k, v in result.items() if k in show_keys}

    if args.json:
        print(json.dumps(safe, indent=2, ensure_ascii=False))
    else:
        sep = "-" * 60
        print(f"{sep}\n ПЛАН\n{sep}")
        print(result.get("plan", ""))
        print(f"\n{sep}\n ВЫЗОВЫ ИНСТРУМЕНТОВ\n{sep}")
        print(result.get("tool_output", ""))
        print(f"\n{sep}\n АНАЛИЗ\n{sep}")
        print(result.get("analysis", ""))
        print(f"\n{sep}")
        print(f"Итерации: {result.get('iteration', '?')}, done={result.get('done', '?')}")


if __name__ == "__main__":
    main()
