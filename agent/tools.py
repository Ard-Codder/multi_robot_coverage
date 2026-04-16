"""
Инструменты агента: запуск симуляции и работа с файлами внутри репозитория.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coverage_sim.simulation.simulation_loop import run_experiment  # noqa: E402

from agent.research_tools import (  # noqa: E402
    search_arxiv,
    search_habr_rss,
    search_semantic_scholar,
)


def _safe_relative_path(path_str: str) -> Path:
    p = (PROJECT_ROOT / path_str).resolve()
    if not str(p).startswith(str(PROJECT_ROOT.resolve())):
        raise ValueError("Путь вне репозитория")
    return p


def read_repo_file(relative_path: str, max_chars: int = 12000) -> str:
    p = _safe_relative_path(relative_path)
    if not p.is_file():
        return f"[error] Файл не найден: {relative_path}"
    text = p.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... [обрезано, всего {len(text)} символов]"
    return text


def write_repo_file(relative_path: str, content: str) -> str:
    p = _safe_relative_path(relative_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"ok: записано {relative_path} ({len(content)} символов)"


def summarize_experiment_result(result: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "algorithm",
        "coverage_percent",
        "time_to_coverage_sec",
        "distance_travelled_m",
        "efficiency",
        "steps",
        "load_balance_cv",
        "distance_mean_m",
        "distance_std_m",
        "experiment_seed",
        "target_coverage",
        "max_steps",
        "use_isaac_sim",
        "blocked_moves",
    )
    out: Dict[str, Any] = {k: result.get(k) for k in keys if k in result}
    return out


def run_coverage_experiment(
    algorithm: str,
    seed: int = 0,
    max_steps: int = 400,
    target_coverage: float = 0.95,
    use_isaac_sim: bool = False,
    save_plots: bool = False,
    output_name: Optional[str] = None,
) -> str:
    """
    Запускает один прогон run_experiment (kinematic fallback при use_isaac_sim=False).
    Возвращает JSON-строку с краткой сводкой + путь к полному JSON.
    """
    stem = output_name or f"agent_{algorithm}_s{seed}"
    out_dir = PROJECT_ROOT / "results" / "agent_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{stem}.json"
    out_plot = out_dir / stem

    try:
        result = run_experiment(
            algorithm_name=algorithm,
            output_json=out_json,
            output_plot_prefix=out_plot,
            root_dir=PROJECT_ROOT,
            keep_visualization_open_sec=0.0,
            require_isaac=False,
            experiment_seed=seed,
            max_steps=max_steps,
            target_coverage=target_coverage,
            save_plots=save_plots,
            use_isaac_sim=use_isaac_sim,
        )
    except Exception as e:
        return json.dumps({"error": str(e), "type": type(e).__name__}, ensure_ascii=False)

    summary = summarize_experiment_result(result)
    summary["full_json_path"] = str(out_json.relative_to(PROJECT_ROOT))
    return json.dumps(summary, indent=2, ensure_ascii=False)


def read_json_result(relative_path: str) -> str:
    p = _safe_relative_path(relative_path)
    if not p.suffix.lower() == ".json" or not p.is_file():
        return f"[error] Нужен .json файл: {relative_path}"
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        slim = {k: data[k] for k in summarize_experiment_result(data) if k in data}
        return json.dumps(slim, indent=2, ensure_ascii=False)
    return json.dumps(data, indent=2, ensure_ascii=False)[:8000]


def run_batch_subprocess(config_relative: str = "experiments/batch_quick.yaml") -> str:
    """Запускает experiments/run_batch.py (отдельный процесс — стабильно на Windows)."""
    cfg = _safe_relative_path(config_relative)
    if not cfg.is_file():
        return json.dumps({"error": f"Нет файла: {config_relative}"}, ensure_ascii=False)
    rel_cfg = cfg.relative_to(PROJECT_ROOT)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "run_batch.py"),
        "--config",
        str(rel_cfg),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out: Dict[str, Any] = {
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }
    summary_csv = PROJECT_ROOT / "results" / "batch_quick" / "summary.csv"
    if summary_csv.is_file():
        out["summary_csv"] = str(summary_csv.relative_to(PROJECT_ROOT))
    return json.dumps(out, indent=2, ensure_ascii=False)


def list_results_dir(subdir: str = "results/agent_runs") -> str:
    d = _safe_relative_path(subdir)
    if not d.is_dir():
        return "[]"
    files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]
    return json.dumps([str(f.relative_to(PROJECT_ROOT)) for f in files], ensure_ascii=False)


ALLOWED_ISAAC_LIVE = frozenset(
    {
        "experiments/run_random_walk_live.py",
        "experiments/run_zonal_live.py",
    }
)


def run_isaac_live_script(script: str = "experiments/run_random_walk_live.py") -> str:
    """
    Запускает live-скрипт с require_isaac=True (окно симулятора).
    Обычно нужен интерпретатор из установки NVIDIA Isaac Sim, не системный Python.
    """
    s = script.replace("\\", "/").strip()
    if s not in ALLOWED_ISAAC_LIVE:
        return json.dumps(
            {
                "error": f"Разрешены только: {sorted(ALLOWED_ISAAC_LIVE)}",
            },
            ensure_ascii=False,
        )
    path = PROJECT_ROOT / s
    if not path.is_file():
        return json.dumps({"error": f"Нет файла: {s}"}, ensure_ascii=False)
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "timeout 600s"}, ensure_ascii=False)
    out: Dict[str, Any] = {
        "script": s,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-8000:],
        "stderr": proc.stderr[-8000:],
    }
    if proc.returncode != 0:
        out["hint"] = (
            "Если в stderr про isaacsim/omni — запустите этот скрипт из python.bat Isaac Sim."
        )
    return json.dumps(out, indent=2, ensure_ascii=False)


TOOL_SPECS: List[Dict[str, str]] = [
    {
        "name": "run_coverage_experiment",
        "description": "Запустить один эксперимент покрытия (algorithm, seed, max_steps, ...). Возвращает JSON со сводкой.",
    },
    {
        "name": "read_repo_file",
        "description": "Прочитать текстовый файл по пути относительно корня репозитория.",
    },
    {
        "name": "read_json_result",
        "description": "Прочитать JSON результат эксперимента (краткая сводка метрик).",
    },
    {
        "name": "run_batch_subprocess",
        "description": "Запустить пакетный YAML (run_batch.py), например experiments/batch_quick.yaml.",
    },
    {
        "name": "list_results_dir",
        "description": "Список последних json в results/agent_runs или другой подпапке.",
    },
    {
        "name": "search_arxiv",
        "description": "Поиск препринтов на arXiv (официальный API). query, max_results.",
    },
    {
        "name": "search_semantic_scholar",
        "description": "Поиск статей Semantic Scholar (публичный API). query, max_results.",
    },
    {
        "name": "search_habr_rss",
        "description": "Поиск статей на Habr через RSS. query, max_items.",
    },
    {
        "name": "run_isaac_live_script",
        "description": "Запуск live-скрипта Isaac (run_random_walk_live или run_zonal_live). script: путь от корня репо.",
    },
]


def dispatch_tool_call(name: str, arguments_json: str) -> str:
    try:
        args: Dict[str, Any] = json.loads(arguments_json) if arguments_json.strip() else {}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Невалидный JSON аргументов: {e}"}, ensure_ascii=False)

    try:
        if name == "run_coverage_experiment":
            return run_coverage_experiment(
                algorithm=str(args.get("algorithm", "random_walk")),
                seed=int(args.get("seed", 0)),
                max_steps=int(args.get("max_steps", 400)),
                target_coverage=float(args.get("target_coverage", 0.95)),
                use_isaac_sim=bool(args.get("use_isaac_sim", False)),
                save_plots=bool(args.get("save_plots", False)),
                output_name=args.get("output_name"),
            )
        if name == "read_repo_file":
            return read_repo_file(str(args.get("path", "")))
        if name == "read_json_result":
            return read_json_result(str(args.get("path", "")))
        if name == "write_repo_file":
            return write_repo_file(str(args.get("path", "")), str(args.get("content", "")))
        if name == "run_batch_subprocess":
            return run_batch_subprocess(str(args.get("config", "experiments/batch_quick.yaml")))
        if name == "list_results_dir":
            return list_results_dir(str(args.get("subdir", "results/agent_runs")))
        if name == "search_arxiv":
            return search_arxiv(
                str(args.get("query", "")),
                max_results=int(args.get("max_results", 8)),
            )
        if name == "search_semantic_scholar":
            return search_semantic_scholar(
                str(args.get("query", "")),
                max_results=int(args.get("max_results", 8)),
            )
        if name == "search_habr_rss":
            return search_habr_rss(
                str(args.get("query", "")),
                max_items=int(args.get("max_items", args.get("max_results", 10))),
            )
        if name == "run_isaac_live_script":
            return run_isaac_live_script(str(args.get("script", "experiments/run_random_walk_live.py")))
    except Exception as e:
        return json.dumps({"error": str(e), "type": type(e).__name__}, ensure_ascii=False)

    return json.dumps({"error": f"Неизвестный tool: {name}"}, ensure_ascii=False)


# --- парсинг вызовов инструментов из текста модели (если нет native tool calling) ---

_TOOL_CALL = re.compile(
    r"<tool_call>\s*(\w+)\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)


def extract_and_run_tool_calls(text: str) -> str:
    """Находит <tool_call>name{...}</tool_call>, выполняет, возвращает лог."""
    logs: List[str] = []
    for m in _TOOL_CALL.finditer(text or ""):
        name, raw_json = m.group(1), m.group(2)
        logs.append(f"=== {name} ===\n{dispatch_tool_call(name, raw_json)}")
    if not logs:
        return ""
    return "\n\n".join(logs)
