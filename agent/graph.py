"""LangGraph: plan → engineer → execute → analyze → (loop)."""

from __future__ import annotations

import re
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent import settings
from agent.research_tools import search_arxiv, search_habr_rss, search_semantic_scholar
from agent.tools import extract_and_run_tool_calls, run_coverage_experiment

MAX_FIELD_CHARS = 1200


def _trim(text: str, limit: int = MAX_FIELD_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[обрезано]"


def _strip_thinking(text: str) -> str:
    """Remove verbose 'Thinking Process:' / numbered reasoning blocks from Qwen output."""
    cleaned = re.sub(
        r"(?:Thinking\s+Process|Internal\s+Monologue)[:\s]*\n.*?(?=\n(?:##|\*\*|STATUS:|$))",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(r"^\d+\.\s+\*\*.*?\*\*:?\s*\n(?:\s+[\*\-].*\n)*", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip() or text


class AgentState(TypedDict, total=False):
    task: str
    plan: str
    engineer_output: str
    tool_output: str
    analysis: str
    iteration: int
    max_iterations: int
    done: bool


def _llm_planner() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.get_base_url(),
        api_key=settings.get_api_key(),
        model=settings.get_model_planner(),
        temperature=settings.get_temperature_planner(),
        max_tokens=1024,
    )


def _llm_coder() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.get_base_url(),
        api_key=settings.get_api_key(),
        model=settings.get_model_coder(),
        temperature=settings.get_temperature_coder(),
        max_tokens=1500,
    )


def _llm_analyzer() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.get_base_url(),
        api_key=settings.get_api_key(),
        model=settings.get_model_planner(),
        temperature=0.2,
        max_tokens=1024,
    )


def node_plan(state: AgentState) -> AgentState:
    task = state.get("task", "")
    llm = _llm_planner()
    msg = llm.invoke(
        [
            SystemMessage(content=prompts.PLANNER_SYSTEM),
            HumanMessage(content=f"Задача:\n{task}"),
        ]
    )
    content = getattr(msg, "content", str(msg))
    raw = content if isinstance(content, str) else str(content)
    return {"plan": _strip_thinking(raw)}


def node_engineer(state: AgentState) -> AgentState:
    task = state.get("task", "")
    plan = _trim(state.get("plan", ""))
    tool_feedback = _trim(state.get("tool_output", ""))
    analysis = _trim(state.get("analysis", ""))
    llm = _llm_coder()
    human = (
        f"Задача: {task}\n\n"
        f"План (кратко):\n{plan}\n\n"
        f"Предыдущий результат:\n{tool_feedback or '(нет)'}\n\n"
        f"Обратная связь аналитика:\n{analysis or '(нет)'}"
    )
    msg = llm.invoke(
        [
            SystemMessage(content=prompts.ENGINEER_SYSTEM),
            HumanMessage(content=human),
        ]
    )
    content = getattr(msg, "content", str(msg))
    text = content if isinstance(content, str) else str(content)
    return {"engineer_output": text}


_KNOWN_ALGOS = {"random_walk", "grid", "voronoi", "frontier", "zonal"}

_RESEARCH_HINT = (
    "arxiv",
    "semantic",
    "scholar",
    "habr",
    "хабр",
    "стать",
    "литератур",
    "обзор",
    "препринт",
    "публикац",
    "доклад",
)


def _wants_research(task_lower: str) -> bool:
    return any(k in task_lower for k in _RESEARCH_HINT)


_ISAAC_HINT = (
    "isaac",
    "айзек",
    "isaac sim",
    "omni",
    "simulationapp",
    "визуализац",
    "live run",
    "run_random_walk_live",
    "run_zonal_live",
    "zonal_live",
    "random_walk_live",
)


def _wants_isaac(task_lower: str) -> bool:
    return any(k in task_lower for k in _ISAAC_HINT)


def _infer_experiments_from_task(state: AgentState) -> list[dict]:
    """Parse task text for algorithm names and params when LLM fails to emit tool_call."""
    task = (state.get("task", "") + " " + state.get("plan", "")).lower()
    algos = [a for a in _KNOWN_ALGOS if a in task]
    if not algos:
        if _wants_research(task):
            return []
        algos = ["random_walk", "grid"]

    seed = 0
    max_steps = 300
    import re as _re
    m = _re.search(r"seed\s*[=:]?\s*(\d+)", task)
    if m:
        seed = int(m.group(1))
    m = _re.search(r"max.?steps\s*[=:]?\s*(\d+)", task)
    if m:
        max_steps = int(m.group(1))

    return [{"algorithm": a, "seed": seed, "max_steps": max_steps} for a in algos]


def node_execute(state: AgentState) -> AgentState:
    text = state.get("engineer_output", "")
    log = extract_and_run_tool_calls(text)
    if not log.strip():
        task_raw = state.get("task", "") or ""
        task_lower = (task_raw + " " + state.get("plan", "")).lower()
        parts: list[str] = []
        if _wants_research(task_lower):
            q = task_raw.strip()[:220] or "multi robot coverage path planning"
            parts.append("=== search_arxiv (fallback) ===\n" + search_arxiv(q, max_results=6))
            parts.append("=== search_semantic_scholar (fallback) ===\n" + search_semantic_scholar(q, max_results=6))
            parts.append("=== search_habr_rss (fallback) ===\n" + search_habr_rss(q, max_items=6))
        use_isaac = _wants_isaac(task_lower)
        experiments = _infer_experiments_from_task(state)
        for exp in experiments:
            parts.append(f"=== run_coverage_experiment({exp['algorithm']}, use_isaac_sim={use_isaac}) ===")
            parts.append(
                run_coverage_experiment(
                    algorithm=exp["algorithm"],
                    seed=exp["seed"],
                    max_steps=exp["max_steps"],
                    target_coverage=0.95,
                    use_isaac_sim=use_isaac,
                    save_plots=False,
                )
            )
        if not parts:
            parts.append("=== run_coverage_experiment (fallback) ===\n")
            parts.append(
                run_coverage_experiment(
                    algorithm="voronoi",
                    seed=0,
                    max_steps=300,
                    target_coverage=0.95,
                    use_isaac_sim=use_isaac,
                    save_plots=False,
                    output_name="agent_fallback_voronoi",
                )
            )
        log = "\n\n".join(parts)
    return {"tool_output": log}


def node_analyze(state: AgentState) -> AgentState:
    task = state.get("task", "")
    tools_out = _trim(state.get("tool_output", ""))
    llm = _llm_analyzer()
    msg = llm.invoke(
        [
            SystemMessage(content=prompts.ANALYZER_SYSTEM),
            HumanMessage(
                content=f"Задача: {task}\n\nРезультаты:\n{tools_out}"
            ),
        ]
    )
    content = getattr(msg, "content", str(msg))
    raw_analysis = content if isinstance(content, str) else str(content)
    analysis = _strip_thinking(raw_analysis)
    done = bool(re.search(r"STATUS:\s*COMPLETE", raw_analysis, re.IGNORECASE))
    it = int(state.get("iteration", 0)) + 1
    return {"analysis": analysis, "done": done, "iteration": it}


def route_after_analyze(state: AgentState) -> Literal["engineer", "end"]:
    if state.get("done"):
        return "end"
    if int(state.get("iteration", 0)) >= int(state.get("max_iterations", 3)):
        return "end"
    return "engineer"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("plan", node_plan)
    g.add_node("engineer", node_engineer)
    g.add_node("execute", node_execute)
    g.add_node("analyze", node_analyze)

    g.add_edge(START, "plan")
    g.add_edge("plan", "engineer")
    g.add_edge("engineer", "execute")
    g.add_edge("execute", "analyze")
    g.add_conditional_edges(
        "analyze",
        route_after_analyze,
        {"engineer": "engineer", "end": END},
    )
    return g.compile()


def run_agent_loop(
    task: str,
    max_iterations: int = 3,
) -> dict[str, Any]:
    app = build_graph()
    out: AgentState = {
        "task": task,
        "max_iterations": max_iterations,
        "iteration": 0,
        "done": False,
    }
    final = app.invoke(out)
    return dict(final)
