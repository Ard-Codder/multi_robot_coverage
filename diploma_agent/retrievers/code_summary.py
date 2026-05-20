"""Lightweight code retriever based on AST symbols.

This keeps the first version cheap: it indexes signatures and docstrings instead
of passing full source files to the LLM.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from diploma_agent import config
from diploma_agent.retrievers.local_docs import SourceChunk, tokenize

CODE_DIRS = ("coverage_lab", "coverage_sim", "experiments_lab", "experiments", "agent")


@dataclass(frozen=True)
class CodeSymbol:
    path: str
    kind: str
    name: str
    line: int
    signature: str
    doc: str

    def to_chunk(self) -> SourceChunk:
        text = f"{self.kind} `{self.name}` at line {self.line}\nSignature: {self.signature}"
        if self.doc:
            text += f"\nDocstring: {self.doc}"
        return SourceChunk(self.path, self.name, text)


def _signature(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = [arg.arg for arg in node.args.args]
        if node.args.vararg:
            args.append("*" + node.args.vararg.arg)
        if node.args.kwarg:
            args.append("**" + node.args.kwarg.arg)
        return f"{node.name}({', '.join(args)})"
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(base) for base in node.bases] if hasattr(ast, "unparse") else []
        suffix = f"({', '.join(bases)})" if bases else ""
        return f"class {node.name}{suffix}"
    return ""


def extract_symbols(path: Path, root: Path | None = None) -> list[CodeSymbol]:
    project_root = root or config.ROOT
    rel = path.relative_to(project_root).as_posix()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    symbols: list[CodeSymbol] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            symbols.append(
                CodeSymbol(rel, "class", node.name, node.lineno, _signature(node), ast.get_docstring(node) or "")
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_") and node.name not in {"__init__"}:
                continue
            symbols.append(
                CodeSymbol(rel, "function", node.name, node.lineno, _signature(node), ast.get_docstring(node) or "")
            )
    return symbols


def _path_bonus(path: str, query: str) -> float:
    score = 0.0
    q = query.lower()
    hints = {
        "алгоритм": "algorithms",
        "algorithm": "algorithms",
        "метрик": "metrics",
        "metric": "metrics",
        "эксперимент": "experiments",
        "experiment": "experiments",
        "rl": "rl",
        "ppo": "rl",
        "ml": "ml_planner",
        "симуляц": "sim",
        "simulation": "sim",
        "mapf": "mapf",
    }
    for word, part in hints.items():
        if word in q and part in path.lower():
            score += 1.0
    return score


class CodeSummaryRetriever:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or config.ROOT
        self._chunks: list[SourceChunk] | None = None

    def chunks(self) -> list[SourceChunk]:
        if self._chunks is not None:
            return self._chunks
        chunks: list[SourceChunk] = []
        for dirname in CODE_DIRS:
            base = self.root / dirname
            if not base.exists():
                continue
            for path in base.rglob("*.py"):
                if "__pycache__" in path.parts:
                    continue
                for symbol in extract_symbols(path, self.root):
                    chunks.append(symbol.to_chunk())
        self._chunks = chunks
        return chunks

    def search(self, query: str, top_k: int = 7) -> list[SourceChunk]:
        terms = tokenize(query)
        scored: list[SourceChunk] = []
        for chunk in self.chunks():
            haystack = tokenize(chunk.source + " " + chunk.title + " " + chunk.text)
            overlap = terms & haystack
            score = float(len(overlap)) + _path_bonus(chunk.source, query)
            if score <= 0 and re.search(r"\b(code|код|архитектур)", query, re.IGNORECASE):
                score = _path_bonus(chunk.source, query)
            if score > 0:
                scored.append(SourceChunk(chunk.source, chunk.title, chunk.text, score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]
