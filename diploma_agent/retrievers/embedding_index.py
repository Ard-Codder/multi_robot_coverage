"""JSONL embedding index for optional semantic retrieval."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from diploma_agent.retrievers.local_docs import SourceChunk
from diploma_agent.state import ensure_workspace


@dataclass(frozen=True)
class EmbeddedChunk:
    source: str
    title: str
    text: str
    embedding: list[float]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class JsonlEmbeddingIndex:
    def __init__(self, name: str, workspace: Path | None = None) -> None:
        self.workspace = ensure_workspace(workspace)
        safe = "".join(ch for ch in name if ch.isalnum() or ch in "._-")
        self.path = self.workspace / "indexes" / f"{safe}.embeddings.jsonl"

    def write(self, chunks: list[SourceChunk], embeddings: list[list[float]]) -> int:
        rows = [
            EmbeddedChunk(chunk.source, chunk.title, chunk.text, embedding)
            for chunk, embedding in zip(chunks, embeddings)
        ]
        with self.path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
        return len(rows)

    def load(self) -> list[EmbeddedChunk]:
        if not self.path.exists():
            return []
        rows: list[EmbeddedChunk] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(EmbeddedChunk(**json.loads(line)))
        return rows

    def search(self, query_embedding: list[float], top_k: int = 7) -> list[SourceChunk]:
        scored = [
            SourceChunk(row.source, row.title, row.text, cosine(query_embedding, row.embedding))
            for row in self.load()
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return [row for row in scored[:top_k] if row.score > 0]
