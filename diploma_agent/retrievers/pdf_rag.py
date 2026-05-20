"""Minimal private PDF RAG index.

The index is deliberately stored under ``thesis_workspace/indexes`` so papers and
derived chunks do not enter git.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from diploma_agent import config
from diploma_agent.retrievers.local_docs import SourceChunk, tokenize
from diploma_agent.state import ensure_workspace


@dataclass(frozen=True)
class PdfChunk:
    source: str
    chunk_id: str
    text: str
    embedding: list[float] | None = None


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Для PDF RAG установи pypdf: `pip install pypdf`.") from exc
    reader = PdfReader(str(path))
    parts = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(parts)


def _chunks(text: str, max_chars: int = 2200, overlap: int = 250) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    out: list[str] = []
    step = max(1, max_chars - overlap)
    for start in range(0, len(cleaned), step):
        piece = cleaned[start : start + max_chars].strip()
        if piece:
            out.append(piece)
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class PdfRagIndex:
    def __init__(self, workspace: Path | None = None) -> None:
        self.workspace = ensure_workspace(workspace)
        self.index_path = self.workspace / "indexes" / "pdf_chunks.jsonl"

    def add_pdf(self, pdf_path: Path, embeddings: list[list[float]] | None = None) -> int:
        text = _extract_pdf_text(pdf_path)
        pieces = _chunks(text)
        rows: list[PdfChunk] = []
        digest = hashlib.sha1(pdf_path.read_bytes()).hexdigest()[:12]
        for idx, piece in enumerate(pieces):
            emb = embeddings[idx] if embeddings and idx < len(embeddings) else None
            rows.append(PdfChunk(pdf_path.name, f"{digest}-{idx}", piece, emb))
        with self.index_path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
        return len(rows)

    def load(self) -> list[PdfChunk]:
        if not self.index_path.exists():
            return []
        rows: list[PdfChunk] = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(PdfChunk(**json.loads(line)))
        return rows

    def search(self, query: str, top_k: int = 5, query_embedding: list[float] | None = None) -> list[SourceChunk]:
        rows = self.load()
        if query_embedding and any(row.embedding for row in rows):
            scored = [
                SourceChunk(row.source, row.chunk_id, row.text, _cosine(query_embedding, row.embedding or []))
                for row in rows
            ]
        else:
            q = tokenize(query)
            scored = [
                SourceChunk(row.source, row.chunk_id, row.text, float(len(q & tokenize(row.text))))
                for row in rows
            ]
        scored = [row for row in scored if row.score > 0]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]
