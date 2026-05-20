"""Configuration helpers for the local thesis agent."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def workspace_dir() -> Path:
    raw = env_str("DIPLOMA_WORKSPACE", str(ROOT / "thesis_workspace"))
    return Path(raw).expanduser().resolve()


def lmstudio_base_url() -> str:
    return env_str("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")


def lmstudio_api_key() -> str:
    return env_str("LMSTUDIO_API_KEY", "lm-studio")


def diploma_model() -> str:
    return env_str("DIPLOMA_MODEL", "google/gemma-4-26b-a4b")


def embedding_model() -> str:
    return env_str("DIPLOMA_EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")


def role_temperature(role: str) -> float:
    defaults = {
        "planner": "0.3",
        "writer": "0.7",
        "reviewer": "0.2",
        "summarizer": "0.2",
    }
    key = f"DIPLOMA_TEMP_{role.upper()}"
    return float(env_str(key, defaults.get(role, "0.4")))
