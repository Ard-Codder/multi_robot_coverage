"""Настройки подключения к LM Studio (OpenAI-compatible API)."""

from __future__ import annotations

import os

# База вида http://127.0.0.1:1234/v1 — проверьте в LM Studio → Server.
def get_base_url() -> str:
    return os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")


def get_api_key() -> str:
    return os.environ.get("LMSTUDIO_API_KEY", "lm-studio")


def get_model_planner() -> str:
    return os.environ.get(
        "AGENT_MODEL_PLANNER",
        "qwen/qwen3.5-9b",
    )


def get_model_coder() -> str:
    return os.environ.get(
        "AGENT_MODEL_CODER",
        "deepseek-coder-v2-lite-instruct",
    )


def get_temperature_planner() -> float:
    return float(os.environ.get("AGENT_TEMP_PLANNER", "0.3"))


def get_temperature_coder() -> float:
    return float(os.environ.get("AGENT_TEMP_CODER", "0.2"))
