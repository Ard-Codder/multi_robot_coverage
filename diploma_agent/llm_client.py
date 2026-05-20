"""Small OpenAI-compatible client for LM Studio.

The project already uses LangChain in ``agent/``. This module keeps the thesis
pipeline lightweight and avoids another abstraction layer in the UI path.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable

from diploma_agent import config

Message = dict[str, str]


@dataclass(frozen=True)
class LLMRequest:
    messages: list[Message]
    role: str = "writer"
    max_tokens: int = 2200
    temperature: float | None = None


class LMStudioClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
        api_key: str | None = None,
        timeout_sec: float = 180.0,
    ) -> None:
        self.base_url = (base_url or config.lmstudio_base_url()).rstrip("/")
        self.model = model or config.diploma_model()
        self.embedding_model = embedding_model or config.embedding_model()
        self.api_key = api_key or config.lmstudio_api_key()
        self.timeout_sec = timeout_sec

    def chat(self, request: LLMRequest) -> str:
        return self._chat(request, retry_count=0)

    def _chat(self, request: LLMRequest, retry_count: int) -> str:
        payload = {
            "model": self.model,
            "messages": request.messages,
            "temperature": request.temperature
            if request.temperature is not None
            else config.role_temperature(request.role),
            "max_tokens": request.max_tokens,
            "reasoning": "off",
            "stream": False,
        }
        data = self._post_json("/chat/completions", payload)
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LM Studio response: {data}") from exc
        if not content and choice.get("finish_reason") == "length" and request.max_tokens < 1024:
            retry = LLMRequest(
                messages=request.messages,
                role=request.role,
                max_tokens=1024,
                temperature=request.temperature,
            )
            return self._chat(retry, retry_count + 1)
        if not content and retry_count < 2:
            retry_messages = request.messages + [
                {
                    "role": "user",
                    "content": "Верни финальный ответ в message.content. Не оставляй content пустым.",
                }
            ]
            retry = LLMRequest(
                messages=retry_messages,
                role=request.role,
                max_tokens=max(request.max_tokens * 2, 4096),
                temperature=request.temperature,
            )
            return self._chat(retry, retry_count + 1)
        if not isinstance(content, str):
            return str(content)
        content = content.strip()
        if not content:
            raise RuntimeError(
                "LM Studio вернул пустой ответ. Увеличь max_tokens/context или отключи чрезмерный reasoning для модели."
            )
        return content

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        payload = {"model": self.embedding_model, "input": list(texts)}
        data = self._post_json("/embeddings", payload)
        try:
            rows = sorted(data["data"], key=lambda item: item["index"])
            return [row["embedding"] for row in rows]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected embeddings response: {data}") from exc

    def is_available(self) -> tuple[bool, str]:
        try:
            data = self._get_json("/models")
        except RuntimeError as exc:
            return False, str(exc)
        models = [item.get("id", "") for item in data.get("data", []) if isinstance(item, dict)]
        if not models:
            return True, "LM Studio отвечает, но список моделей пуст."
        if self.model not in models:
            return True, f"LM Studio отвечает. Модель `{self.model}` не найдена; доступны: {', '.join(models[:5])}."
        return True, f"LM Studio готова, модель `{self.model}` доступна."

    def _post_json(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        return self._open(req)

    def _get_json(self, path: str) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        return self._open(req)

    def _open(self, req: urllib.request.Request) -> dict:
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LM Studio HTTP {exc.code}: {detail}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LM Studio request failed: {exc}") from exc


def system_user(system: str, user: str) -> list[Message]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
