from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Optional, AsyncGenerator

import httpx
from openai import OpenAI

from .settings import settings


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_OBJ_RE.search(text)
    if not m:
        raise ValueError(f"Model response is not JSON. head={text[:200]!r}")
    return json.loads(m.group(0))


class DeepSeekClient:
    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            timeout=settings.TIMEOUT_S,
        )

    def chat_json(self, system: str, user: str) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for attempt in range(1, settings.MAX_RETRIES + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=settings.TEMPERATURE,
                    max_tokens=settings.MAX_TOKENS,
                    stream=False,
                )
                content = resp.choices[0].message.content or ""
                return _extract_json_object(content)
            except Exception as e:
                last_err = e
                time.sleep(settings.RETRY_BACKOFF_S * attempt)
        raise RuntimeError(f"DeepSeek call failed after retries. last_err={last_err}") from last_err

    async def chat_completion_async(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> Dict[str, Any]:
        payload = {
            "model": model or settings.LLM_MODEL,
            "messages": messages,
            "stream": stream,
            "temperature": settings.TEMPERATURE if temperature is None else temperature,
            "max_tokens": settings.MAX_TOKENS if max_tokens is None else max_tokens,
        }
        async with httpx.AsyncClient(base_url=settings.LLM_BASE_URL, timeout=settings.TIMEOUT_S) as client:
            r = await client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
                json=payload,
            )
            r.raise_for_status()
            return r.json()

    async def chat_completion_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> AsyncGenerator[str, None]:
        payload = {
            "model": model or settings.LLM_MODEL,
            "messages": messages,
            "stream": True,
            "temperature": settings.TEMPERATURE if temperature is None else temperature,
            "max_tokens": settings.MAX_TOKENS if max_tokens is None else max_tokens,
        }
        async with httpx.AsyncClient(base_url=settings.LLM_BASE_URL, timeout=None) as client:
            async with client.stream(
                "POST",
                "/chat/completions",
                headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
                json=payload,
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0].get("delta", {})
                        token = delta.get("content")
                        if token:
                            yield token
                    except Exception:
                        continue
