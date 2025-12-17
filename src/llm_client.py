from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Optional

from openai import OpenAI

from .settings import settings
from typing import Any, Dict, Optional, AsyncGenerator
import httpx



# 尽量从输出里“只提取 JSON 对象”
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    # 1) 直接就是 JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) 可能夹杂了说明文字/markdown，抓最外层 {...}
    m = _JSON_OBJ_RE.search(text)
    if not m:
        raise ValueError(f"Model response is not JSON. head={text[:200]!r}")

    return json.loads(m.group(0))


class DeepSeekClient:
    """
    按 DeepSeek 官方示例的 OpenAI SDK 调用方式封装：
      client = OpenAI(api_key=..., base_url="https://api.deepseek.com")
      client.chat.completions.create(...)
    """

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
        """
        用 httpx 走 OpenAI-compatible /chat/completions。
        用于非流式（stream=False）时拿完整 JSON。
        """
        payload = {
            "model": model or settings.LLM_MODEL,
            "messages": messages,
            "stream": stream,
            "temperature": settings.TEMPERATURE if temperature is None else temperature,
            "max_tokens": settings.MAX_TOKENS if max_tokens is None else max_tokens,
        }

        async with httpx.AsyncClient(
            base_url=settings.LLM_BASE_URL,
            timeout=settings.TIMEOUT_S,
        ) as client:
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
        """
        流式输出：yield 每个 token（字符串）。
        兼容 DeepSeek/OpenAI streaming：data: {...}\n\n 直到 data: [DONE]
        """
        payload = {
            "model": model or settings.LLM_MODEL,
            "messages": messages,
            "stream": True,
            "temperature": settings.TEMPERATURE if temperature is None else temperature,
            "max_tokens": settings.MAX_TOKENS if max_tokens is None else max_tokens,
        }

        # 流式最好不要用 TIMEOUT_S 的总超时，改成 None 或者更长
        async with httpx.AsyncClient(
            base_url=settings.LLM_BASE_URL,
            timeout=None,
        ) as client:
            async with client.stream(
                "POST",
                "/chat/completions",
                headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
                json=payload,
            ) as r:
                r.raise_for_status()

                async for line in r.aiter_lines():
                    if not line:
                        continue

                    # DeepSeek/OpenAI 通常是：data: {...}
                    if not line.startswith("data:"):
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
                        # 某些行可能不是标准 JSON，忽略即可
                        continue
