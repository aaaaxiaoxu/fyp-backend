from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Optional

from openai import OpenAI

from settings import settings


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
