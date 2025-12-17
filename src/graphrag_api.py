from __future__ import annotations

import json
import uuid
from typing import Any, AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from .graphrag_schema import GraphRAGChatRequest
from .graphrag_retriever import neo4j_retrieve, build_context
from .llm_client import DeepSeekClient


router = APIRouter(prefix="/graphrag", tags=["GraphRAG"])

llm = DeepSeekClient()


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def extract_entities_via_llm(messages: list[dict[str, str]]) -> dict[str, Any]:
    """
    用 LLM 从最后一条 user 问句抽取实体/关键词，输出严格 JSON。
    """
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

    system = (
        "你是信息抽取器。给定小说问句，抽取可能的实体与关键词。"
        "只输出严格 JSON，不要输出多余文字。字段：persons, locations, orgs, events, keywords，值为字符串数组。"
    )

    resp = await llm.chat_completion_async(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": last_user},
        ],
        stream=False,
        temperature=0.0,
        # max_tokens 可以不传，默认走 settings.MAX_TOKENS
    )

    content = (resp["choices"][0]["message"].get("content") or "").strip()

    try:
        data = json.loads(content)
        # 基础兜底，避免缺字段
        return {
            "persons": data.get("persons", []) or [],
            "locations": data.get("locations", []) or [],
            "orgs": data.get("orgs", []) or [],
            "events": data.get("events", []) or [],
            "keywords": data.get("keywords", []) or [],
        }
    except Exception:
        # 抽取失败就退化为关键词检索
        return {"persons": [], "locations": [], "orgs": [], "events": [], "keywords": [last_user]}


@router.post("/chat")
async def graphrag_chat(req: GraphRAGChatRequest):
    """
    SSE 流式输出：
      - meta: entity_extracted / retrieved
      - token: 每个增量 token
      - done: 完成
    """
    req_id = str(uuid.uuid4())
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    async def event_gen() -> AsyncGenerator[str, None]:
        # A) 抽取实体
        ent = await extract_entities_via_llm(messages)
        yield sse("meta", {"id": req_id, "stage": "entity_extracted", "entities": ent})

        # B) Neo4j 检索
        retrieved = await neo4j_retrieve(ent, top_k_chunks=req.top_k_chunks, max_hops=req.max_hops)
        yield sse(
            "meta",
            {
                "id": req_id,
                "stage": "retrieved",
                "edges": len(retrieved.get("edges", [])),
                "chunks": [c["chunk_id"] for c in retrieved.get("chunks", [])],
            },
        )

        # C) 构造上下文 + 流式回答
        context = build_context(retrieved)

        augmented = [
            {"role": "system", "content": "回答要求：不得编造；若证据不足请说明；末尾输出：Citations: [chunk_id,...]"},
            {"role": "system", "content": context},
            *messages,
        ]

        async for token in llm.chat_completion_stream(
            messages=augmented,
            temperature=0.2,
        ):
            yield sse("token", {"id": req_id, "delta": token})

        yield sse("done", {"id": req_id, "stage": "completed"})

    return StreamingResponse(event_gen(), media_type="text/event-stream")
