from __future__ import annotations

import json
import uuid
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from .auth_deps import get_current_user
from .db import SessionLocal
from .chat_store import create_conversation, list_conversations, get_conversation, list_messages, add_message
from .graphrag_retriever import neo4j_retrieve, build_context
from .llm_client import DeepSeekClient

router = APIRouter(prefix="/conversations", tags=["Conversations"])
llm = DeepSeekClient()


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class CreateConversationRequest(BaseModel):
    title: str | None = None


class ChatRequest(BaseModel):
    content: str = Field(min_length=1)
    stream: bool = True
    top_k_chunks: int = Field(default=8, ge=1, le=30)
    max_hops: int = Field(default=2, ge=1, le=3)


async def extract_entities(question: str) -> dict[str, Any]:
    system = (
        "你是信息抽取器。给定小说问句，抽取可能的实体与关键词。"
        "只输出严格 JSON，不要输出多余文字。字段：persons, locations, orgs, events, keywords，值为字符串数组。"
    )
    resp = await llm.chat_completion_async(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        stream=False,
        temperature=0.0,
    )
    content = (resp["choices"][0]["message"].get("content") or "").strip()
    try:
        data = json.loads(content)
        return {
            "persons": data.get("persons", []) or [],
            "locations": data.get("locations", []) or [],
            "orgs": data.get("orgs", []) or [],
            "events": data.get("events", []) or [],
            "keywords": data.get("keywords", []) or [],
        }
    except Exception:
        return {"persons": [], "locations": [], "orgs": [], "events": [], "keywords": [question]}


@router.post("")
async def create_conv(req: CreateConversationRequest, user=Depends(get_current_user)):
    async with SessionLocal() as db:
        conv = await create_conversation(db, user_id=user.id, title=req.title)
        return {"conversation_id": conv.id, "title": conv.title, "created_at": conv.created_at, "updated_at": conv.updated_at}


@router.get("")
async def list_convs(limit: int = Query(default=50, ge=1, le=200), user=Depends(get_current_user)):
    async with SessionLocal() as db:
        convs = await list_conversations(db, user_id=user.id, limit=limit)
        return [
            {"conversation_id": c.id, "title": c.title, "created_at": c.created_at, "updated_at": c.updated_at}
            for c in convs
        ]


@router.get("/{conversation_id}/messages")
async def get_msgs(
    conversation_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    user=Depends(get_current_user),
):
    async with SessionLocal() as db:
        conv = await get_conversation(db, user_id=user.id, conversation_id=conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

        msgs = await list_messages(db, conversation_id, limit=limit)
        return [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at,
                "meta": json.loads(m.meta_json or "{}"),
            }
            for m in msgs
        ]


@router.post("/{conversation_id}/chat")
async def chat(conversation_id: str, req: ChatRequest, user=Depends(get_current_user)):
    req_id = str(uuid.uuid4())

    async def event_gen() -> AsyncGenerator[str, None]:
        async with SessionLocal() as db:
            conv = await get_conversation(db, user_id=user.id, conversation_id=conversation_id)
            if not conv:
                yield sse("error", {"id": req_id, "message": "Conversation not found"})
                return

            # 1) 先写入 user 消息
            await add_message(
                db,
                conversation_id=conversation_id,
                role="user",
                content=req.content,
                meta={"request_id": req_id},
            )

            # 2) 取最近历史（后端管理多轮）
            history = await list_messages(db, conversation_id, limit=20)
            history_msgs = [{"role": m.role, "content": m.content} for m in history]

            # 3) 抽实体
            ent = await extract_entities(req.content)
            yield sse("meta", {"id": req_id, "stage": "entity_extracted", "entities": ent})

            # 4) Neo4j 检索
            retrieved = await neo4j_retrieve(ent, top_k_chunks=req.top_k_chunks, max_hops=req.max_hops)
            chunk_ids = [c["chunk_id"] for c in retrieved.get("chunks", [])]
            yield sse("meta", {"id": req_id, "stage": "retrieved", "edges": len(retrieved.get("edges", [])), "chunks": chunk_ids})

            # 5) 构造上下文 + 流式回答
            context = build_context(retrieved)
            augmented = [
                {"role": "system", "content": "回答要求：不得编造；若证据不足请说明；末尾输出：Citations: [chunk_id,...]（必须来自检索到的 chunks，去重后输出）"},
                {"role": "system", "content": context},
                *history_msgs,
            ]

            answer_buf: list[str] = []
            async for token in llm.chat_completion_stream(messages=augmented, temperature=0.2):
                answer_buf.append(token)
                yield sse("token", {"id": req_id, "delta": token})

            answer = "".join(answer_buf).strip()

            # 6) 写入 assistant 消息（带检索元信息）
            await add_message(
                db,
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
                meta={
                    "request_id": req_id,
                    "entities": ent,
                    "retrieved": {"edges_count": len(retrieved.get("edges", [])), "chunks": chunk_ids},
                },
            )

            yield sse("done", {"id": req_id, "stage": "completed"})

    return StreamingResponse(event_gen(), media_type="text/event-stream")
