from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .auth_deps import get_current_user
from .db import SessionLocal
from .chat_store import (
    add_message,
    create_conversation,
    get_conversation,
    get_message_count,
    list_conversations,
    list_messages,
    list_recent_messages_for_llm,
    update_conversation_title,
)
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
    last_n_history: int = Field(default=20, ge=1, le=100)


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


def _clean_title(s: str) -> str:
    s = (s or "").strip()
    s = s.strip('"').strip("'").strip()
    s = s.splitlines()[0].strip() if s else s
    # 去掉常见句尾标点/引号
    s = re.sub(r"[。！？!?,，:：；;“”\"'()\[\]{}]+$", "", s).strip()
    # 太长就截断
    if len(s) > 30:
        s = s[:30].strip()
    return s


async def generate_conversation_title(question: str, answer: str) -> str:
    """
    使用 LLM 生成简短会话标题；失败则 fallback。
    """
    q = (question or "").strip()
    a = (answer or "").strip()

    fallback = (q[:18] + "…") if len(q) > 18 else q
    if not fallback:
        fallback = "New chat"

    system = (
        "你是对话标题生成器。根据“用户问题”和“助手回答”，生成一个简短会话标题。\n"
        "要求：\n"
        "1) 中文为主，5-20字；或英文不超过8个单词\n"
        "2) 不要引号，不要换行，不要前缀（如：标题：）\n"
        "3) 只输出标题文本"
    )
    user = f"用户问题：{q}\n\n助手回答（摘要）：{a[:400]}"

    try:
        resp = await llm.chat_completion_async(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=False,
            temperature=0.2,
        )
        raw = (resp["choices"][0]["message"].get("content") or "").strip()
        title = _clean_title(raw)
        return title or fallback
    except Exception:
        return fallback


@router.post("")
async def create_conv(req: CreateConversationRequest, user=Depends(get_current_user)):
    async with SessionLocal() as db:
        conv = await create_conversation(db, user_id=user.id, title=req.title)
        return {
            "conversation_id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at,
            "updated_at": conv.updated_at,
        }


@router.get("")
async def list_convs(
    limit: int = Query(default=50, ge=1, le=200),
    user=Depends(get_current_user),
):
    async with SessionLocal() as db:
        convs = await list_conversations(db, user_id=user.id, limit=limit)
        return [
            {
                "conversation_id": c.id,
                "title": c.title,
                "created_at": c.created_at,
                "updated_at": c.updated_at,
            }
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

    # SSE 开始前：鉴权 + 写 user 消息 + 取历史（避免占用 DB 贯穿整个流式）
    async with SessionLocal() as db:
        conv = await get_conversation(db, user_id=user.id, conversation_id=conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

        await add_message(
            db,
            conversation_id=conversation_id,
            role="user",
            content=req.content,
            meta={"request_id": req_id},
        )

        history_msgs = await list_recent_messages_for_llm(
            db,
            conversation_id=conversation_id,
            limit=req.last_n_history,
        )

    async def event_gen() -> AsyncGenerator[str, None]:
        ent: dict[str, Any] = {}
        retrieved: dict[str, Any] = {}
        chunk_ids: list[str] = []
        answer_buf: list[str] = []

        try:
            # 1) 抽实体
            ent = await extract_entities(req.content)
            yield sse("meta", {"id": req_id, "stage": "entity_extracted", "entities": ent})

            # 2) Neo4j 检索
            retrieved = await neo4j_retrieve(ent, top_k_chunks=req.top_k_chunks, max_hops=req.max_hops)
            chunk_ids = [c["chunk_id"] for c in retrieved.get("chunks", [])]
            yield sse(
                "meta",
                {"id": req_id, "stage": "retrieved", "edges": len(retrieved.get("edges", [])), "chunks": chunk_ids},
            )

            # 3) 构造上下文 + 流式回答
            context = build_context(retrieved)
            augmented = [
                {
                    "role": "system",
                    "content": (
                        "回答要求：不得编造；若证据不足请说明；"
                        "末尾输出：Citations: [chunk_id,...]（必须来自检索到的 chunks，去重后输出）"
                    ),
                },
                {"role": "system", "content": context},
                *history_msgs,
            ]

            async for token in llm.chat_completion_stream(messages=augmented, temperature=0.2):
                answer_buf.append(token)
                yield sse("token", {"id": req_id, "delta": token})

            yield sse("done", {"id": req_id, "stage": "completed"})

        except asyncio.CancelledError:
            yield sse("error", {"id": req_id, "message": "client cancelled"})
            raise

        except Exception as e:
            yield sse("error", {"id": req_id, "message": str(e)})
            raise

        finally:
            # 流结束后：写 assistant + 自动标题（不影响前端拿到 done）
            answer = "".join(answer_buf).strip()
            if not answer:
                return

            async with SessionLocal() as db2:
                conv2 = await get_conversation(db2, user_id=user.id, conversation_id=conversation_id)
                if not conv2:
                    return

                # 写 assistant
                await add_message(
                    db2,
                    conversation_id=conversation_id,
                    role="assistant",
                    content=answer,
                    meta={
                        "request_id": req_id,
                        "entities": ent,
                        "retrieved": {"edges_count": len(retrieved.get("edges", [])) if retrieved else 0, "chunks": chunk_ids},
                    },
                )

                # ✅ 自动标题：仅在默认标题且首轮（消息数<=2）时生成
                if (conv2.title or "").strip() == "New chat":
                    n = await get_message_count(db2, conversation_id=conversation_id)
                    # 首轮通常就是：user + assistant = 2
                    if n <= 2:
                        new_title = await generate_conversation_title(req.content, answer)
                        new_title = _clean_title(new_title)
                        if new_title and new_title != "New chat":
                            await update_conversation_title(
                                db2,
                                user_id=user.id,
                                conversation_id=conversation_id,
                                title=new_title,
                            )

    return StreamingResponse(event_gen(), media_type="text/event-stream")
