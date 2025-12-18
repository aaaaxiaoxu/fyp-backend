from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Conversation, Message


def _uuid() -> str:
    return uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def create_conversation(
    session: AsyncSession, user_id: str, title: str | None = None
) -> Conversation:
    conv = Conversation(
        id=_uuid(),
        user_id=user_id,
        title=(title or "New chat"),
        created_at=_now(),
        updated_at=_now(),
    )
    session.add(conv)
    await session.commit()
    return conv


async def list_conversations(
    session: AsyncSession, user_id: str, limit: int = 50
) -> list[Conversation]:
    q = (
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(desc(Conversation.updated_at))
        .limit(limit)
    )
    res = await session.execute(q)
    return list(res.scalars().all())


async def get_conversation(
    session: AsyncSession, user_id: str, conversation_id: str
) -> Conversation | None:
    q = select(Conversation).where(
        Conversation.id == conversation_id, Conversation.user_id == user_id
    )
    res = await session.execute(q)
    return res.scalar_one_or_none()


async def touch_conversation(session: AsyncSession, conversation_id: str) -> None:
    await session.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id)
        .values(updated_at=_now())
    )
    await session.commit()


async def add_message(
    session: AsyncSession,
    conversation_id: str,
    role: str,
    content: str,
    meta: dict[str, Any] | None = None,
) -> Message:
    """
    写入消息，并同步更新 Conversation.updated_at（同一次 commit）。
    """
    msg = Message(
        id=_uuid(),
        conversation_id=conversation_id,
        role=role,
        content=content,
        created_at=_now(),
        meta_json=json.dumps(meta or {}, ensure_ascii=False),
    )
    session.add(msg)

    await session.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id)
        .values(updated_at=_now())
    )

    await session.commit()
    return msg


async def list_messages(
    session: AsyncSession, conversation_id: str, limit: int = 50
) -> list[Message]:
    """
    返回时间正序 messages（用于前端展示）。
    """
    q = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .limit(limit)
    )
    res = await session.execute(q)
    return list(res.scalars().all())


async def list_recent_messages_for_llm(
    session: AsyncSession, conversation_id: str, limit: int = 20
) -> list[dict[str, str]]:
    """
    给 LLM 用：最近 N 条（时间正序）
    """
    limit = max(1, min(limit, 100))
    q = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(desc(Message.created_at))
        .limit(limit)
    )
    res = await session.execute(q)
    rows = list(res.scalars().all())
    rows.reverse()
    return [{"role": m.role, "content": m.content} for m in rows]


async def get_message_count(session: AsyncSession, conversation_id: str) -> int:
    q = select(func.count(Message.id)).where(Message.conversation_id == conversation_id)
    res = await session.execute(q)
    n = res.scalar_one()
    return int(n or 0)


async def update_conversation_title(
    session: AsyncSession,
    user_id: str,
    conversation_id: str,
    title: str,
) -> None:
    """
    仅更新归属于 user_id 的会话标题，避免越权写。
    """
    title = (title or "").strip()
    if not title:
        return

    await session.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
        .values(title=title, updated_at=_now())
    )
    await session.commit()
