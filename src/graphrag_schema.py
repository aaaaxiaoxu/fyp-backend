from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]

class ChatMessage(BaseModel):
    role: Role
    content: str

class GraphRAGChatRequest(BaseModel):
    messages: list[ChatMessage]
    stream: bool = True
    top_k_chunks: int = Field(default=8, ge=1, le=30)
    max_hops: int = Field(default=2, ge=1, le=3)
