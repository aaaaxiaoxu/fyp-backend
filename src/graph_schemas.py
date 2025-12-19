from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    id: str  # elementId(n)
    labels: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    id: str  # elementId(r)
    type: str
    source: str  # elementId(startNode(r))
    target: str  # elementId(endNode(r))
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class SubgraphRequest(BaseModel):
    seed_eid: str
    depth: int = 1
    direction: Literal["both", "out", "in"] = "both"
    limit_paths: int = 200
    include_snippet: bool = True
    snippet_len: int = 120


# ===== 新增：左侧目录/列表 =====

EntityLabel = Literal["Person", "Event"]


class CatalogResponse(BaseModel):
    person_count: int
    event_count: int


class EntityListItem(BaseModel):
    eid: str
    label: str
    name: str | None = None
    score: int = 0  # mentions/evidence_count 之类
    first_seen_chapter: str | None = None
    first_seen_chunk: str | None = None


class EntityListResponse(BaseModel):
    items: list[EntityListItem]
    total: int


class EvidenceItem(BaseModel):
    chunk_eid: str
    chunk_id: str | None = None
    chapter_id: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    snippet: str | None = None


class EvidenceResponse(BaseModel):
    items: list[EvidenceItem]
