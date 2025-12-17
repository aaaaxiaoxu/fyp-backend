from __future__ import annotations

from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver

from .settings import settings


REL_WHITELIST = [
    "FRIEND_OF",
    "FAMILY_OF",
    "LOVES",
    "MARRIED_TO",
    "WORKS_FOR",
    "STUDIES_AT",
    "PARTICIPATES_IN",
    "HAPPENS_AT",
    "INVOLVES",
    "MENTIONED_IN",
    "SUPPORTED_BY",
]


def _build_driver() -> AsyncDriver:
    return AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )


driver: AsyncDriver = _build_driver()


async def close_driver() -> None:
    """
    可选：在 FastAPI shutdown 事件里调用，优雅关闭连接池。
    """
    await driver.close()


async def neo4j_retrieve(
    entity_pack: dict[str, Any],
    top_k_chunks: int,
    max_hops: int,
) -> dict[str, Any]:
    """
    返回：
      {
        "edges": [{"from":..., "rel":..., "to":...}, ...],
        "chunks": [{"chunk_id":..., "chapter_id":..., "text":...}, ...]
      }
    你需要确保 Neo4j 中存在对应 label/字段：
      - (:Person {name, alias?})
      - (:Event  {name, alias?})
      - (:Chunk  {chunk_id, chapter_id, text})
      - (:Event)-[:SUPPORTED_BY]->(:Chunk)
    """
    persons = (entity_pack.get("persons") or [])[:5]
    events = (entity_pack.get("events") or [])[:5]
    keywords = (entity_pack.get("keywords") or [])[:8]

    # 兜底关键词：如果抽取器没给 keywords，就把 persons/events 也当作关键词用于 contains
    fallback_keywords = []
    fallback_keywords.extend(persons)
    fallback_keywords.extend(events)

    # 去空、去重，避免 contains 匹配过多垃圾
    def _norm_list(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            x = (x or "").strip()
            if not x or x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    persons = _norm_list(persons)
    events = _norm_list(events)
    keywords = _norm_list(keywords)
    fallback_keywords = _norm_list(fallback_keywords)

    cypher = """
    WITH
      $persons AS persons,
      $events AS events,
      $keywords AS keywords,
      $fallback_keywords AS fallback_keywords

    // 1) 命中 Person / Event（alias 可选字段，用 coalesce 兼容）
    OPTIONAL MATCH (p:Person)
      WHERE p.name IN persons OR any(a IN coalesce(p.alias, []) WHERE a IN persons)

    OPTIONAL MATCH (e:Event)
      WHERE e.name IN events OR any(a IN coalesce(e.alias, []) WHERE a IN events)

    // 2) 子图扩展（Person 出发 1-2 hop）
    OPTIONAL MATCH (p)-[r1]->(x)
      WHERE type(r1) IN $rel_whitelist

    OPTIONAL MATCH (x)-[r2]->(y)
      WHERE $max_hops >= 2 AND type(r2) IN $rel_whitelist

    // 3) 证据：任一节点如果能找到 Event->Chunk 证据就带上
    OPTIONAL MATCH (e)-[:SUPPORTED_BY]->(c1:Chunk)
    OPTIONAL MATCH (x)-[:SUPPORTED_BY]->(c2:Chunk)
    OPTIONAL MATCH (y)-[:SUPPORTED_BY]->(c3:Chunk)

    // 4) 关键词兜底：对 Chunk 做 contains（PoC 简单版本）
    //    如果 keywords 为空，用 fallback_keywords（persons/events）补一下
    OPTIONAL MATCH (ck:Chunk)
      WHERE (
        (size(keywords) > 0 AND any(k IN keywords WHERE ck.text CONTAINS k))
        OR
        (size(keywords) = 0 AND size(fallback_keywords) > 0 AND any(k IN fallback_keywords WHERE ck.text CONTAINS k))
      )

    WITH
      collect(DISTINCT {from: coalesce(p.name,""), rel: type(r1), to: coalesce(x.name,"")}) +
      collect(DISTINCT {from: coalesce(x.name,""), rel: type(r2), to: coalesce(y.name,"")}) AS edges,
      collect(DISTINCT c1) + collect(DISTINCT c2) + collect(DISTINCT c3) + collect(DISTINCT ck) AS chunks

    UNWIND chunks AS c
    WITH edges, c
    WHERE c IS NOT NULL

    RETURN
      edges[0..80] AS edges,
      collect(DISTINCT {chunk_id: c.chunk_id, chapter_id: c.chapter_id, text: c.text})[0..$top_k] AS chunks
    """

    async with driver.session(database=settings.NEO4J_DATABASE) as session:
        res = await session.run(
            cypher,
            persons=persons,
            events=events,
            keywords=keywords,
            fallback_keywords=fallback_keywords,
            top_k=top_k_chunks,
            max_hops=max_hops,
            rel_whitelist=REL_WHITELIST,
        )
        row = await res.single()

    if not row:
        return {"edges": [], "chunks": []}

    edges = [e for e in (row.get("edges") or []) if e.get("rel")]
    chunks = row.get("chunks") or []

    # chunk 去重（按 chunk_id）
    seen_chunk_ids: set[str] = set()
    dedup_chunks: list[dict[str, Any]] = []
    for c in chunks:
        cid = c.get("chunk_id")
        if not cid or cid in seen_chunk_ids:
            continue
        seen_chunk_ids.add(cid)
        dedup_chunks.append(c)

    return {"edges": edges, "chunks": dedup_chunks}


def build_context(retrieved: dict[str, Any]) -> str:
    edges = retrieved.get("edges", []) or []
    chunks = retrieved.get("chunks", []) or []

    fact_lines: list[str] = []
    for e in edges[:80]:
        frm, rel, to = e.get("from"), e.get("rel"), e.get("to")
        if frm and rel and to:
            fact_lines.append(f"- ({frm}) -[{rel}]-> ({to})")

    chunk_blocks: list[str] = []
    for c in chunks:
        chunk_blocks.append(
            f"[chunk_id={c.get('chunk_id','')}, chapter_id={c.get('chapter_id','')}] {c.get('text','')}"
        )

    return (
        "你将基于“子图事实”和“证据片段”回答用户问题。不得编造；若证据不足就说明不足。\n\n"
        "子图事实：\n"
        + ("\n".join(fact_lines) if fact_lines else "(无)\n")
        + "\n\n证据片段：\n"
        + ("\n\n".join(chunk_blocks) if chunk_blocks else "(无)\n")
    )
