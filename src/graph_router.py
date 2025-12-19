from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from neo4j import AsyncDriver

from .neo4j_client import get_neo4j_driver
from .settings import settings
from .graph_schemas import (
    CatalogResponse,
    EntityListItem,
    EntityListResponse,
    EvidenceItem,
    EvidenceResponse,
    GraphResponse,
    SubgraphRequest,
)

router = APIRouter(prefix="/graph", tags=["Graph"])

ALLOWED_LABELS = {"Person", "Event"}


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _validate_eid(eid: str, field_name: str = "eid") -> None:
    # elementId 形如 "4:xxxx:123"，一定包含冒号
    if not eid or ":" not in eid:
        raise HTTPException(status_code=400, detail=f"{field_name} must be a valid Neo4j elementId")


# =========================
# Catalog / List / Search
# =========================

@router.get("/catalog", response_model=CatalogResponse)
async def get_catalog(driver: AsyncDriver = Depends(get_neo4j_driver)) -> CatalogResponse:
    """
    左侧目录总数：Person / Event
    """
    cypher = """
    CALL { MATCH (p:Person) RETURN count(p) AS person_count }
    CALL { MATCH (e:Event)  RETURN count(e) AS event_count }
    RETURN person_count, event_count
    """
    async with driver.session(database=settings.NEO4J_DATABASE) as session:
        rec = await (await session.run(cypher)).single()

    return CatalogResponse(
        person_count=int((rec and rec["person_count"]) or 0),
        event_count=int((rec and rec["event_count"]) or 0),
    )


@router.get("/search/person")
async def search_person(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    driver: AsyncDriver = Depends(get_neo4j_driver),
) -> list[dict]:
    """
    给前端搜索人物起点：返回 elementId + name
    """
    cypher = """
    MATCH (p:Person)
    WHERE p.name CONTAINS $q
    RETURN elementId(p) AS eid, p.name AS name
    ORDER BY name ASC
    LIMIT $limit
    """
    async with driver.session(database=settings.NEO4J_DATABASE) as session:
        rows = await (await session.run(cypher, q=q, limit=limit)).data()

    return [{"eid": r["eid"], "name": r.get("name")} for r in rows]


@router.get("/entities", response_model=EntityListResponse)
async def list_entities(
    label: str = Query(..., description="Person or Event"),
    q: str = Query("", description="keyword"),
    sort: str = Query("score", description="score|name|first_seen"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    driver: AsyncDriver = Depends(get_neo4j_driver),
) -> EntityListResponse:
    """
    左侧列表：分页列出 Person/Event
    score 默认用“提及次数/证据次数”实时计算（PoC 够用，后期可预计算）
    """
    if label not in ALLOWED_LABELS:
        raise HTTPException(status_code=400, detail="label must be Person or Event")
    if sort not in {"score", "name", "first_seen"}:
        raise HTTPException(status_code=400, detail="sort must be score|name|first_seen")

    q = q or ""

    if label == "Person":
        total_cypher = """
        MATCH (n:Person)
        WHERE $q = '' OR n.name CONTAINS $q
        RETURN count(n) AS total
        """
        base_cypher = """
        MATCH (n:Person)
        WHERE $q = '' OR n.name CONTAINS $q
        WITH n, count { (n)-[:MENTIONED_IN]->(:Chunk) } AS score
        RETURN
          elementId(n) AS eid,
          'Person' AS label,
          n.name AS name,
          score AS score,
          n.first_seen_chapter AS first_seen_chapter,
          n.first_seen_chunk AS first_seen_chunk
        """
    else:
        total_cypher = """
        MATCH (n:Event)
        WHERE $q = '' OR coalesce(n.title, n.name) CONTAINS $q
        RETURN count(n) AS total
        """
        base_cypher = """
        MATCH (n:Event)
        WHERE $q = '' OR coalesce(n.title, n.name) CONTAINS $q
        WITH n, count { (n)-[:MENTIONED_IN]->(:Chunk) } AS score
        RETURN
          elementId(n) AS eid,
          'Event' AS label,
          coalesce(n.title, n.name) AS name,
          score AS score,
          n.first_seen_chapter AS first_seen_chapter,
          n.first_seen_chunk AS first_seen_chunk
        """

    if sort == "score":
        order_by = "ORDER BY score DESC, name ASC"
    elif sort == "name":
        order_by = "ORDER BY name ASC"
    else:
        order_by = "ORDER BY first_seen_chapter ASC, first_seen_chunk ASC, name ASC"

    list_cypher = f"{base_cypher}\n{order_by}\nSKIP $offset LIMIT $limit"

    async with driver.session(database=settings.NEO4J_DATABASE) as session:
        total_rec = await (await session.run(total_cypher, q=q)).single()
        total = int((total_rec and total_rec["total"]) or 0)

        rows = await (await session.run(list_cypher, q=q, offset=offset, limit=limit)).data()

    items = [
        EntityListItem(
            eid=row["eid"],
            label=row["label"],
            name=row.get("name"),
            score=int(row.get("score") or 0),
            first_seen_chapter=row.get("first_seen_chapter"),
            first_seen_chunk=row.get("first_seen_chunk"),
        )
        for row in rows
    ]
    return EntityListResponse(items=items, total=total)


# =========================
# Node detail / Evidence
# =========================

@router.get("/node/by-eid/{eid}")
async def get_node_detail(
    eid: str,
    driver: AsyncDriver = Depends(get_neo4j_driver),
) -> dict:
    """
    节点详情：返回 labels + properties
    注意：Chunk 详情不返回 text（全文使用 /chunks/by-eid）
    """
    _validate_eid(eid, "eid")

    cypher = """
    MATCH (n) WHERE elementId(n) = $eid
    RETURN labels(n) AS labels, properties(n) AS props
    """
    async with driver.session(database=settings.NEO4J_DATABASE) as session:
        rec = await (await session.run(cypher, eid=eid)).single()

    if rec is None:
        raise HTTPException(status_code=404, detail="Node not found")

    labels = rec["labels"] or []
    props = rec["props"] or {}

    if "Chunk" in labels and "text" in props:
        props.pop("text", None)

    return {"eid": eid, "labels": labels, "properties": props}


@router.get("/evidence", response_model=EvidenceResponse)
async def get_evidence(
    eid: str = Query(..., description="node elementId"),
    limit: int = Query(20, ge=1, le=200),
    snippet_len: int = Query(120, ge=0, le=settings.GRAPH_MAX_SNIPPET_LEN),
    driver: AsyncDriver = Depends(get_neo4j_driver),
) -> EvidenceResponse:
    """
    证据列表：返回关联 Chunk 的 snippet（不返回全文）
    """
    _validate_eid(eid, "eid")

    cypher = """
    MATCH (n) WHERE elementId(n) = $eid
    MATCH (n)-[:MENTIONED_IN]->(c:Chunk)
    RETURN
      elementId(c) AS chunk_eid,
      c.chunk_id AS chunk_id,
      c.chapter_id AS chapter_id,
      c.start_char AS start_char,
      c.end_char AS end_char,
      CASE
        WHEN c.text IS NOT NULL AND $snippet_len > 0 THEN substring(c.text, 0, $snippet_len)
        ELSE NULL
      END AS snippet
    ORDER BY c.chapter_id ASC, c.start_char ASC
    LIMIT $limit
    """

    async with driver.session(database=settings.NEO4J_DATABASE) as session:
        rows = await (await session.run(cypher, eid=eid, limit=limit, snippet_len=snippet_len)).data()

    items = [
        EvidenceItem(
            chunk_eid=r["chunk_eid"],
            chunk_id=r.get("chunk_id"),
            chapter_id=r.get("chapter_id"),
            start_char=r.get("start_char"),
            end_char=r.get("end_char"),
            snippet=r.get("snippet"),
        )
        for r in rows
    ]
    return EvidenceResponse(items=items)


# =========================
# ✅ Missing endpoints (你缺的两个)
# =========================

@router.post("/subgraph", response_model=GraphResponse)
async def get_subgraph(
    req: SubgraphRequest,
    driver: AsyncDriver = Depends(get_neo4j_driver),
) -> GraphResponse:
    """
    ✅ POST /graph/subgraph
    获取子图：从 seed_eid 出发展开 depth 跳关系，返回 nodes+edges
    - Chunk 节点：只返回部分字段 + 可选 snippet，不返回全文 text
    """
    _validate_eid(req.seed_eid, "seed_eid")

    depth = _clamp(req.depth, 1, settings.GRAPH_MAX_DEPTH)
    limit_paths = _clamp(req.limit_paths, 1, settings.GRAPH_MAX_PATHS)
    snippet_len = _clamp(req.snippet_len, 0, settings.GRAPH_MAX_SNIPPET_LEN)

    if req.direction == "out":
        pattern = f"(seed)-[*1..{depth}]->(m)"
    elif req.direction == "in":
        pattern = f"(seed)<-[*1..{depth}]-(m)"
    else:
        pattern = f"(seed)-[*1..{depth}]-(m)"

    cypher = f"""
    MATCH (seed)
    WHERE elementId(seed) = $seed_eid
    MATCH p={pattern}
    WITH p LIMIT $limit_paths
    WITH collect(p) AS paths

    // nodes
    UNWIND paths AS path
    UNWIND nodes(path) AS node
    WITH collect(DISTINCT node) AS allNodes, paths

    // rels
    UNWIND paths AS path2
    UNWIND relationships(path2) AS rel
    WITH allNodes, collect(DISTINCT rel) AS allRels

    // project nodes (Chunk light)
    UNWIND allNodes AS n
    WITH
      collect(
        CASE
          WHEN n:Chunk THEN {{
            id: toString(elementId(n)),
            labels: labels(n),
            properties: {{
              chunk_id: n.chunk_id,
              book_title: n.book_title,
              chapter_id: n.chapter_id,
              start_char: n.start_char,
              end_char: n.end_char,
              snippet: CASE
                WHEN $include_snippet AND n.text IS NOT NULL AND $snippet_len > 0
                  THEN substring(n.text, 0, $snippet_len)
                ELSE NULL
              END
            }}
          }}
          ELSE {{
            id: toString(elementId(n)),
            labels: labels(n),
            properties: properties(n)
          }}
        END
      ) AS nodes_out,
      allRels

    // project edges
    UNWIND allRels AS r
    WITH nodes_out,
      collect(
        {{
          id: toString(elementId(r)),
          type: type(r),
          source: toString(elementId(startNode(r))),
          target: toString(elementId(endNode(r))),
          properties: properties(r)
        }}
      ) AS edges_out

    RETURN nodes_out AS nodes, edges_out AS edges
    """

    params = {
        "seed_eid": req.seed_eid,
        "limit_paths": limit_paths,
        "include_snippet": req.include_snippet,
        "snippet_len": snippet_len,
    }

    async with driver.session(database=settings.NEO4J_DATABASE) as session:
        record = await (await session.run(cypher, **params)).single()

    if record is None:
        return GraphResponse(nodes=[], edges=[])

    return GraphResponse(nodes=record["nodes"], edges=record["edges"])


@router.get("/chunks/by-eid/{chunk_eid}")
async def get_chunk_by_eid(
    chunk_eid: str,
    driver: AsyncDriver = Depends(get_neo4j_driver),
) -> dict:
    """
    ✅ GET /graph/chunks/by-eid/{chunk_eid}
    获取 Chunk 全文（包含 text）
    """
    _validate_eid(chunk_eid, "chunk_eid")

    cypher = """
    MATCH (c:Chunk)
    WHERE elementId(c) = $chunk_eid
    RETURN {
      eid: elementId(c),
      chunk_id: c.chunk_id,
      book_title: c.book_title,
      chapter_id: c.chapter_id,
      start_char: c.start_char,
      end_char: c.end_char,
      text: c.text
    } AS chunk
    """

    async with driver.session(database=settings.NEO4J_DATABASE) as session:
        record = await (await session.run(cypher, chunk_eid=chunk_eid)).single()

    if record is None or record.get("chunk") is None:
        raise HTTPException(status_code=404, detail="Chunk not found")

    return record["chunk"]
