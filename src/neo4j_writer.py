from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from neo4j import GraphDatabase


# ----------------------------
# Utils
# ----------------------------
def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def project_root() -> Path:
    # src/neo4j_writer.py -> project root = parents[1]
    return Path(__file__).resolve().parents[1]


# ----------------------------
# Participant naming rules
# ----------------------------
# 代词：永远不建节点
PRONOUNS = {"他", "她", "他们", "她们", "我们", "你", "你们", "众人", "大家"}

# 很常见的群体名/职务泛称，你可以按需要继续扩充
GENERIC_PEOPLE = {"学生们", "值日生", "走读生", "男男女女", "女生", "青年人", "瘦高个的青年人", "跛女子"}

# 泛称参与者白名单（推荐保守：避免把“路人甲/某人/他”这类污染进图）
# 你可以逐步扩充，或者把白名单逻辑关掉（见下方注释）
GENERIC_PARTICIPANTS = {
    "值日生",
    "走读生们",
    "男男女女",
    "众人",
    "大家",
    "一位女生",
    "学生们",
    "同学们",
    "高一（1）班的值日生",
    "瘦高个的青年人",
}


def normalize_person_name(name: Optional[str], mention: Optional[str]) -> Optional[str]:
    """
    事件 participants 里：优先用 name，否则用 mention。
    对明显代词返回 None（跳过）。
    """
    cand = (name or "").strip() or (mention or "").strip()
    if not cand:
        return None
    if cand in PRONOUNS:
        return None
    return cand


def normalize_participant_mention(mention: Optional[str]) -> Optional[str]:
    """
    泛称参与者：只使用 mention，不接受代词。
    """
    if not isinstance(mention, str):
        return None
    m = mention.strip()
    if not m:
        return None
    if m in PRONOUNS:
        return None
    return m


# ----------------------------
# Neo4j Writer
# ----------------------------
class Neo4jWriter:
    def __init__(self, uri: str, user: str, password: str, database: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self) -> None:
        self.driver.close()

    def write(self, func, *args, **kwargs):
        with self.driver.session(database=self.database) as session:
            return session.execute_write(func, *args, **kwargs)


# ----------------------------
# Schema / Constraints
# ----------------------------
CONSTRAINTS = [
    "CREATE CONSTRAINT book_title IF NOT EXISTS FOR (b:Book) REQUIRE b.title IS UNIQUE",
    "CREATE CONSTRAINT chapter_id IF NOT EXISTS FOR (c:Chapter) REQUIRE c.chapter_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT place_name IF NOT EXISTS FOR (p:Place) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT org_name IF NOT EXISTS FOR (o:Org) REQUIRE o.name IS UNIQUE",
    "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
    # 新增：Participant 唯一键
    "CREATE CONSTRAINT participant_key IF NOT EXISTS FOR (p:Participant) REQUIRE p.key IS UNIQUE",
]


def tx_create_constraints(tx) -> None:
    for c in CONSTRAINTS:
        tx.run(c)


# ----------------------------
# Ingest: Book / Chapter / Chunk
# ----------------------------
def tx_ingest_chunks(tx, book_title: str, rows: List[Dict[str, Any]]) -> None:
    cypher = """
    MERGE (b:Book {title: $book_title})
    WITH b
    UNWIND $rows AS row
      MERGE (ch:Chapter {chapter_id: row.chapter_id})
        SET ch.title = coalesce(ch.title, row.chapter_title)
      MERGE (b)-[:HAS_CHAPTER]->(ch)
      MERGE (ck:Chunk {chunk_id: row.chunk_id})
        SET ck.chapter_id = row.chapter_id,
            ck.book_title  = $book_title,
            ck.text        = row.text,
            ck.start_char  = row.start_char,
            ck.end_char    = row.end_char
      MERGE (ch)-[:HAS_CHUNK]->(ck)
    """
    tx.run(cypher, book_title=book_title, rows=rows)


# ----------------------------
# Ingest: Entities
# ----------------------------
def tx_ingest_people_and_mentions(tx, rows: List[Dict[str, Any]]) -> None:
    """
    rows:
      {name, chapter_id, chunk_id, evidence, occupation, status, hometown, traits, is_generic}
    """
    cypher = """
    UNWIND $rows AS row
      MERGE (p:Person {name: row.name})
        SET p.is_generic = coalesce(p.is_generic, row.is_generic),
            p.first_seen_chunk = coalesce(p.first_seen_chunk, row.chunk_id),
            p.first_seen_chapter = coalesce(p.first_seen_chapter, row.chapter_id)

      // 只在属性为空时填入（避免反复覆盖；更复杂的合并可在 normalize 阶段做）
      SET p.occupation = coalesce(p.occupation, row.occupation),
          p.status     = coalesce(p.status, row.status),
          p.hometown   = coalesce(p.hometown, row.hometown),
          p.traits     = coalesce(p.traits, row.traits)

      WITH p, row
      MATCH (ck:Chunk {chunk_id: row.chunk_id})
      MERGE (p)-[m:MENTIONED_IN {chunk_id: row.chunk_id}]->(ck)
        SET m.evidence = coalesce(m.evidence, row.evidence)
    """
    tx.run(cypher, rows=rows)


def tx_ingest_places(tx, rows: List[Dict[str, Any]]) -> None:
    cypher = """
    UNWIND $rows AS row
      MERGE (p:Place {name: row.name})
        SET p.first_seen_chunk = coalesce(p.first_seen_chunk, row.chunk_id),
            p.first_seen_chapter = coalesce(p.first_seen_chapter, row.chapter_id)
      WITH p, row
      MATCH (ck:Chunk {chunk_id: row.chunk_id})
      MERGE (p)-[m:MENTIONED_IN {chunk_id: row.chunk_id}]->(ck)
        SET m.evidence = coalesce(m.evidence, row.evidence)
    """
    tx.run(cypher, rows=rows)


def tx_ingest_orgs(tx, rows: List[Dict[str, Any]]) -> None:
    cypher = """
    UNWIND $rows AS row
      MERGE (o:Org {name: row.name})
        SET o.first_seen_chunk = coalesce(o.first_seen_chunk, row.chunk_id),
            o.first_seen_chapter = coalesce(o.first_seen_chapter, row.chapter_id)
      WITH o, row
      MATCH (ck:Chunk {chunk_id: row.chunk_id})
      MERGE (o)-[m:MENTIONED_IN {chunk_id: row.chunk_id}]->(ck)
        SET m.evidence = coalesce(m.evidence, row.evidence)
    """
    tx.run(cypher, rows=rows)


# ----------------------------
# Ingest: Events
# ----------------------------
def tx_ingest_events(tx, rows: List[Dict[str, Any]]) -> None:
    """
    rows:
      {event_id, chunk_id, chapter_id, event_type, summary, trigger, salience}
    """
    cypher = """
    UNWIND $rows AS row
      MERGE (e:Event {event_id: row.event_id})
        SET e.chunk_id   = row.chunk_id,
            e.chapter_id = row.chapter_id,
            e.event_type = row.event_type,
            e.summary    = row.summary,
            e.trigger    = row.trigger,
            e.salience   = row.salience
      WITH e, row
      MATCH (ck:Chunk {chunk_id: row.chunk_id})
      MERGE (e)-[s:SUPPORTED_BY {chunk_id: row.chunk_id}]->(ck)
        SET s.evidence = coalesce(s.evidence, row.trigger)
    """
    tx.run(cypher, rows=rows)


def tx_ingest_event_places(tx, rows: List[Dict[str, Any]]) -> None:
    """
    rows:
      {event_id, place_name, chunk_id, evidence}
    """
    cypher = """
    UNWIND $rows AS row
      MATCH (e:Event {event_id: row.event_id})
      MERGE (p:Place {name: row.place_name})
      MERGE (e)-[r:HAPPENS_AT {chunk_id: row.chunk_id}]->(p)
        SET r.evidence = coalesce(r.evidence, row.evidence)
    """
    tx.run(cypher, rows=rows)


def tx_ingest_event_participants(tx, rows: List[Dict[str, Any]]) -> None:
    """
    rows:
      {event_id, person_name, chunk_id, role, evidence}
    """
    cypher = """
    UNWIND $rows AS row
      MATCH (e:Event {event_id: row.event_id})
      MERGE (p:Person {name: row.person_name})
      MERGE (p)-[r:PARTICIPATES_IN {event_id: row.event_id, chunk_id: row.chunk_id}]->(e)
        SET r.role     = coalesce(r.role, row.role),
            r.evidence = coalesce(r.evidence, row.evidence)
    """
    tx.run(cypher, rows=rows)


def tx_ingest_event_participants_generic(tx, rows: List[Dict[str, Any]]) -> None:
    """
    rows:
      {event_id, participant_key, mention, chunk_id, role, evidence}
    """
    cypher = """
    UNWIND $rows AS row
      MATCH (e:Event {event_id: row.event_id})
      MERGE (p:Participant {key: row.participant_key})
        SET p.mention = row.mention
      MERGE (p)-[r:PARTICIPATES_IN {event_id: row.event_id, chunk_id: row.chunk_id}]->(e)
        SET r.role     = coalesce(r.role, row.role),
            r.evidence = coalesce(r.evidence, row.evidence)
    """
    tx.run(cypher, rows=rows)


# ----------------------------
# Ingest: Relations (dynamic rel type)
# ----------------------------
ALLOWED_REL_TYPES = {
    "FAMILY_OF", "FRIEND_OF", "LOVES", "MARRIED_TO", "WORKS_FOR", "STUDIES_AT"
}


def tx_ingest_relations_of_type(tx, rel_type: str, rows: List[Dict[str, Any]]) -> None:
    # Neo4j 关系类型不能参数化，只能拼字符串；所以一定要白名单校验
    if rel_type not in ALLOWED_REL_TYPES:
        raise ValueError(f"Unsupported relation type: {rel_type}")

    cypher = f"""
    UNWIND $rows AS row
      MERGE (h:Person {{name: row.head}})
      MERGE (t:Person {{name: row.tail}})
      MERGE (h)-[r:{rel_type} {{chunk_id: row.chunk_id, evidence: row.evidence}}]->(t)
        SET r.confidence = row.confidence,
            r.family_role = row.family_role
    """
    tx.run(cypher, rows=rows)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--entities", default="data/processed/entities_clean.jsonl")
    ap.add_argument("--events", default="data/processed/events_raw.jsonl")
    ap.add_argument("--relations", default="data/processed/relations_raw.jsonl")
    ap.add_argument("--book_title", default="平凡的世界")
    ap.add_argument("--batch", type=int, default=300)

    # Neo4j conn (env override)
    ap.add_argument("--neo4j_uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    ap.add_argument("--neo4j_user", default=os.getenv("NEO4J_USER", "neo4j"))
    ap.add_argument("--neo4j_password", default=os.getenv("NEO4J_PASSWORD", "neo4j"))
    ap.add_argument("--neo4j_db", default=os.getenv("NEO4J_DB", "neo4j"))

    args = ap.parse_args()

    root = project_root()
    chunks_path = (root / args.chunks).resolve()
    entities_path = (root / args.entities).resolve()
    events_path = (root / args.events).resolve()
    relations_path = (root / args.relations).resolve()

    writer = Neo4jWriter(args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.neo4j_db)

    # 1) Constraints
    writer.write(tx_create_constraints)
    print("[neo4j] constraints ensured")

    # 2) Book/Chapter/Chunk
    chunk_rows: List[Dict[str, Any]] = []
    for obj in iter_jsonl(chunks_path):
        # 基本字段校验
        if not isinstance(obj.get("chunk_id"), str) or not isinstance(obj.get("chapter_id"), str):
            continue
        if not isinstance(obj.get("text"), str):
            continue

        chunk_rows.append(
            {
                "chapter_id": obj["chapter_id"],
                "chapter_title": obj.get("chapter_title"),
                "chunk_id": obj["chunk_id"],
                "text": obj["text"],
                "start_char": obj.get("start_char", 0),
                "end_char": obj.get("end_char", 0),
            }
        )
        if len(chunk_rows) >= args.batch:
            writer.write(tx_ingest_chunks, args.book_title, chunk_rows)
            chunk_rows = []
    if chunk_rows:
        writer.write(tx_ingest_chunks, args.book_title, chunk_rows)
    print("[neo4j] chunks ingested")

    # 3) Entities
    people_rows: List[Dict[str, Any]] = []
    place_rows: List[Dict[str, Any]] = []
    org_rows: List[Dict[str, Any]] = []

    def pick_attr(attrs: Any, key: str) -> Optional[str]:
        if not isinstance(attrs, list):
            return None
        for a in attrs:
            if isinstance(a, dict) and a.get("key") == key and isinstance(a.get("value"), str):
                return a["value"]
        return None

    for obj in iter_jsonl(entities_path):
        cid = obj.get("chunk_id")
        chap = obj.get("chapter_id")
        if not isinstance(cid, str) or not isinstance(chap, str):
            continue

        ents = obj.get("entities", [])
        if not isinstance(ents, list):
            continue

        for e in ents:
            if not isinstance(e, dict):
                continue
            et = e.get("type")
            name = e.get("name")
            evidence = e.get("evidence")
            attrs = e.get("attributes", [])

            if not isinstance(name, str) or not name.strip():
                continue

            row_base = {
                "name": name.strip(),
                "chunk_id": cid,
                "chapter_id": chap,
                "evidence": evidence if isinstance(evidence, str) else None,
            }

            if et == "Person":
                people_rows.append(
                    {
                        **row_base,
                        "occupation": pick_attr(attrs, "occupation"),
                        "status": pick_attr(attrs, "status"),
                        "hometown": pick_attr(attrs, "hometown"),
                        "traits": pick_attr(attrs, "traits"),
                        "is_generic": (name.strip() in GENERIC_PEOPLE),
                    }
                )
            elif et == "Place":
                place_rows.append(row_base)
            elif et == "Org":
                org_rows.append(row_base)

            if len(people_rows) >= args.batch:
                writer.write(tx_ingest_people_and_mentions, people_rows)
                people_rows = []
            if len(place_rows) >= args.batch:
                writer.write(tx_ingest_places, place_rows)
                place_rows = []
            if len(org_rows) >= args.batch:
                writer.write(tx_ingest_orgs, org_rows)
                org_rows = []

    if people_rows:
        writer.write(tx_ingest_people_and_mentions, people_rows)
    if place_rows:
        writer.write(tx_ingest_places, place_rows)
    if org_rows:
        writer.write(tx_ingest_orgs, org_rows)

    print("[neo4j] entities ingested")

    # 4) Events
    ev_rows: List[Dict[str, Any]] = []
    ev_place_rows: List[Dict[str, Any]] = []
    ev_part_rows: List[Dict[str, Any]] = []
    ev_part_generic_rows: List[Dict[str, Any]] = []

    for obj in iter_jsonl(events_path):
        cid = obj.get("chunk_id")
        chap = obj.get("chapter_id")
        if not isinstance(cid, str) or not isinstance(chap, str):
            continue

        events = obj.get("events", [])
        if not isinstance(events, list):
            continue

        for idx, e in enumerate(events):
            if not isinstance(e, dict):
                continue

            event_id = f"{cid}__{idx:03d}"
            event_type = e.get("event_type")
            summary = e.get("summary")
            trigger = e.get("trigger")
            salience = e.get("salience", 1)

            if not isinstance(event_type, str) or not isinstance(summary, str) or not isinstance(trigger, str):
                continue

            ev_rows.append(
                {
                    "event_id": event_id,
                    "chunk_id": cid,
                    "chapter_id": chap,
                    "event_type": event_type,
                    "summary": summary,
                    "trigger": trigger,
                    "salience": int(salience) if isinstance(salience, int) else 1,
                }
            )

            # place
            place = e.get("place")
            if isinstance(place, dict):
                pn = place.get("name")
                pe = place.get("evidence")
                if isinstance(pn, str) and pn.strip():
                    ev_place_rows.append(
                        {
                            "event_id": event_id,
                            "place_name": pn.strip(),
                            "chunk_id": cid,
                            "evidence": pe if isinstance(pe, str) else None,
                        }
                    )

            # participants
            participants = e.get("participants", [])
            if isinstance(participants, list):
                for p in participants:
                    if not isinstance(p, dict):
                        continue

                    # 1) 具名 -> Person
                    person_name = normalize_person_name(p.get("name"), p.get("mention"))
                    if person_name and person_name not in PRONOUNS:
                        ev_part_rows.append(
                            {
                                "event_id": event_id,
                                "person_name": person_name,
                                "chunk_id": cid,
                                "role": p.get("role") if isinstance(p.get("role"), str) else None,
                                "evidence": p.get("evidence") if isinstance(p.get("evidence"), str) else None,
                            }
                        )
                        continue

                    # 2) 非具名 -> Participant（只处理 mention 且非代词）
                    mention = normalize_participant_mention(p.get("mention"))
                    if not mention:
                        continue

                    # 默认：白名单控制污染（推荐）
                    if mention not in GENERIC_PARTICIPANTS:
                        continue

                    # 如果你想“只要不是代词就建 Participant”，把上面两行替换为：
                    # pass

                    participant_key = mention  # 简单起见：全局唯一 key = mention
                    ev_part_generic_rows.append(
                        {
                            "event_id": event_id,
                            "participant_key": participant_key,
                            "mention": mention,
                            "chunk_id": cid,
                            "role": p.get("role") if isinstance(p.get("role"), str) else None,
                            "evidence": p.get("evidence") if isinstance(p.get("evidence"), str) else None,
                        }
                    )

            # batch flush
            if len(ev_rows) >= args.batch:
                writer.write(tx_ingest_events, ev_rows)
                ev_rows = []
            if len(ev_place_rows) >= args.batch:
                writer.write(tx_ingest_event_places, ev_place_rows)
                ev_place_rows = []
            if len(ev_part_rows) >= args.batch:
                writer.write(tx_ingest_event_participants, ev_part_rows)
                ev_part_rows = []
            if len(ev_part_generic_rows) >= args.batch:
                writer.write(tx_ingest_event_participants_generic, ev_part_generic_rows)
                ev_part_generic_rows = []

    if ev_rows:
        writer.write(tx_ingest_events, ev_rows)
    if ev_place_rows:
        writer.write(tx_ingest_event_places, ev_place_rows)
    if ev_part_rows:
        writer.write(tx_ingest_event_participants, ev_part_rows)
    if ev_part_generic_rows:
        writer.write(tx_ingest_event_participants_generic, ev_part_generic_rows)

    print("[neo4j] events ingested")

    # 5) Relations
    rel_buckets: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ALLOWED_REL_TYPES}

    for obj in iter_jsonl(relations_path):
        cid = obj.get("chunk_id")
        if not isinstance(cid, str):
            continue

        rels = obj.get("relations", [])
        if not isinstance(rels, list) or not rels:
            continue  # 空的就跳过（你问的“要不要去掉空的”：这里已经等价处理）

        for r in rels:
            if not isinstance(r, dict):
                continue
            t = r.get("type")
            head = r.get("head")
            tail = r.get("tail")
            evidence = r.get("evidence")
            confidence = r.get("confidence", None)
            meta = r.get("meta", {})

            if t not in ALLOWED_REL_TYPES:
                continue
            if not isinstance(head, str) or not isinstance(tail, str) or not isinstance(evidence, str):
                continue

            family_role = None
            if isinstance(meta, dict) and isinstance(meta.get("family_role"), str):
                family_role = meta.get("family_role")

            rel_buckets[t].append(
                {
                    "head": head.strip(),
                    "tail": tail.strip(),
                    "chunk_id": cid,
                    "evidence": evidence,
                    "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
                    "family_role": family_role,
                }
            )

            if len(rel_buckets[t]) >= args.batch:
                writer.write(tx_ingest_relations_of_type, t, rel_buckets[t])
                rel_buckets[t] = []

    for t, rows in rel_buckets.items():
        if rows:
            writer.write(tx_ingest_relations_of_type, t, rows)

    print("[neo4j] relations ingested")

    writer.close()
    print("[neo4j] DONE")


if __name__ == "__main__":
    main()
