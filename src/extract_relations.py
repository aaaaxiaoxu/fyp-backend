from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from io_utils import read_jsonl, append_jsonl, load_done_ids
from llm_client import DeepSeekClient


SYSTEM_PROMPT = "你是一个严格的结构化信息抽取器。必须只输出 JSON。禁止输出多余文字。"


def load_people_candidates(entities_raw_path: Optional[Path]) -> Dict[str, List[str]]:
    """
    从 entities_raw.jsonl 构建 chunk_id -> [PersonName...] 映射。
    这能明显降低关系抽取噪声（推荐先跑 entities，再跑 relations）。
    """
    if not entities_raw_path or not entities_raw_path.exists():
        return {}

    m: Dict[str, List[str]] = {}
    with entities_raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("chunk_id")
            ents = obj.get("entities", [])
            if not isinstance(cid, str) or not isinstance(ents, list):
                continue
            people = []
            for e in ents:
                if isinstance(e, dict) and e.get("type") == "Person" and isinstance(e.get("name"), str):
                    people.append(e["name"])
            if people:
                m[cid] = sorted(set(people))
    return m


def _render(prompt_template: str, chunk: dict, candidates: list[str]) -> str:
    filled = (
        prompt_template
        .replace("{chunk_id}", chunk["chunk_id"])
        .replace("{chapter_id}", chunk["chapter_id"])
    )
    cand_text = ", ".join(candidates) if candidates else ""
    return f"""{filled}

candidate_people: {cand_text}

\"\"\"{chunk["text"]}\"\"\"
"""


def _sanitize(obj: Dict[str, Any], chunk: Dict[str, Any]) -> Dict[str, Any]:
    obj["chunk_id"] = chunk["chunk_id"]
    obj["chapter_id"] = chunk["chapter_id"]

    rels = obj.get("relations")
    if not isinstance(rels, list):
        rels = []

    cleaned: List[Dict[str, Any]] = []
    for r in rels:
        if not isinstance(r, dict):
            continue
        cleaned.append(r)

    obj["relations"] = cleaned
    obj["extractor_version"] = "relations_v1_deepseek"
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--out", default="data/processed/relations_raw.jsonl")
    ap.add_argument("--prompt", default="prompts/relations.prompt.txt")
    ap.add_argument("--entities_raw", default="data/processed/entities_raw.jsonl",
                    help="可选：用于给关系抽取提供候选人物名单（强烈推荐先跑实体再跑关系）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    chunks_path = Path(args.chunks)
    out_path = Path(args.out)
    prompt_template = Path(args.prompt).read_text(encoding="utf-8")

    done = load_done_ids(out_path, "chunk_id") if args.resume else set()

    candidates_map = load_people_candidates(Path(args.entities_raw)) if args.entities_raw else {}

    client = DeepSeekClient()

    processed = 0
    for chunk in read_jsonl(chunks_path):
        if args.limit and processed >= args.limit:
            break
        cid = chunk["chunk_id"]
        if cid in done:
            continue

        candidates = candidates_map.get(cid, [])
        user = _render(prompt_template, chunk, candidates)
        obj = client.chat_json(system=SYSTEM_PROMPT, user=user)
        obj = _sanitize(obj, chunk)

        append_jsonl(out_path, obj)
        processed += 1
        if processed % 10 == 0:
            print(f"[relations] processed={processed} last={cid}")

    print(f"[relations] DONE total_processed={processed} out={out_path.resolve()}")


if __name__ == "__main__":
    main()
