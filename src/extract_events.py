from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from io_utils import read_jsonl, append_jsonl, load_done_ids
from llm_client import DeepSeekClient


SYSTEM_PROMPT = "你是一个严格的结构化信息抽取器。必须只输出 JSON。禁止输出多余文字。"


def _uniq_keep_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _load_entity_candidates(entities_path: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Load entities_raw.jsonl and build:
      chunk_id -> { "persons": [...], "places": [...], "orgs": [...] }
    """
    idx: Dict[str, Dict[str, List[str]]] = {}

    for row in read_jsonl(entities_path):
        if not isinstance(row, dict):
            continue
        cid = row.get("chunk_id")
        if not isinstance(cid, str) or not cid:
            continue

        ents = row.get("entities", [])
        if not isinstance(ents, list):
            ents = []

        persons: List[str] = []
        places: List[str] = []
        orgs: List[str] = []

        for e in ents:
            if not isinstance(e, dict):
                continue
            t = e.get("type")
            name = e.get("name")
            if not isinstance(t, str) or not isinstance(name, str) or not name.strip():
                continue

            name = name.strip()
            if t == "Person":
                persons.append(name)
            elif t == "Place":
                places.append(name)
            elif t == "Org":
                orgs.append(name)

        idx[cid] = {
            "persons": _uniq_keep_order(persons),
            "places": _uniq_keep_order(places),
            "orgs": _uniq_keep_order(orgs),
        }

    return idx


def _render(
    prompt_template: str,
    chunk: dict,
    candidates: Optional[Dict[str, List[str]]] = None,
) -> str:
    filled = (
        prompt_template
        .replace("{chunk_id}", chunk["chunk_id"])
        .replace("{chapter_id}", chunk["chapter_id"])
    )

    # 把候选实体拼到 prompt 里（不要求你改 prompts/events.prompt.txt）
    cand_block = ""
    if candidates is not None:
        persons = candidates.get("persons", [])
        places = candidates.get("places", [])
        orgs = candidates.get("orgs", [])

        # 为避免 prompt 过长：这里不强制截断，你也可以改成 [:30]
        persons_json = json.dumps(persons, ensure_ascii=False)
        places_json = json.dumps(places, ensure_ascii=False)
        orgs_json = json.dumps(orgs, ensure_ascii=False)

        cand_block = (
            "\n\n【实体候选（来自 entities_raw，可为空）】\n"
            f"- persons: {persons_json}\n"
            f"- places: {places_json}\n"
            f"- orgs: {orgs_json}\n"
            "\n【使用规则】\n"
            "- participants[].name 优先从 persons 中选；如果原文没有明确姓名，则 name 必须为 null，并把原文称呼写入 mention。\n"
            "- place.name 优先从 places/orgs 中选；如果原文没有明确地点，则 place 必须为 null。\n"
            "- 禁止编造不在原文中的人名/地名。\n"
        )

    return f'{filled}{cand_block}\n\n"""' + chunk["text"] + '"""'


def _sanitize(obj: Dict[str, Any], chunk: Dict[str, Any]) -> Dict[str, Any]:
    obj["chunk_id"] = chunk["chunk_id"]
    obj["chapter_id"] = chunk["chapter_id"]

    evs = obj.get("events")
    if not isinstance(evs, list):
        evs = []

    cleaned: List[Dict[str, Any]] = []
    for e in evs:
        if not isinstance(e, dict):
            continue
        cleaned.append(e)

    obj["events"] = cleaned
    # 版本号建议改一下，方便你回归对比
    obj["extractor_version"] = "events_v2_deepseek_with_entity_candidates"
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--out", default="data/processed/events_raw.jsonl")
    ap.add_argument("--prompt", default="prompts/events.prompt.txt")

    # 新增：实体候选输入（可选）
    ap.add_argument("--entities", default="")  # e.g. data/processed/entities_raw.jsonl

    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    chunks_path = Path(args.chunks)
    out_path = Path(args.out)
    prompt_path = Path(args.prompt)

    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks file not found: {chunks_path}")

    prompt_template = prompt_path.read_text(encoding="utf-8")

    # 可选加载实体候选
    entity_index: Optional[Dict[str, Dict[str, List[str]]]] = None
    if args.entities:
        entities_path = Path(args.entities)
        if not entities_path.exists():
            raise FileNotFoundError(f"entities file not found: {entities_path}")
        entity_index = _load_entity_candidates(entities_path)

    done = load_done_ids(out_path, "chunk_id") if args.resume else set()

    client = DeepSeekClient()

    processed = 0
    for chunk in read_jsonl(chunks_path):
        if args.limit and processed >= args.limit:
            break
        cid = chunk.get("chunk_id")
        if not isinstance(cid, str) or not cid:
            continue
        if cid in done:
            continue

        candidates = entity_index.get(cid) if entity_index is not None else None
        user = _render(prompt_template, chunk, candidates=candidates)

        obj = client.chat_json(system=SYSTEM_PROMPT, user=user)
        if not isinstance(obj, dict):
            obj = {"events": []}

        obj = _sanitize(obj, chunk)
        append_jsonl(out_path, obj)

        processed += 1
        if processed % 10 == 0:
            print(f"[events] processed={processed} last={cid}")

    print(f"[events] DONE total_processed={processed} out={out_path.resolve()}")


if __name__ == "__main__":
    main()
