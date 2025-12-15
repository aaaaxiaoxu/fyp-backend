from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from io_utils import read_jsonl, append_jsonl, load_done_ids
from llm_client import DeepSeekClient


SYSTEM_PROMPT = "你是一个严格的结构化信息抽取器。必须只输出 JSON。禁止输出多余文字。"


def _render(prompt_template: str, chunk: dict) -> str:
    filled = (
        prompt_template
        .replace("{chunk_id}", chunk["chunk_id"])
        .replace("{chapter_id}", chunk["chapter_id"])
    )
    return f'{filled}\n\n"""' + chunk["text"] + '"""'



def _sanitize(obj: Dict[str, Any], chunk: Dict[str, Any]) -> Dict[str, Any]:
    obj["chunk_id"] = chunk["chunk_id"]
    obj["chapter_id"] = chunk["chapter_id"]

    ents = obj.get("entities")
    if not isinstance(ents, list):
        ents = []

    cleaned: List[Dict[str, Any]] = []
    for e in ents:
        if not isinstance(e, dict):
            continue
        etype = e.get("type")
        name = e.get("name")
        if etype not in {"Person", "Place", "Org"}:
            continue
        if not isinstance(name, str) or not name.strip():
            continue

        evidence = e.get("evidence")
        if not isinstance(evidence, str):
            evidence = ""

        attrs = e.get("attributes")
        if not isinstance(attrs, list):
            attrs = []

        if etype in {"Place", "Org"}:
            attrs = []

        cleaned.append(
            {
                "type": etype,
                "name": name.strip(),
                "evidence": evidence.strip(),
                "attributes": attrs,
            }
        )

    obj["entities"] = cleaned
    obj["extractor_version"] = "entities_v1_deepseek"
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--out", default="data/processed/entities_raw.jsonl")
    ap.add_argument("--prompt", default="prompts/entities.prompt.txt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    chunks_path = Path(args.chunks)
    out_path = Path(args.out)
    prompt_template = Path(args.prompt).read_text(encoding="utf-8")

    done = load_done_ids(out_path, "chunk_id") if args.resume else set()

    client = DeepSeekClient()

    processed = 0
    for chunk in read_jsonl(chunks_path):
        if args.limit and processed >= args.limit:
            break
        cid = chunk["chunk_id"]
        if cid in done:
            continue

        user = _render(prompt_template, chunk)
        obj = client.chat_json(system=SYSTEM_PROMPT, user=user)
        obj = _sanitize(obj, chunk)

        append_jsonl(out_path, obj)
        processed += 1
        if processed % 10 == 0:
            print(f"[entities] processed={processed} last={cid}")

    print(f"[entities] DONE total_processed={processed} out={out_path.resolve()}")


if __name__ == "__main__":
    main()
