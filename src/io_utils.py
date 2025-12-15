from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Set


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_done_ids(out_path: Path, id_field: str = "chunk_id") -> Set[str]:
    if not out_path.exists():
        return set()
    done: Set[str] = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cid = obj.get(id_field)
                if isinstance(cid, str) and cid:
                    done.add(cid)
            except Exception:
                continue
    return done
