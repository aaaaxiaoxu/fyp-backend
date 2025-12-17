# src/normalize_entities.py
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional


# =========================
# PoC 过滤：泛称人物（不当作 Person 节点）
# =========================
GENERIC_PERSON = {
    "学生", "学生们", "走读生", "值日生", "女生", "男生", "同学", "老师", "班主任", "主任",
    "父亲", "母亲", "祖母", "奶奶", "爷爷", "大哥", "二哥", "三哥", "大姐", "二姐", "姐姐", "妹妹",
    "村民", "老百姓", "群众", "干部",
}


def project_root() -> Path:
    # 约定：本文件在 <root>/src/normalize_entities.py
    return Path(__file__).resolve().parents[1]


def _is_generic_person(name: str) -> bool:
    name = name.strip()
    if not name:
        return True
    if name in GENERIC_PERSON:
        return True
    if name.endswith("们"):
        return True
    # 描述性短语：PoC 先过滤（避免把“金波他父亲”等当节点）
    if "他父亲" in name or "她父亲" in name or "他妈" in name or "她妈" in name:
        return True
    return False


def _sniff_format(path: Path) -> str:
    """
    返回: "jsonl" 或 "json"
    - jsonl: 每行一个 JSON object
    - json:  整体是一个 JSON list / dict
    """
    with path.open("rb") as f:
        head = f.read(4096)
    s = head.decode("utf-8", errors="ignore").lstrip()
    if s.startswith("[") or s.startswith("{") and "\n" not in s[:200]:
        # 这里稍保守：如果是 '[' 基本就是 JSON；如果是 '{' 但前200字符内没有换行，可能是单个 JSON
        # 但 entities_raw 我们通常是 jsonl，所以后续读失败会自动 fallback
        if s.startswith("["):
            return "json"
    # 默认按 jsonl 读
    return "jsonl"


def read_records(path: Path) -> list[dict[str, Any]]:
    fmt = _sniff_format(path)

    if fmt == "json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
        raise ValueError(f"Unsupported JSON top-level type: {type(obj)}")

    # fmt == "jsonl"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    # 如果 jsonl 误判（例如整个文件是一行 JSON list），做一次 fallback
    if len(rows) == 1 and isinstance(rows[0], list):
        # 极少见：一行里是 list
        return [x for x in rows[0] if isinstance(x, dict)]
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def merge_attributes(attr_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """
    简单合并：同 key+value 去重，保留第一条 evidence
    """
    seen: set[tuple[str, str]] = set()
    merged: list[dict[str, Any]] = []
    for attrs in attr_lists:
        for a in attrs or []:
            key = a.get("key")
            val = a.get("value")
            ev = a.get("evidence")
            if not key or val is None:
                continue
            k = (str(key), str(val))
            if k in seen:
                continue
            seen.add(k)
            merged.append({"key": str(key), "value": str(val), "evidence": ev})
    return merged


def resolve_entities_raw_path(cli_value: Optional[str]) -> Path:
    root = project_root()
    processed_dir = root / "data" / "processed"

    # 1) 如果用户传了 --entities_raw
    if cli_value:
        p = Path(cli_value)
        if not p.is_absolute():
            p = root / p
        if p.exists():
            return p

        # 自动尝试另一种后缀
        if p.suffix == ".jsonl":
            alt = p.with_suffix(".json")
            if alt.exists():
                return alt
        elif p.suffix == ".json":
            alt = p.with_suffix(".jsonl")
            if alt.exists():
                return alt
        else:
            # 没后缀，尝试补全
            alt1 = p.with_suffix(".jsonl")
            alt2 = p.with_suffix(".json")
            if alt1.exists():
                return alt1
            if alt2.exists():
                return alt2

        raise FileNotFoundError(f"Not found: {p.resolve()}")

    # 2) 未传参数：按默认文件名自动选择存在的那个
    default_jsonl = processed_dir / "entities_raw.jsonl"
    default_json = processed_dir / "entities_raw.json"

    if default_jsonl.exists():
        return default_jsonl
    if default_json.exists():
        return default_json

    # 3) 再兜底：在 data/processed 下找任何 entities_raw.json*
    candidates = sorted(processed_dir.glob("entities_raw.json*"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Cannot locate entities_raw.(jsonl|json) under: {processed_dir.resolve()}\n"
        f"Hint: check your output filename in data/processed/."
    )


def resolve_out_dir(cli_value: Optional[str]) -> Path:
    root = project_root()
    if cli_value:
        p = Path(cli_value)
        if not p.is_absolute():
            p = root / p
        return p
    return root / "data" / "processed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--entities_raw",
        default=None,
        help="Path to entities_raw.jsonl (or .json). If relative, resolved from project root.",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. If relative, resolved from project root. Default: data/processed",
    )
    args = ap.parse_args()

    entities_raw_path = resolve_entities_raw_path(args.entities_raw)
    out_dir = resolve_out_dir(args.out_dir)

    rows = read_records(entities_raw_path)

    # 1) 收集每章的人名集合（过滤泛称）
    chapter_people: dict[str, set[str]] = defaultdict(set)
    chapter_person_all: dict[str, set[str]] = defaultdict(set)

    for r in rows:
        ch = r.get("chapter_id")
        if not ch:
            continue
        for e in r.get("entities", []) or []:
            if e.get("type") != "Person":
                continue
            name = (e.get("name") or "").strip()
            if not name:
                continue
            chapter_person_all[ch].add(name)
            if not _is_generic_person(name):
                chapter_people[ch].add(name)

    # 2) 章内 alias_map（短名 -> 长名，唯一候选才映射）
    alias_map: dict[str, str] = {}
    for ch, people in chapter_people.items():
        all_names = list(people)
        for alias in chapter_person_all[ch]:
            alias = alias.strip()
            if not alias or _is_generic_person(alias):
                continue
            candidates = [n for n in all_names if n.endswith(alias) and len(n) > len(alias)]
            candidates = list(set(candidates))
            if len(candidates) == 1:
                alias_map[alias] = candidates[0]

    def canon_name(name: str) -> str:
        return alias_map.get(name, name)

    # 3) canonical 人物库：canonical -> aliases + merged attributes
    canon_store: dict[str, dict[str, Any]] = {}
    attrs_bucket: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    alias_bucket: dict[str, set[str]] = defaultdict(set)

    for r in rows:
        for e in r.get("entities", []) or []:
            if e.get("type") != "Person":
                continue
            name = (e.get("name") or "").strip()
            if not name or _is_generic_person(name):
                continue
            cn = canon_name(name)
            alias_bucket[cn].add(name)
            attrs_bucket[cn].append(e.get("attributes") or [])

    for cn in sorted(alias_bucket.keys()):
        canon_store[cn] = {
            "name": cn,
            "aliases": sorted(alias_bucket[cn]),
            "attributes": merge_attributes(attrs_bucket[cn]),
        }

    # 4) 输出 clean 版 entities：把每个 chunk 的 Person name 映射成 canonical，并丢弃泛称 Person
    clean_rows: list[dict[str, Any]] = []
    for r in rows:
        new_r = dict(r)
        new_entities = []
        for e in r.get("entities", []) or []:
            if e.get("type") == "Person":
                name = (e.get("name") or "").strip()
                if not name or _is_generic_person(name):
                    continue
                e = dict(e)
                e["name"] = canon_name(name)
            new_entities.append(e)
        new_r["entities"] = new_entities
        clean_rows.append(new_r)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "alias_map.json", alias_map)
    write_json(out_dir / "entities_canon.json", canon_store)
    write_jsonl(out_dir / "entities_clean.jsonl", clean_rows)

    print(f"[normalize_entities] input={entities_raw_path.resolve()}")
    print(f"[normalize_entities] out_dir={out_dir.resolve()}")
    print(f"[normalize_entities] alias_map={len(alias_map)} canon_people={len(canon_store)}")
    if alias_map:
        sample = list(alias_map.items())[:10]
        print("[alias_map_sample]", sample)


if __name__ == "__main__":
    main()
