#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1: Parse book text into chapters and split into overlapping chunks.

Chapter heading format (multiline):
    第X部 卷Y 第Z章
Example:
    第一部 卷一 第一章

Outputs (JSONL):
- chapters.jsonl: chapter metadata
- chunks.jsonl: chunk records with offsets + text

Run (PoC: first 2 chapters):
    python split.py --input "data/raw/book.txt" --out_dir "data/processed" --max_chapters 2

Run (full book):
    python split.py --input "data/raw/book.txt" --out_dir "data/processed"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ----------------------------
# Chinese numeral to int
# ----------------------------
_DIGIT = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "两": 2}
_UNIT = {"十": 10, "百": 100, "千": 1000}


def cn2int(cn: str) -> int:
    """
    Convert common Chinese numerals up to 9999 into int.
    Handles: 零 一 二 三 四 五 六 七 八 九 十 百 千 两
    """
    cn = cn.strip()
    if not cn:
        raise ValueError("empty Chinese numeral")

    total = 0
    num = 0
    for ch in cn:
        if ch in _DIGIT:
            num = _DIGIT[ch]
        elif ch in _UNIT:
            u = _UNIT[ch]
            if num == 0:
                num = 1  # "十" == 10
            total += num * u
            num = 0
        else:
            # ignore uncommon chars if any
            continue
    total += num
    return total


# ----------------------------
# Chapter parsing
# ----------------------------
CHAPTER_PAT = re.compile(
    r"^\s*第(?P<part>[一二三四五六七八九十百零两]+)部\s+卷(?P<vol>[一二三四五六七八九十百零两]+)\s+第(?P<ch>[一二三四五六七八九十百零两]+)章\s*$",
    re.M,
)


def parse_chapters(book_text: str) -> List[Dict[str, Any]]:
    """
    Split raw book text into chapters by headings.
    Returns chapters ordered by appearance.
    """
    matches = list(CHAPTER_PAT.finditer(book_text))
    if not matches:
        raise ValueError(
            "No chapter headings found. "
            "Please check CHAPTER_PAT against your raw txt headings."
        )

    chapters: List[Dict[str, Any]] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(book_text)
        header = m.group(0).strip()

        part_n = cn2int(m.group("part"))
        vol_n = cn2int(m.group("vol"))
        ch_n = cn2int(m.group("ch"))

        body = book_text[m.end():end].strip("\n")
        chapter_id = f"p{part_n}_v{vol_n}_ch{ch_n:03d}"

        chapters.append(
            {
                "chapter_id": chapter_id,
                "chapter_title": header,
                "part": part_n,
                "volume": vol_n,
                "chapter_no": ch_n,
                "text": body,
            }
        )
    return chapters


# ----------------------------
# Chunking helpers
# ----------------------------
def _find_breakpoint(text: str, start: int, hard_end: int, min_ratio: float = 0.7) -> int:
    """
    Move chunk end backward to a nicer boundary (blank line/newline/punctuation).
    Only searches within [start + min_ratio*len, hard_end].
    """
    if hard_end >= len(text):
        return len(text)

    window_start = start + int((hard_end - start) * min_ratio)
    window = text[window_start:hard_end]

    # prefer blank line boundary
    idx = window.rfind("\n\n")
    if idx != -1:
        return window_start + idx + 2

    # then normal newline
    idx = window.rfind("\n")
    if idx != -1:
        return window_start + idx + 1

    # then punctuation boundary
    for p in ["。", "！", "？", "；", "…", "”", "）"]:
        idx = window.rfind(p)
        if idx != -1:
            return window_start + idx + 1

    return hard_end


def _adjust_start_forward(text: str, pos: int, max_lookahead: int = 200) -> int:
    """
    Move start forward to next paragraph boundary if it's close,
    to avoid starting from the middle of a paragraph/sentence.
    """
    if pos <= 0:
        return 0

    look = text[pos:min(len(text), pos + max_lookahead)]

    m = re.search(r"\n\s*\n", look)
    if m:
        return pos + m.end()

    n = look.find("\n")
    if n != -1:
        return pos + n + 1

    return pos


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[Tuple[int, int]]:
    """
    Return list of (start_char, end_char) spans over the input text.
    start/end are offsets relative to chapter text.
    """
    spans: List[Tuple[int, int]] = []
    start = 0
    n = len(text)

    while start < n:
        hard_end = min(n, start + chunk_size)
        end = _find_breakpoint(text, start, hard_end)
        if end <= start:
            end = hard_end

        spans.append((start, end))

        if end >= n:
            break

        next_start = max(0, end - overlap)
        next_start = _adjust_start_forward(text, next_start)

        # fail-safe to avoid infinite loop
        if next_start <= start:
            next_start = end

        start = next_start

    return spans


# ----------------------------
# IO
# ----------------------------
def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Raw book txt path")
    ap.add_argument("--out_dir", default="data/processed", help="Output directory")
    ap.add_argument("--book_title", default="平凡的世界")

    ap.add_argument("--chunk_size", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=200)

    ap.add_argument("--max_chapters", type=int, default=0, help="PoC: keep first N chapters; 0 means all")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)

    raw = in_path.read_text(encoding="utf-8", errors="ignore").lstrip("\ufeff")

    chapters = parse_chapters(raw)
    if args.max_chapters and args.max_chapters > 0:
        chapters = chapters[:args.max_chapters]

    # chapters.jsonl
    chapters_meta = [
        {k: ch[k] for k in ("chapter_id", "chapter_title", "part", "volume", "chapter_no")}
        for ch in chapters
    ]
    write_jsonl(out_dir / "chapters.jsonl", chapters_meta)

    # chunks.jsonl
    chunk_rows: List[Dict[str, Any]] = []
    for ch in chapters:
        spans = chunk_text(ch["text"], chunk_size=args.chunk_size, overlap=args.overlap)
        for idx, (s, e) in enumerate(spans):
            chunk_rows.append(
                {
                    "book_title": args.book_title,
                    "chapter_id": ch["chapter_id"],
                    "chapter_title": ch["chapter_title"],
                    "chunk_id": f"{ch['chapter_id']}_{idx:04d}",
                    "chunk_index": idx,
                    "start_char": s,
                    "end_char": e,
                    "text": ch["text"][s:e].strip(),
                }
            )
    write_jsonl(out_dir / "chunks.jsonl", chunk_rows)

    print(f"[OK] chapters={len(chapters_meta)} chunks={len(chunk_rows)} out_dir={out_dir.resolve()}")

def run(
    input_path: str,
    out_dir: str,
    book_title: str = "平凡的世界",
    chunk_size: int = 1200,
    overlap: int = 200,
    max_chapters: int = 2,  # PoC 默认 2
) -> None:
    in_path = Path(input_path)
    out_dir_path = Path(out_dir)

    raw = in_path.read_text(encoding="utf-8", errors="ignore").lstrip("\ufeff")
    chapters = parse_chapters(raw)
    if max_chapters and max_chapters > 0:
        chapters = chapters[:max_chapters]

    chapters_meta = [
        {k: ch[k] for k in ("chapter_id", "chapter_title", "part", "volume", "chapter_no")}
        for ch in chapters
    ]
    write_jsonl(out_dir_path / "chapters.jsonl", chapters_meta)

    chunk_rows: List[Dict[str, Any]] = []
    for ch in chapters:
        spans = chunk_text(ch["text"], chunk_size=chunk_size, overlap=overlap)
        for idx, (s, e) in enumerate(spans):
            chunk_rows.append(
                {
                    "book_title": book_title,
                    "chapter_id": ch["chapter_id"],
                    "chapter_title": ch["chapter_title"],
                    "chunk_id": f"{ch['chapter_id']}_{idx:04d}",
                    "chunk_index": idx,
                    "start_char": s,
                    "end_char": e,
                    "text": ch["text"][s:e].strip(),
                }
            )
    write_jsonl(out_dir_path / "chunks.jsonl", chunk_rows)

    print(f"[OK] chapters={len(chapters_meta)} chunks={len(chunk_rows)} out_dir={out_dir_path.resolve()}")

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    input_path = ROOT / "data" / "1_《平凡的世界》（实体版全本）作者：路遥.txt"
    out_dir = ROOT / "data" / "processed"

    run(
        input_path=str(input_path),
        out_dir=str(out_dir),
        max_chapters=0,   # 0 表示全书
        chunk_size=1200,
        overlap=200,
        book_title="平凡的世界",
    )


