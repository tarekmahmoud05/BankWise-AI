"""
BankWise AI — Command Line Interface (Stages 1–3)

Commands:
  python -m bankwise_rag.cli load --data data/sample
  python -m bankwise_rag.cli chunk --data data/sample --out out/chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .config import resolve_data_dir
from .loaders import load_directory
from .chunking import chunk_record
from .metadata import make_chunks


def cmd_load(a: argparse.Namespace) -> int:
    data_dir = resolve_data_dir(a.data)
    records, rep = load_directory(data_dir)
    print(f"data dir : {data_dir}")
    print(f"files    : {rep.files}   records: {len(records)}")
    for key in ("kind", "language", "category", "source_file"):
        print(f"{key:<9}: {dict(Counter(getattr(r, key) or '(none)' for r in records))}")
    for w in rep.warnings:
        print("  warn :", w)
    for e in rep.errors:
        print("  ERROR:", e)
    if a.show:
        for r in records[: a.show]:
            print("\n" + json.dumps(asdict(r), ensure_ascii=False, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        print(f"\nwrote {len(records)} documents -> {a.out}")
    return 0 if not rep.errors else 1


def cmd_chunk(a: argparse.Namespace) -> int:
    data_dir = resolve_data_dir(a.data)
    records, rep = load_directory(data_dir)
    if rep.errors:
        for e in rep.errors:
            print("  ERROR:", e)
        return 1

    all_chunks = []
    for record in records:
        raw_chunks = chunk_record(record)
        chunks = make_chunks(record, raw_chunks)
        all_chunks.extend(chunks)

    print(f"Ingested {len(records)} records from {rep.files} file(s).")
    print(f"Generated {len(all_chunks)} total Chunk objects across Stage 2 & Stage 3.")

    if a.show:
        for c in all_chunks[: a.show]:
            print("\n" + json.dumps(asdict(c), ensure_ascii=False, indent=2))

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        print(f"\nSuccessfully exported {len(all_chunks)} chunks -> {a.out}")

    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bankwise_rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    
    # Stage 1 Load Command
    s_load = sub.add_parser("load", help="Stage 1: load & normalize JSON into SourceRecords")
    s_load.add_argument("--data", type=Path)
    s_load.add_argument("--out", type=Path, help="write documents as JSONL")
    s_load.add_argument("--show", type=int, default=0, metavar="N", help="print the first N normalized records")
    s_load.set_defaults(fn=cmd_load)

    # Stage 2 & 3 Chunk Command
    s_chunk = sub.add_parser("chunk", help="Stage 2 & 3: chunk records and add metadata")
    s_chunk.add_argument("--data", type=Path)
    s_chunk.add_argument("--out", type=Path, help="export final Chunk objects as JSONL for indexing")
    s_chunk.add_argument("--show", type=int, default=0, metavar="N", help="print the first N chunks")
    s_chunk.set_defaults(fn=cmd_chunk)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
