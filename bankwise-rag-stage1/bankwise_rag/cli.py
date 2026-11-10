"""Stage 1 command line.

  python -m bankwise_rag.cli load --data data/sample            # inspect + validate
  python -m bankwise_rag.cli load --data data/raw --out out/documents.jsonl   # export for stage 2
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bankwise_rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("load", help="step 01: load & normalize JSON into the shared document structure")
    s.add_argument("--data", type=Path)
    s.add_argument("--out", type=Path, help="write documents as JSONL (input for stage 2)")
    s.add_argument("--show", type=int, default=0, metavar="N", help="print the first N normalized records")
    s.set_defaults(fn=cmd_load)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
