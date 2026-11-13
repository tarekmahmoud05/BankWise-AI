"""
BankWise AI - command line interface (stages 1-4 + retrieval evaluation)

Short forms from the project root (recommended):
  python ingest.py [--reset]                 # = cli ingest
  python eval_retrieval.py                   # = cli eval

Full forms:
  python -m bankwise_rag.cli load --data data/sample
  python -m bankwise_rag.cli chunk --data data/sample --out out/chunks.jsonl
  python -m bankwise_rag.cli ingest [--reset]            # stage 4: embed + index into Chroma (bge-m3)
  python -m bankwise_rag.cli stats                       # what is in the index
  python -m bankwise_rag.cli query "ما رسوم القرض؟"      # quick look at the top results (debug)
  python -m bankwise_rag.cli eval [questions.json]       # hit@k / MRR on eval/eval_questions_bankwise.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .config import ROOT, default_embedder, resolve_data_dir, resolve_index_dir
from .loaders import load_directory
from .chunking import chunk_record
from .metadata import make_chunks


def _note_if_sample(data_dir: Path) -> None:
    if Path(data_dir).resolve() == (ROOT / "data" / "sample").resolve():
        print("  NOTE : data/sample is SYNTHETIC demo data (invented fees, example.com URLs), not real bank data")


def _print_warnings(rep, limit: int = 10) -> None:
    for w in rep.warnings[:limit]:
        print("  warn :", w)
    if len(rep.warnings) > limit:
        print(f"  warn : ... and {len(rep.warnings) - limit} more warnings")


def cmd_load(a: argparse.Namespace) -> int:
    data_dir = resolve_data_dir(a.data)
    records, rep = load_directory(data_dir)
    print(f"data dir : {data_dir}")
    _note_if_sample(data_dir)
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
    print(f"data dir : {data_dir}")
    _note_if_sample(data_dir)
    if rep.errors:
        for e in rep.errors:
            print("  ERROR:", e)
        return 1

    all_chunks = []
    for record in records:
        raw_chunks = chunk_record(record)
        chunks = make_chunks(record, raw_chunks)
        all_chunks.extend(chunks)

    _print_warnings(rep)
    if not all_chunks:
        print("  ERROR: no chunks were produced (all records skipped?)")
        return 1
    dup_ids = [cid for cid, n in Counter(c.chunk_id for c in all_chunks).items() if n > 1]
    if dup_ids:
        print(f"  ERROR: {len(dup_ids)} duplicate chunk_id(s), e.g. {dup_ids[:3]} - nothing exported")
        return 1

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


def cmd_ingest(a: argparse.Namespace) -> int:
    from .embeddings import build_embeddings
    from .pipeline import ingest

    data_dir, index_dir = resolve_data_dir(a.data), resolve_index_dir(a.index)
    print(f"data dir : {data_dir}")
    _note_if_sample(data_dir)
    print(f"index dir: {index_dir}")
    print(f"embedder : {a.embedder}  (first run with bge-m3 downloads the model, ~2.3 GB)")
    try:
        emb = build_embeddings(a.embedder)
        rep = ingest(data_dir, index_dir, emb, reset=a.reset, progress=print)
    except RuntimeError as exc:          # IngestError, EmbedderMismatchError, missing packages
        print("  ERROR:", exc)
        return 1
    if rep.reset:
        print("index was reset before ingesting")
    if rep.warnings:
        print(f"  warn : {rep.warnings} loader warnings (run `chunk` to see them)")
    s = rep.sync
    print(f"records {rep.records} -> chunks {rep.chunks} | added {s.added}  updated {s.updated}  "
          f"unchanged {s.unchanged}  deleted {s.deleted} | in index now: {s.total}")
    return 0


def cmd_stats(a: argparse.Namespace) -> int:
    from .index import index_stats

    index_dir = resolve_index_dir(a.index)
    st = index_stats(index_dir)
    print(f"index dir: {index_dir}")
    if st is None:
        print("  no collection yet - run `ingest` first")
        return 1
    for key in ("embedder", "chunks", "language", "kind", "source_file", "missing_source_url", "missing_retrieved_date"):
        print(f"{key:<22}: {st[key]}")
    return 0


def cmd_query(a: argparse.Namespace) -> int:
    from .embeddings import build_embeddings
    from .index import index_stats, open_store, search

    index_dir = resolve_index_dir(a.index)
    if index_stats(index_dir) is None:
        print("  ERROR: no collection yet - run `ingest` first")
        return 1
    try:
        store = open_store(index_dir, build_embeddings(a.embedder))
    except RuntimeError as exc:          # EmbedderMismatchError, missing packages
        print("  ERROR:", exc)
        return 1
    for doc, score in search(store, a.text, k=a.k, language=a.language):
        m = doc.metadata
        print(f"\n[{score:.3f}] {doc.id or m.get('record_id', '')}  ({m.get('language')}, {m.get('category') or '-'})")
        print(doc.page_content[:300].replace("\n", "\n         "))
    return 0


def cmd_eval(a: argparse.Namespace) -> int:
    from .eval_retrieval import run

    return run(a)


def main(argv=None) -> int:
    from .eval_retrieval import build_parser as build_eval_parser

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

    # Stage 4: embed & index
    s_ing = sub.add_parser("ingest", help="Stage 4: chunk, embed (bge-m3) and index into Chroma; safe to re-run")
    s_ing.add_argument("--data", type=Path)
    s_ing.add_argument("--index", type=Path, help="Chroma folder (default: <project>/chroma_db)")
    s_ing.add_argument("--embedder", choices=["bge-m3", "hash"], default=default_embedder())
    s_ing.add_argument("--reset", action="store_true", help="delete the collection first (needed to change embedder)")
    s_ing.set_defaults(fn=cmd_ingest)

    s_st = sub.add_parser("stats", help="show what is in the index (no model needed)")
    s_st.add_argument("--index", type=Path)
    s_st.set_defaults(fn=cmd_stats)

    s_q = sub.add_parser("query", help="debug: show the top results for a question")
    s_q.add_argument("text")
    s_q.add_argument("--index", type=Path)
    s_q.add_argument("--embedder", choices=["bge-m3", "hash"], default=default_embedder())
    s_q.add_argument("-k", type=int, default=4)
    s_q.add_argument("--language", choices=["ar", "en"])
    s_q.set_defaults(fn=cmd_query)

    s_ev = sub.add_parser("eval", help="evaluate retrieval (hit@k, MRR) on a questions file")
    build_eval_parser(s_ev)
    s_ev.set_defaults(fn=cmd_eval)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
