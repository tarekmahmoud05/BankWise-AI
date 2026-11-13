"""Retrieval evaluation - measures whether the right record is retrieved (hit@k, MRR) and shows
how the top-1 score behaves on tool / out-of-scope questions (to help choose a min_score).

Question file items (default: eval/eval_questions_bankwise.json):
  {"type": "faq",          "query": "...", "expected_id": "LOAN-001"}              # or a list = any is accepted
  {"type": "followup",     "context": "previous user msg", "query": "...", "expected_id": "..."}
  {"type": "tool",         "query": "...", "expected_id": null}                    # not scored, top-1 printed
  {"type": "out_of_scope", "query": "...", "expected_id": null}                    # not scored, top-1 printed

expected_id = the record id in your JSON data (e.g. "ABT-001"), NOT the chunk id.
Bilingual records are stored as ABT-001-ar / ABT-001-en; "ABT-001" matches both, and the two twins
plus any extra chunks of the same record count once (results are collapsed to distinct records).

Run from the project root:   python eval_retrieval.py
Everything has a default (questions file, embedder, language); see `python eval_retrieval.py -h`.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .config import DEFAULT_LANGUAGE, EVAL_QUESTIONS, default_embedder, resolve_index_dir
from .embeddings import build_embeddings
from .index import index_stats, open_store, search


def _as_list(x):
    return x if isinstance(x, list) else [x]


def _base_id(record_id: str) -> str:
    """ABT-001-ar / ABT-001-en -> ABT-001 (the Arabic and English twin count as the same record)."""
    return record_id[:-3] if record_id.endswith(("-ar", "-en")) else record_id


def _retrieve(store, query: str, k: int, language: Optional[str]) -> List[Tuple[str, float]]:
    """Top-k DISTINCT records (best chunk per record), as [(record_id, score)]."""
    raw = search(store, query, k=k * 4, language=language)
    seen, out = set(), []
    for doc, score in raw:
        rid = _base_id(doc.metadata.get("record_id", ""))
        if rid in seen:
            continue
        seen.add(rid)
        out.append((rid, score))
        if len(out) == k:
            break
    return out


def _rank_of(store, query: str, want, k: int, language: Optional[str]):
    hits = _retrieve(store, query, k, language)
    rank = next((i for i, (rid, _) in enumerate(hits, 1) if _base_id(rid) in want), None)
    return rank, hits


def _summary(name: str, ranks: list) -> None:
    n = len(ranks)
    if not n:
        return
    h = lambda k: sum(1 for r in ranks if r and r <= k)
    mrr = sum(1 / r for r in ranks if r) / n
    print(f"{name:<28} n={n:<3} hit@1={h(1)}/{n}  hit@3={h(3)}/{n}  hit@5={h(5)}/{n}  MRR={mrr:.3f}")


def build_parser(p: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    p = p or argparse.ArgumentParser(prog="eval_retrieval")
    p.add_argument("questions", nargs="?", type=Path, default=EVAL_QUESTIONS,
                   help=f"questions JSON (default: {EVAL_QUESTIONS.name} in eval/)")
    p.add_argument("--index", type=Path, help="Chroma folder (default: project chroma_db)")
    p.add_argument("--embedder", choices=["bge-m3", "hash"], default=default_embedder())
    p.add_argument("--language", choices=["ar", "en", "all"], default=DEFAULT_LANGUAGE,
                   help="filter the search to one language; 'all' = no filter (default: %(default)s)")
    p.add_argument("--k", type=int, default=5, help="how many distinct records to keep per query")
    return p


def run(a: argparse.Namespace) -> int:
    language = None if a.language == "all" else a.language
    if a.embedder == "hash":
        print("WARNING: 'hash' embedder is lexical, not semantic. Don't trust these numbers or pick min_score from them.\n")

    index_dir = resolve_index_dir(a.index)
    stats = index_stats(index_dir)
    if stats is None:
        print(f"  ERROR: no collection in {index_dir} - run: python ingest.py")
        return 1
    try:
        store = open_store(index_dir, build_embeddings(a.embedder))
    except RuntimeError as exc:          # EmbedderMismatchError, missing packages
        print("  ERROR:", exc)
        return 1

    with open(a.questions, encoding="utf-8") as f:
        items = json.load(f)
    print(f"questions: {a.questions} ({len(items)} items)")
    print(f"index: {index_dir} | embedder {stats['embedder']} | {stats['chunks']} chunks | language filter: {language or 'none'}\n")

    faq_ranks, fu_alone, fu_ctx = [], [], []
    misses, correct_scores, unscored = [], [], []

    for it in items:
        typ, q = it.get("type", "faq"), it["query"]
        if typ == "faq":
            want = _as_list(it["expected_id"])
            r, hits = _rank_of(store, q, want, a.k, language)
            faq_ranks.append(r)
            if r == 1:
                correct_scores.append(hits[0][1])
            else:
                misses.append((typ, q, want, r, [(i, round(s, 3)) for i, s in hits[:3]]))
        elif typ == "followup":
            want = _as_list(it["expected_id"])
            r1, h1 = _rank_of(store, q, want, a.k, language)
            r2, _ = _rank_of(store, it["context"] + " " + q, want, a.k, language)
            fu_alone.append(r1)
            fu_ctx.append(r2)
            if r1 != 1:
                misses.append((typ + " (alone)", q, want, r1, [(i, round(s, 3)) for i, s in h1[:3]]))
        else:
            hits = _retrieve(store, q, a.k, language)
            unscored.append((typ, q, hits[0][0] if hits else None, hits[0][1] if hits else None))

    print("=== scores ===")
    _summary("faq", faq_ranks)
    _summary("followup - query alone", fu_alone)
    _summary("followup - context+query", fu_ctx)

    print("\n=== tool / out_of_scope: top-1 hit (should be rejected or routed to tools) ===")
    for typ, q, cid, sc in unscored:
        print(f"  {typ:<13} {'-' if sc is None else f'{sc:.3f}'}  {str(cid):<26} {q}")
    for typ in ("tool", "out_of_scope"):
        s = [sc for t, _, _, sc in unscored if t == typ and sc is not None]
        if s:
            print(f"  max top-1 score for {typ}: {max(s):.3f}")
    if correct_scores:
        print(f"  faq correct top-1 scores: min={min(correct_scores):.3f}  median={st.median(correct_scores):.3f}")
    print("  -> a good min_score is above the out_of_scope max and below most of the faq correct scores.")

    print("\n=== misses (faq rank != 1, followup alone rank != 1) ===")
    for typ, q, want, rank, top in misses:
        print(f"\n[{typ}] {q}\n  expected {want} (rank {rank}) | top3 {top}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
