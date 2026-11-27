#!/usr/bin/env python3
"""eval_retrieval.py - measures whether the right chunk is retrieved.

Each item in the JSON file:
  {"query": "...", "expected_id": "RET-001"}                      # type defaults to "faq"
  {"type": "faq",          "query": "...", "expected_id": ["A","B"]}   # several acceptable ids
  {"type": "followup",     "context": "previous user message", "query": "...", "expected_id": "..."}
  {"type": "tool",         "query": "...", "expected_id": null}    # should go to eligibility/estimate tools
  {"type": "out_of_scope", "query": "...", "expected_id": null}    # nothing in the FAQ should match

faq / followup are scored (hit@1/3/5, MRR). followup is scored twice: query alone,
and "context + query" (shows whether you need to rewrite follow-ups with chat history).
tool / out_of_scope are NOT scored: their top-1 scores are printed so you can pick a min_score
that rejects them while keeping the correct faq hits.

Usage (from anywhere):
  python eval_retrieval.py                              # uses files next to the script
  python eval_retrieval.py eval_questions.json --db ..\\chroma_db
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# search_chunks.py lives in embedding/. This script may sit in embedding/ or in embedding/eval_real/,
# so both folders are put on the import path.
for p in (HERE, HERE.parent):
    sys.path.insert(0, str(p))
from search_chunks import search, _collection  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("questions", nargs="?", help="eval json (default: eval_questions_realistic.json next to this script)")
ap.add_argument("--db", help="Chroma folder (default: chroma_db next to this script, else ../chroma_db)")
ap.add_argument("--collection", default="bm_faq")
args = ap.parse_args()

if args.questions:
    q_path = Path(args.questions)
else:
    q_path = next((HERE / n for n in ("eval_questions_realistic.json", "eval_questions.json")
                   if (HERE / n).exists()), None)
if q_path is None or not q_path.exists():
    sys.exit("questions file not found: pass it as the first argument")

if args.db:
    DB = Path(args.db)
else:
    DB = HERE / "chroma_db" if (HERE / "chroma_db").exists() else HERE.parent / "chroma_db"
if not DB.exists():
    sys.exit(f"Chroma folder not found: {DB}  (pass --db)")
DB = str(DB.resolve())

items = json.load(open(q_path, encoding="utf-8"))
print(f"questions: {q_path}  ({len(items)} items)")
print(f"chroma db: {DB}  | collection '{args.collection}' holds {_collection(DB, args.collection).count()} chunks\n")


def as_list(x):
    return x if isinstance(x, list) else [x]


def rank_of(query, want):
    hits = search(query, k=5, db_dir=DB, collection=args.collection)
    return next((i for i, h in enumerate(hits, 1) if h["id"] in want), None), hits


def summary(name, ranks):
    n = len(ranks)
    if not n:
        return
    h = lambda k: sum(1 for r in ranks if r and r <= k)
    mrr = sum(1 / r for r in ranks if r) / n
    print(f"{name:<28} n={n:<3} hit@1={h(1)}/{n}  hit@3={h(3)}/{n}  hit@5={h(5)}/{n}  MRR={mrr:.3f}")


faq_ranks, fu_alone, fu_ctx = [], [], []
misses, correct_scores, unscored = [], [], []

for it in items:
    typ = it.get("type", "faq")
    q = it["query"]
    if typ == "faq":
        want = as_list(it["expected_id"])
        r, hits = rank_of(q, want)
        faq_ranks.append(r)
        if r == 1 and hits:
            correct_scores.append(hits[0]["score"])
        if r != 1:
            misses.append((typ, q, want, r, [(h["id"], h["score"]) for h in hits[:3]]))
    elif typ == "followup":
        want = as_list(it["expected_id"])
        r1, h1 = rank_of(q, want)
        r2, _ = rank_of(it["context"] + " " + q, want)
        fu_alone.append(r1)
        fu_ctx.append(r2)
        if r1 != 1:
            misses.append((typ + " (alone)", q, want, r1, [(h["id"], h["score"]) for h in h1[:3]]))
    else:
        hits = search(q, k=1, db_dir=DB, collection=args.collection)
        unscored.append((typ, q, hits[0]["id"] if hits else None, hits[0]["score"] if hits else None))

print("=== scores ===")
summary("faq", faq_ranks)
summary("followup - query alone", fu_alone)
summary("followup - context+query", fu_ctx)

print("\n=== tool / out_of_scope: top-1 hit (these should be rejected or routed to tools) ===")
for typ, q, cid, sc in unscored:
    print(f"  {typ:<13} {sc if sc is None else f'{sc:.3f}'}  {str(cid):<26} {q}")
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