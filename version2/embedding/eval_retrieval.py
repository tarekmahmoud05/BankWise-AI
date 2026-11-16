#!/usr/bin/env python3
"""eval_retrieval.py - measures whether the right chunk is retrieved.

eval_questions.json format (use REAL colloquial questions):
  [{"query": "عايزة اعمل قرض شخصي اعمل ايه", "expected_id": "RET-001"},
   {"query": "...", "expected_id": ["PL-003", "PL-004"]}]      # several acceptable ids are allowed

Usage:  python eval_retrieval.py eval_questions.json
Prints hit@1 / hit@3 / hit@5 / MRR and every miss with what was returned instead.
"""
import json
import sys

from search_chunks import search

path = sys.argv[1] if len(sys.argv) > 1 else "eval_questions.json"
items = json.load(open(path, encoding="utf-8"))
ranks, misses = [], []
for it in items:
    want = it["expected_id"] if isinstance(it["expected_id"], list) else [it["expected_id"]]
    hits = search(it["query"], k=5)
    rank = next((i for i, h in enumerate(hits, 1) if h["id"] in want), None)
    ranks.append(rank)
    if rank != 1:
        misses.append((it["query"], want, rank, [(h["id"], h["score"]) for h in hits[:3]]))

n = len(ranks)
for k in (1, 3, 5):
    print(f"hit@{k}: {sum(1 for r in ranks if r and r <= k)}/{n}")
print(f"MRR: {sum(1 / r for r in ranks if r) / n:.3f}")
for q, want, rank, top in misses:
    print(f"\nMISS  {q}\n  expected {want} (rank {rank}) | top3 {top}")
