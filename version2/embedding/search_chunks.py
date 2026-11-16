#!/usr/bin/env python3
"""search_chunks.py - retrieval over the Chroma index built by index_chunks.py.

From code (e.g. your orchestrator):
    from search_chunks import search
    hits = search("عايز اقدم على قرض شخصي", k=5, product="personal_loans", min_score=0.4)
    for h in hits: print(h["score"], h["id"], h["context_text"])

From the terminal (interactive):
    python search_chunks.py
    python search_chunks.py "ازاي افتح حساب جاري؟" --k 5 --product current_account
"""
import argparse
from functools import lru_cache

import chromadb

DB_DIR = "chroma_db"
COLLECTION = "bm_faq"
MODEL = "BAAI/bge-m3"


@lru_cache(maxsize=1)
def _model(name=MODEL):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


@lru_cache(maxsize=4)
def _collection(db_dir=DB_DIR, name=COLLECTION):
    return chromadb.PersistentClient(path=db_dir).get_collection(name)


def _where(product, lang, scope):
    clauses = [{k: v} for k, v in (("product", product), ("lang", lang), ("scope", scope)) if v]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def search(query, k=5, product=None, lang=None, scope=None, min_score=0.0,  # lang chain => vectordb.similaritysearch(query , k) 
           db_dir=DB_DIR, collection=COLLECTION, model=MODEL):
    """Return up to k hits, best first. score = cosine similarity (higher = closer).
    Hits below min_score are dropped, so an empty list means "nothing relevant"."""
    emb = _model(model).encode([query], normalize_embeddings=True).tolist()
    res = _collection(db_dir, collection).query(
        query_embeddings=emb, n_results=k, where=_where(product, lang, scope),
        include=["documents", "metadatas", "distances"])
    hits = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                    res["metadatas"][0], res["distances"][0]):
        score = round(1 - dist, 4)
        if score >= min_score:
            hits.append({"chunk_id": cid, "id": meta.get("id"), "score": score,
                         "context_text": doc, "metadata": meta})
    return hits


def _print(hits):
    if not hits:
        print("  (no result above min_score)")
    for h in hits:
        m = h["metadata"]
        flags = []
        if m.get("needs_review"):
            flags.append("NEEDS_REVIEW")
        if m.get("accessed_estimated"):
            flags.append("date~estimated")
        print(f"  {h['score']:.3f}  {h['id']:<18} [{m.get('product')}] {' '.join(flags)}")
        print("         " + h["context_text"].splitlines()[1][:110])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--product")
    ap.add_argument("--min-score", type=float, default=0.0)
    args = ap.parse_args()
    if args.query:
        _print(search(args.query, args.k, args.product, min_score=args.min_score))
        return
    print("اكتبي سؤال (Enter فاضي للخروج)")
    while (q := input("> ").strip()):
        _print(search(q, args.k, args.product, min_score=args.min_score))


if __name__ == "__main__":
    main()
