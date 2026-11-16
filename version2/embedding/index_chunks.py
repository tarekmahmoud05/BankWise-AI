#!/usr/bin/env python3
"""index_chunks.py - embed `embed_text` of each chunk with bge-m3 and store in Chroma.

Only `embed_text` is embedded. `context_text` is stored as the Chroma document
(what the LLM receives). Selected metadata is stored for filtering / warnings.

Usage:
    python index_chunks.py --chunks chunks_ar_1.json --reset
    python index_chunks.py --chunks chunks_ar_1.json chunks_en.json   # later, to add English
"""
import argparse
import json
from pathlib import Path

import chromadb

MODEL = "BAAI/bge-m3"

# scalar metadata kept per chunk (None values are dropped: Chroma rejects them)
SCALAR_FIELDS = ["id", "lang", "product", "scope", "needs_review", "accessed",
                 "accessed_estimated", "source_json", "source_document",
                 "program", "card_type", "translated"]


def build_metadata(chunk):
    meta = {k: chunk[k] for k in SCALAR_FIELDS if chunk.get(k) is not None}
    if chunk.get("tags"):
        meta["tags"] = ",".join(chunk["tags"])   # Chroma metadata cannot hold lists
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="+", required=True)
    ap.add_argument("--db", default="chroma_db")
    ap.add_argument("--collection", default="bm_faq")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--reset", action="store_true", help="delete the collection first")
    args = ap.parse_args()

    chunks = []
    for f in args.chunks:
        chunks += json.loads(Path(f).read_text(encoding="utf-8"))
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk_ids"
    assert all(c["embed_text"].strip() and c["context_text"].strip() for c in chunks), "empty text"

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)          # uses the GPU automatically if available
    print(f"model: {args.model} | device: {model.device} | chunks: {len(chunks)}")

    client = chromadb.PersistentClient(path=args.db)
    if args.reset:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass
    col = client.get_or_create_collection(args.collection, metadata={"hnsw:space": "cosine"})

    for i in range(0, len(chunks), args.batch):
        batch = chunks[i:i + args.batch]
        emb = model.encode([c["embed_text"] for c in batch],
                           normalize_embeddings=True, batch_size=args.batch).tolist()
        col.upsert(ids=[c["chunk_id"] for c in batch],
                   embeddings=emb,
                   documents=[c["context_text"] for c in batch],
                   metadatas=[build_metadata(c) for c in batch])
        print(f"  indexed {min(i + args.batch, len(chunks))}/{len(chunks)}")

    print(f"done. collection '{args.collection}' now holds {col.count()} chunks -> {args.db}")


if __name__ == "__main__":
    main()
