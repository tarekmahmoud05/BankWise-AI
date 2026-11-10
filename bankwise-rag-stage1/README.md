# BankWise AI — RAG stage 1: Load & normalize

Only slide step **01 — Load & normalize**: *read JSON files and map records into a shared document
structure.* No chunking, embeddings, ChromaDB or LLM yet. Standard library only.

```
data/*.json | *.jsonl ──► loaders.py ──► SourceRecord (schema.py) ──► documents.jsonl ──► stage 2 (chunking)
                          └ normalize.py (Arabic text helpers)
```

## Run

```bash
python -m bankwise_rag.cli load --data data/sample                       # inspect + validate
python -m bankwise_rag.cli load --data data/raw --out out/documents.jsonl  # export for the next stage
python -m bankwise_rag.cli load --data data/sample --show 2              # print normalized records
pip install pytest && pytest                                             # 19 tests
```

## What the loader accepts (so the final JSON schema can still change)
* file = list of records, or `{"records": [...], <file-level defaults>}` (also `faqs`, `items`, `data`, …); `.json` and `.jsonl`
* file-level defaults `language, category, product, source_url, retrieved_date, updated_at` apply to every record unless it sets its own
* English or Arabic key aliases, listed in `FIELD_ALIASES` in `loaders.py`
* record with question + answer → `faq`; with title and content/sections → `policy`

## What it guarantees
* Nothing is dropped silently: unreadable files → `errors`; unusable records → `warnings` (skipped)
* Missing `id` → stable generated id; duplicate ids in a file → renamed (later chunk ids can't collide)
* Missing `source_url` / `retrieved_date` → warning per record (KPI K-07)
* Missing `language` → detected (Arabic vs Latin letters)
* Original wording is preserved in `SourceRecord`. `normalize.py` (digits, diacritics, alef, tatweel, invisible chars) is provided for stage 2 to apply to **embedding text and queries only**, never to what the customer reads.

`data/sample/` is **synthetic** (invented fees, `example.com` URLs). Put the real files in `data/raw/`.
