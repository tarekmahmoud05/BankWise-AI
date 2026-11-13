# BankWise AI — RAG stages 1–4: Load, chunk, add metadata, embed & index

Stage 1 (load & normalize), Stage 2 (prepare chunks), Stage 3 (add metadata), Stage 4 (embed & index with
LangChain + Chroma + BAAI/bge-m3). No retrieval/LLM answer yet (stage 5). Stages 1-3 use the standard library only;
stage 4 needs `requirements.txt`.

Project layout: `bankwise_rag/` = code only; `ingest.py`, `eval_retrieval.py` = shortcuts; `eval/` = questions;
`data/` = input JSON; `out/` = exports; `chroma_db/` = index (generated); `docs/` = handoff notes; `tests/`.
All paths are resolved from the project root (`ROOT` in config.py), so the commands work from any directory.

```
data/*.json | *.jsonl ──► loaders.py ──► SourceRecord ──► chunking.py ──► RawChunk ──► metadata.py ──► Chunk ──► out/chunks.jsonl
                          └ normalize.py (Arabic text helpers, used for embed_text)

Chunk ──► embeddings.py (bge-m3, normalises text before embedding) ──► index.py / pipeline.py ──► chroma_db/
```

## Run

```bash
python -m bankwise_rag.cli load  --data data/sample --show 2                  # stage 1: inspect + validate
python -m bankwise_rag.cli chunk --out out/chunks.jsonl                       # stages 1-3 on data/raw (real data)
python -m bankwise_rag.cli chunk --data data/sample --out out/chunks.jsonl    # same on the synthetic demo data
pip install pytest && pytest                                                  # tests (stage 4 tests need requirements.txt)
```

## Stage 4 — embed & index (Python 3.12 recommended, not 3.14)

```powershell
py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python ingest.py                                  # chunk + embed (bge-m3, first run downloads ~2.3 GB) + index
python ingest.py                                  # run again: added 0 updated 0 deleted 0
python eval_retrieval.py                          # hit@k / MRR on eval/eval_questions_bankwise.json (bge-m3, language ar)
python eval_retrieval.py --language all           # same, without the language filter
python -m bankwise_rag.cli stats                  # what is in the index
python -m bankwise_rag.cli query "ما رسوم بطاقة الخصم؟" -k 4 --language ar     # quick look, debug only
python ingest.py --reset                          # rebuild from scratch (needed to change embedder)
python ingest.py --embedder hash                  # offline test double (lexical, NOT for accuracy)
```
* `python ingest.py` / `python eval_retrieval.py` are thin shortcuts for `python -m bankwise_rag.cli ingest` / `eval`
  (all other commands: `python -m bankwise_rag.cli <load|chunk|stats|query>`). Defaults live in `bankwise_rag/config.py`.
  After `pip install -e .` you can also type `bankwise ingest`, `bankwise eval`, ...
* Index folder: `chroma_db/` (override with `--index` or `$BANKWISE_INDEX_DIR`); one collection `bankwise_kb`, cosine.
* The collection remembers its embedder; opening it with another one raises an error (use `--reset`).
* Ingest is idempotent: unchanged chunks (same `content_hash`) are skipped, edited ones replaced, vanished ones deleted.
* The stored document is the ORIGINAL chunk text; vectors are computed from `normalize_text(text)` (= `embed_text`),
  and queries are normalised the same way.
Data folder priority: `--data` > `$BANKWISE_DATA_DIR` > `data/raw` (if it has JSON) > `data/sample`.
The CLI prints a NOTE when it is using `data/sample` (synthetic).

## What the loader accepts
* file = list of records, or `{"records": [...], <file-level defaults>}` (also `faqs`, `items`, `data`, ...); `.json` and `.jsonl`
* file-level defaults `language, category, product, source_url, retrieved_date, updated_at` apply to every record unless it sets its own
* English or Arabic key aliases, listed in `FIELD_ALIASES` in `loaders.py`
* record with question + answer -> `faq`; with title and content/sections -> `policy`
* **bilingual records** `{"question": {"ar": .., "en": ..}, "answer": {"ar": .., "en": ..}}` become one record per language,
  ids `ABT-001-ar` / `ABT-001-en`

## What it guarantees
* Nothing is dropped silently: unreadable files -> `errors`; unusable records -> `warnings` (skipped)
* Missing `id` -> stable generated id; duplicate ids in a file -> renamed
* Missing `source_url` / `retrieved_date` -> warning per record (KPI K-07). `chunk` prints the first 10 warnings and a count.
* Chunks: FAQ Q&A stays together (question repeated if the answer is split); policies split on sections / Markdown `#` headings
  with a `title — heading` prefix; max 900 chars, 120 overlap, whole lines kept, no text truncated.
* `chunk` refuses to export if two chunks share a `chunk_id` or if no chunk was produced.
* Original wording is kept in `text`; `embed_text = normalize_text(text)` is for embedding and queries only.

`data/sample/` is **synthetic** (invented fees, `example.com` URLs). Real data goes in `data/raw/`.
Rule-engine files (e.g. `car_loan_eligibility_rules_*.json`) do **not** go in `data/raw/`: they are not RAG content.
