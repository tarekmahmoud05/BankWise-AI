# BankWise AI — RAG pipeline: how to continue (stages 2–5)

Handoff for the team. Stage 1 (Load & normalize) is done and tested. This file tells you what each remaining
stage must take in, give out, and prove before you merge it.

Stage numbers follow the team slide **"Current step: building the RAG pipeline"**:

| Stage | Slide step | Status |
|---|---|---|
| 1 | 01 Load & normalize | **Done** — `bankwise-rag-stage1.zip`, 19 tests |
| 2 | 02 Prepare chunks | To do |
| 3 | 03 Add metadata | To do |
| 4 | 04 Embed & index (ChromaDB) | To do |
| 5 | 05 Retrieve & answer | To do |

A **working reference implementation of stages 2–5** exists in `bankwise-rag.zip` (52 tests passing offline).
Use it as a starting point or as an answer key. Read it critically; it has known gaps (section 9).

---

## 0. Start in 10 minutes

```bash
unzip bankwise-rag-stage1.zip && cd bankwise-rag-stage1
pip install pytest && pytest                                   # 19 passed
python -m bankwise_rag.cli load --data data/sample --show 1    # see one normalized record
python -m bankwise_rag.cli load --data data/raw --out out/documents.jsonl   # when real JSON is in data/raw/
```

Work one stage per branch and one pull request per stage. Do not start stage N+1 before stage N's
"Done when" list is green.

---

## 1. The contract between stages

Everything flows through two dataclasses. Stage 1 owns `SourceRecord`; stages 2–3 add `RawChunk` and `Chunk`.

```python
# Stage 1 output (already in bankwise_rag/schema.py)
SourceRecord(record_id, kind="faq"|"policy", language="ar"|"en", category, product,
             source_file, source_url, retrieved_date, updated_at,
             title, question, answer, body, sections=[(heading, text), ...])

# Add these in stage 2/3 (copy from the reference zip's schema.py)
RawChunk(text, heading="")                                  # stage 2 output
Chunk(chunk_id, text, embed_text, metadata: dict)           # stage 3 output
```

* `text` = original wording. It is shown to customers and given to the LLM.
* `embed_text` = `normalize_text(text)`. It is only used for embedding. Never show it to anyone.
* `metadata` = flat dict of `str|int|float|bool`. ChromaDB rejects `None`, lists and nested dicts.

Reading stage 1's export back in: `SourceRecord(**row)`, converting each `sections` entry from a list to a tuple.
Better: call `load_directory(data_dir)` in-process and use `documents.jsonl` only for debugging.

## 2. Rules every stage must respect

1. **RAG explains; the rule engine decides.** RAG answers policy/FAQ questions. It never decides loan
   eligibility and never calculates installments. That belongs to the deterministic rule engine (team slide, key design rule).
2. **Abstain instead of guessing.** Weak evidence means the fixed Arabic fallback and **no LLM call**.
3. **Every answer carries its source** (document, URL, retrieval date), built from metadata, never written by the LLM (KPI K-07).
4. **Original text is sacred.** Normalise for matching, never for display. A customer must read the bank's exact fee wording.
5. **Retrieved text is untrusted data**, never instructions (prompt-injection risk).
6. **No sensitive personal data** is asked for, logged or stored.
7. **Re-running anything must be safe** (idempotent) and must never crash on odd input.

---

## 3. Target layout

```
bankwise_rag/
  loaders.py  normalize.py  schema.py  config.py     # stage 1 (done)
  chunking.py                                         # stage 2
  metadata.py                                         # stage 3
  embeddings.py  index.py  pipeline.py                # stage 4
  retriever.py  answer.py  evaluate.py                # stage 5
  cli.py                                              # add: chunk | ingest | query | ask | stats | eval
tests/ test_<stage>.py per stage
eval/golden.jsonl                                      # stage 5, but start collecting questions NOW
```

---

## 4. Stage 2 — Prepare chunks (`chunking.py`)

**In:** `SourceRecord`. **Out:** `list[RawChunk]`.

Slide rule: *keep each FAQ Q&A together; split longer policy text by topic or heading.*

| Case | Behaviour |
|---|---|
| FAQ that fits (≤ 900 chars) | **One chunk**: `السؤال: …\nالإجابة: …` (`Question:/Answer:` for English). Never split Q from A. |
| FAQ longer than the limit | Split the *answer* only; **repeat the question** in every piece so each is self-explanatory. |
| Policy with `sections` | One group per section, heading kept. |
| Policy with `body` | Split on Markdown headings (`# …`). No headings → one section titled with the document title. |
| Long section | Pack whole lines up to ~900 chars with ~120 chars overlap. Oversize lines → sentences (`. ! ? ؟ ؛`) → hard split at a space. |
| Every policy chunk | Prefix `title — heading` so a chunk retrieved alone still says what it is about. |

Pitfalls
* Keep lines as the packing unit: bullet lists and fee tables stay readable. Do not flatten with spaces.
* A single 5,000-character string with no spaces must still terminate (hard split).
* Do not drop the prefix from the size budget calculation or chunks drift above the limit.
* 900 / 120 are starting values. Tune them with the stage 5 evaluation on real data.

Done when: FAQ pair is one chunk; long FAQ repeats its question in each piece; headings split policies;
overlap is carried forward; no chunk exceeds the limit by more than the prefix; giant unbroken text terminates.
Reference: `chunk_record()` and `tests/test_normalize_chunking.py`. On the sample data: 18 records → 22 chunks.

## 5. Stage 3 — Add metadata (`metadata.py`)

**In:** `SourceRecord` + `list[RawChunk]`. **Out:** `list[Chunk]`.

Required by the slide: **language, category, product, source file, record ID, update date.**
Also store `source_url` and `retrieved_date` (KPI K-07), `title`, `heading`, `chunk_index`, `chunk_count`, `kind`.

* `chunk_id = "<file stem>::<record_id>::<chunk_index>"`. It must be **deterministic**: same input, same id.
* `content_hash` = hash of the chunk text **plus all other metadata**. A metadata-only edit (say a category change)
  must change the hash, or the index will keep serving stale metadata.
* `embed_text = normalize_text(text)`.
* Unknown values are `""`, never `None`. Empty `source_url`/`retrieved_date` should already have been
  warned about in stage 1; consider making them a hard error for production data.

Done when: required keys present; every value is a scalar; ids deterministic; hash changes on text edit **and** on metadata edit.
Reference: `make_chunks()` and `tests/test_loaders_metadata.py`.

## 6. Stage 4 — Embed & index (`embeddings.py`, `index.py`, `pipeline.py`)

**In:** `list[Chunk]`. **Out:** a persistent ChromaDB collection.

**Embedder interface** (keep it this small so models are swappable):
```python
class Embedder(Protocol):
    name: str                      # stored on the collection
    default_min_score: float       # used by stage 5 when no threshold is configured
    def embed_documents(self, texts) -> list[list[float]]: ...
    def embed_query(self, text) -> list[float]: ...
```
Two implementations:
* **Real:** `sentence-transformers` with BGE-M3 (`BAAI/bge-m3`). `normalize_embeddings=True` so cosine works.
  For `multilingual-e5-*` add the prefixes `"query: "` / `"passage: "` or quality drops silently.
  GATE-AraBERT is named in the proposal; **verify its Hugging Face id** before using it.
* **Offline test double:** a hashing/character-n-gram embedder. It is **lexical, not semantic**. It exists so
  tests and CI run without downloading a model. Never use it to report accuracy or to pick a threshold.

**Index rules** (all learned the hard way):
1. Cosine space, persistent client, one collection.
2. Store the **embedder name on the collection** and refuse to open it with a different embedder. Mixed vector
   spaces ruin retrieval without raising any error. The fix is `ingest --reset`.
3. **Idempotent ingest** per source file: skip chunks whose `content_hash` is unchanged, replace edited ones,
   delete chunks that disappeared from the file, and delete chunks of source files that were removed from the data folder.
4. Upsert in batches (~64). Embed only the changed chunks.
5. Metadata filters with more than one condition need `{"$and": [...]}`.

Done when: ingest twice → second run adds/updates/deletes 0; editing a record updates exactly one chunk; removing a
record or file removes its chunks; opening with a different embedder raises; every stored chunk has source_url, date, language, record_id.
Reference: `index.py`, `pipeline.py`, tests in `tests/test_index_retrieval_answer.py`.

## 7. Stage 5 — Retrieve & answer (`retriever.py`, `answer.py`, `evaluate.py`)

### 7a. Retrieval with an abstain gate
**In:** question text. **Out:** `RetrievalResult(candidates, hits, abstained, reason, top_score, threshold)`.

1. Normalise the query (same function as the chunks). Empty → abstain (`empty_query`). Truncate very long input (~1,000 chars). Empty index → abstain.
2. Embed, search top-k (default 4), optional metadata filters (`language`, `product`, `category`).
3. Score = `1 − cosine distance`. Keep only chunks with score ≥ threshold as `hits`.
4. No hits → `abstained=True`.

**The threshold is model-specific and must be calibrated.** The reference default of `0.45` for BGE-M3 is a guess.
Procedure: build the golden set (below), run `eval --sweep`, look at "answerable questions wrongly refused" versus
"trap questions wrongly answered" per threshold, choose the knee, set `BANKWISE_MIN_SCORE`.
Re-calibrate whenever the embedder, chunking or knowledge base changes materially.

### 7b. Grounded answer
`answer_question(question, retriever, llm)` must enforce in **code**, not only in the prompt:

| Situation | Behaviour |
|---|---|
| Retriever abstains | Fixed Arabic fallback, **LLM never called** |
| Retrieval or LLM throws | Same fallback (no stack traces to the user) |
| LLM itself says it cannot answer | Normalise to the same fallback, no sources shown |
| Normal answer | LLM text + `المصادر:` footer **built from metadata** (title, URL, retrieval date) |

Prompt design: instructions only in the system message; each chunk wrapped in `<source id=…>` tags with `<` and `>`
escaped (so a chunk cannot close the tag); say explicitly that source text is untrusted; require Arabic; require
numbers copied verbatim and never calculated; forbid promising approval; tell the model to defer eligibility to the calculator.

LLM interface (the LLM owner implements one class):
```python
class LLM(Protocol):
    def generate(self, system: str, user: str) -> str: ...
```
`llm=None` means "return the best chunk verbatim" — useful for UI work without an API key.

### 7c. Evaluation (`evaluate.py`, `eval/golden.jsonl`)
Golden file, one JSON per line: `{"question": "...", "expected": ["record_id", ...], "answerable": true|false}`.
Report hit@1, hit@k, MRR on answerable questions, and false-abstain / correct-abstain counts.

**Start collecting questions now**, in the real Egyptian dialect people use ("ورق", "الكارت", "اقسط على كام سنة"), not textbook Arabic:
* the 10 representative questions required for KPI K-01 (target: at least 8 of 10 correct),
* **trap questions**: unrelated (restaurants), banking-adjacent but not in the knowledge base (savings-certificate rates, exchange rate), and a prompt-injection attempt,
* odd inputs: empty, whitespace, very long, emoji, HTML.

Done when: abstain returns the fallback with zero LLM calls; LLM exception → fallback; sources come from metadata;
prompt escaping test passes; all odd inputs return without raising; the threshold was calibrated on real data and written down.
Reference: `retriever.py`, `answer.py`, `evaluate.py`.

---

## 8. Integration with the rest of the system

* **Backend (FastAPI):** build the embedder, index and retriever **once at startup**, not per request.
  Call `answer_question(...)`. Return `text`, `abstained`, `reason`, `sources`, `latency_ms`.
* **Rule engine:** separate module with its own config file. The chat layer routes eligibility intents there. RAG only explains the policy text.
* **Dashboard / KPIs:** log per request: question, abstained, reason, top_score, latency_ms, and token counts from the LLM client. Do not log personal data.
* **Follow-up questions** (session memory): rewrite "and what about the fees?" into a standalone query *before* retrieval. Needs the session store and is out of scope for stages 2–5.
* **Arabic-only or bilingual?** The proposal says both in different places. The index stores `language` and BGE-M3 retrieves across languages, so either works. Decide, then put the language guard in the chat layer, not here.

## 9. Known gaps in the reference implementation — fix or verify these

1. **BGE-M3 has never been run.** The environment it was built in could not download from Hugging Face. The
   real-model embedder is written but **untested**. First task for whoever has internet: run stage 4 and `eval` with it.
2. **The offline embedder is weak on dialect.** On the sample golden set it reached hit@1 ≈ 0.71 and could not
   separate answerable from trap questions by score alone. Expect a real model to do better, and verify it.
3. **`data/sample/` is synthetic** (invented fees, `example.com` URLs). Never ship or demo it as real bank data.
4. **ChromaDB or pgvector?** The proposal's architecture names pgvector; the slide says ChromaDB. Decide whether
   Chroma is the dev store or the final one. Only `index.py` would change.
5. **No reranker.** If hit@1 is low after calibration, a cross-encoder reranker over the top 10 is the usual next step.
6. **Chunk size, overlap and top-k are untuned.** Tune on real data with the eval, one variable at a time.
7. **PII masking** of incoming questions is not implemented.

## 10. Definition of done for each stage's pull request

- [ ] New code has tests; the full suite passes **offline** (no network, no model download)
- [ ] Re-running the stage on unchanged input changes nothing
- [ ] Odd input (empty, huge, Arabic digits, emoji) cannot raise
- [ ] No `None` in metadata; no personal data in logs
- [ ] README section for the stage: input, output, how to run, known limits
- [ ] For stage 5: threshold calibrated on **real** questions and the number recorded in the README

## 11. Suggested order

Stages 2 and 3 are small and can be one PR. Stage 4 is the biggest risk (real model, Chroma behaviour), so start
it early on a machine with internet. Stage 5's evaluation needs the golden set, so write the questions **in parallel
with stages 2–4**, not after.
