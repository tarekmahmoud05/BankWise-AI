# BankWise AI — Stages 2 & 3 Implementation & Handoff Summary

**Project Component:** RAG Knowledge Engine  
**Completed Stages:** Stage 2 (Prepare Chunks) & Stage 3 (Add Metadata)  
**Status:** Successfully Completed & Verified

---

## 1. Executive Summary

During this session, we expanded the **BankWise AI** RAG pipeline from Stage 1 (Document Load & Normalize) into **Stage 2 (Chunking)** and **Stage 3 (Metadata Processing)**.

All code modifications adhere strictly to the project pipeline specifications:

* **Q&A Preservation:** Question and Answer pairs remain unified; oversized answers repeat the question context header in every split chunk.
* **Dual-Text Schema:** Original wording is preserved in `text` for user display and LLM context, while normalized text is generated in `embed_text` strictly for vector generation.
* **Vector Store Compatibility:** Metadata contains exclusively flat scalar values (`str`, `int`, `float`, `bool`).
* **Deterministic Hashing:** `content_hash` tracks changes across text and metadata attributes to ensure proper cache invalidation.

---

## 2. File Updates & Additions Map

The following files were created or modified in the `bankwise_rag` package:

```text
bankwise-rag-stage1/
├── bankwise_rag/
│   ├── schema.py          # UPDATED: Added RawChunk and Chunk dataclasses
│   ├── chunking.py        # CREATED: Stage 2 chunking engine & splitter
│   ├── metadata.py        # CREATED: Stage 3 metadata enrichment & hashing
│   └── cli.py             # UPDATED: Added `chunk` command pipeline
└── tests/
    └── test_stages_2_3.py # CREATED: 4 unit tests for Stage 2 & 3 rules
```

---

## 3. Detailed Code Modifications

### 3.1. Dataclass Extensions (`bankwise_rag/schema.py`)

Extended `schema.py` to define the output contracts for Stage 2 (`RawChunk`) and Stage 3 (`Chunk`).

* **`RawChunk`**: Holds intermediate chunk text and section/FAQ heading.
* **`Chunk`**: Holds deterministic `chunk_id`, display `text`, normalized `embed_text`, and flat scalar `metadata` dictionary.

### 3.2. Stage 2: Prepare Chunks (`bankwise_rag/chunking.py`)

Implemented `chunk_record(record: SourceRecord) -> List[RawChunk]` with the following logic:

* **Target Size:** `MAX_CHUNK_CHARS = 900` characters, `OVERLAP_CHARS = 120` characters.
* **FAQ Handling:** Merges `Question` and `Answer` into one chunk. If answer length exceeds maximum limit, splits by lines/sentences and prefixes every sub-chunk with `السؤال: <Question>\nالإجابة: `.
* **Policy Handling:** Splits document sections while prefixing each chunk with `<Document Title> — <Heading>`.

### 3.3. Stage 3: Add Metadata (`bankwise_rag/metadata.py`)

Implemented `make_chunks(record: SourceRecord, raw_chunks: List[RawChunk]) -> List[Chunk]`:

* **Deterministic Chunk ID:** Formatted as `<file_stem>::<record_id>::<chunk_index>`.
* **Normalized Embedding Text:** Applies `normalize_text(raw.text)` to generate `embed_text`.
* **MD5 Hashing:** Computes `content_hash` over text + metadata dict using JSON key sorting.
* **Scalar Metadata Guarantee:** Ensures all metadata entries are enforced as string, integer, float, or boolean.

### 3.4. Test Suite (`tests/test_stages_2_3.py`)

Created automated test cases covering:

1. `test_faq_chunking_preserves_qa`: Validates FAQ Q&A retention.
2. `test_long_faq_repeats_question`: Validates question prefix repetition on long split answers.
3. `test_metadata_scalar_types_and_hashing`: Verifies scalar metadata data types and `chunk_id` format.
4. `test_content_hash_changes_on_metadata_edit`: Verifies cache invalidation when record metadata changes.

### 3.5. CLI Pipeline Integration (`bankwise_rag/cli.py`)

Added the `chunk` subcommand to process raw JSON files through Stages 1–3 and export JSONL files for Stage 4 indexing.

---

## 4. Verification Commands & Execution Logs

> These logs are from the first version of Stages 2 & 3 (4 tests, 19 chunks). See section 7 for the later fixes (34 tests, 22 chunks on the sample data).

### 4.1. Unit Test Suite Verification

**Command:**

```bash
python -m pytest tests/test_stages_2_3.py -v
```

**Output:**

```text
==================================== test session starts ====================================
platform win32 -- Python 3.14.4, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\HP\Desktop\bankwise-rag-stage1

tests/test_stages_2_3.py::test_faq_chunking_preserves_qa PASSED                      [ 25%]
tests/test_stages_2_3.py::test_long_faq_repeats_question PASSED                     [ 50%]
tests/test_stages_2_3.py::test_metadata_scalar_types_and_hashing PASSED             [ 75%]
tests/test_stages_2_3.py::test_content_hash_changes_on_metadata_edit PASSED         [100%]

===================================== 4 passed in 0.07s =====================================
```

### 4.2. End-to-End Pipeline Export Verification

**Command:**

```bash
python -m bankwise_rag.cli chunk --data data/sample --show 2 --out out/chunks.jsonl
```

**Output:**

```text
Ingested 18 records from 4 file(s).
Generated 19 total Chunk objects across Stage 2 & Stage 3.

{
  "chunk_id": "car_loan_policy_ar::car-policy::0",
  "text": "سياسة تمويل السيارات\n# شروط التقدم\nيجب ألا يقل عمر المتقدم عن 21 سنة...",
  "embed_text": "سياسة تمويل السيارات # شروط التقدم يجب الا يقل عمر المتقدم عن 21 سنة...",
  "metadata": {
    "record_id": "car-policy",
    "kind": "policy",
    "language": "ar",
    "category": "loans",
    "product": "car_loan",
    "source_file": "car_loan_policy_ar.json",
    "source_url": "https://example.com/sample-bank/car-loan-policy.pdf",
    "retrieved_date": "2026-11-05",
    "updated_at": "2026-08-15",
    "title": "سياسة تمويل السيارات",
    "heading": "سياسة تمويل السيارات",
    "chunk_index": 0,
    "chunk_count": 2,
    "content_hash": "50023aa4e2d5567ba7c7468a32d06734"
  }
}

Successfully exported 19 chunks -> out\chunks.jsonl
```

---

## 5. Recommended Git Commit Message

When committing these changes to your team repository, use the following structured commit message:

```bash
git add bankwise_rag/schema.py bankwise_rag/chunking.py bankwise_rag/metadata.py bankwise_rag/cli.py tests/test_stages_2_3.py STAGE_2_3_COMPLETION_REPORT.md
git commit -m "feat(rag): implement Stage 2 (chunking) and Stage 3 (metadata enrichment)

- Extend schema.py with RawChunk and Chunk dataclasses
- Add chunking.py for Q&A pair preservation and document heading context
- Add metadata.py for embed_text normalization, scalar metadata enforcement, and MD5 hashing
- Integrate 'chunk' command in CLI exporting to JSONL
- Add pytest test suite for Stages 2 & 3 (4/4 tests passing)"
git push origin main
```

---

## 6. Next Steps for Team Handover (Stage 4)

When resuming work for **Stage 4 (Embed & Index)**:

1. Install ChromaDB and Sentence Transformers (`pip install chromadb sentence-transformers`).
2. Implement `bankwise_rag/embeddings.py` (supporting BGE-M3 or `BAAI/bge-m3`).
3. Implement `bankwise_rag/index.py` for persistent vector indexing and update handling via `content_hash`.

---

## 7. Post-review fixes

* `loaders.py`: bilingual `{ar, en}` records are expanded to one record per language (ids get `-ar` / `-en`); nested or unexpected values are skipped with a warning instead of crashing.
* `chunking.py`: long sentences are no longer truncated (sentence -> word -> hard split, nothing lost); policy bodies are split on Markdown `#` headings with the heading in the chunk prefix and the metadata; line breaks are preserved; overlap can no longer push a chunk past the limit.
* `schema.py`: `Chunk` has only the 4 contract fields; `SourceRecord.sections` is typed as a list of (heading, text) pairs.
* `cli.py`: `chunk` prints loader warnings, refuses duplicate chunk ids or an empty result, and prints a NOTE when the synthetic sample data is used.
* Tests: 34 in total (11 new). Sample data now gives 22 chunks; `data/raw/banque_misr_faq.json` gives 212 chunks.
