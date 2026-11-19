# BankWise AI — How the RAG works and how the LLM layer plugs into it

This guide explains (1) what your existing RAG does, step by step, (2) what the new LLM layer
(`version2/chatbot/`) adds, (3) how to run and verify it, and (4) what is still open.

---

## 1. The big picture

A bank chatbot must do **two different jobs**, and they must not be mixed up:

| Job | Example question | Who is allowed to produce the facts? |
|---|---|---|
| **Explain** bank information | "إيه المستندات المطلوبة لقرض السيارة؟" | **RAG**: the answer must come from the bank's FAQ text |
| **Decide / calculate** | "أنا 30 سنة ومرتبي 15000، أقدر آخد قرض 200 ألف؟" | **Rule engine** (`eligibility/`): deterministic Python, no LLM |

The LLM is the **conversation layer**. It understands the customer's words, decides which tool to call,
asks for missing data, and phrases the result in friendly Arabic. It is **never** the source of a fee, a rate,
a limit, an eligibility decision, or an installment amount.

```
                      ┌────────────────────────── version2/chatbot/ ──────────────────────────┐
 customer message ──► │  agent.py : memory + tool loop + guards                               │
                      │      │                                                                │
                      │      ▼                                                                │
                      │  LLM (llm_clients.py)  ── "which tool do I need?" ──┐                 │
                      │      ▲                                              ▼                 │
                      │      │ tool result                      ┌───────────┴────────────┐    │
                      │      └──────────────────────────────────┤ agent_tools.py         │    │
                      │                                         │  search_faq ──► retriever.py ─► Chroma (240 chunks)
                      │                                         │  check_*_eligibility ─► eligibility/ rule engine
                      │                                         │  estimate_payment    ─► eligibility/calculator.py
                      │                                         └────────────────────────┘    │
                      │  guards (in code): abstain · sources from metadata · bank disclaimer  │
                      └───────────────────────────────────────────────────────────────────────┘
 reply (+ sources, + disclaimer) ◄──
```

---

## 2. How your RAG works (what already existed)

RAG = **R**etrieval-**A**ugmented **G**eneration: *first look up the relevant text, then let the LLM write
the answer using only that text.* Your pipeline has an **offline part** (done once) and an **online part**
(done for every question).

### Offline part — building the knowledge base

**Step 1 — Source data.** Five FAQ JSON files in `version2/data/` (current account, debit cards,
personal loans, car loans, general FAQ). Each entry has an id (e.g. `CAR-003`), a question, an answer,
a category and a source document.

**Step 2 — Chunking** (`chunking/prepare_chunks.py`). One chunk = one question + its answer. Answers are
never edited. Each chunk gets two texts:
- `embed_text` = `[قروض السيارات] إيه المستندات…؟` → the **question only**, used for matching.
- `context_text` = `[label] السؤال: … الإجابة: …` → the **full Q&A**, shown to the LLM.

Why embed only the question? Customers also ask *questions*, so question-to-question similarity is more
reliable than question-to-long-answer similarity. Calculator-generated entries (`LOAN-calc-*`) are
**excluded**: numbers belong to the rule engine, not to RAG.

**Step 3 — Embedding** (`embedding/index_chunks.py`). The model `BAAI/bge-m3` turns each `embed_text`
into a vector of 1,024 numbers. Texts with similar *meaning* get vectors pointing in similar directions,
even if the words differ ("عايز قرض" ≈ "أريد تمويل شخصي"). Vectors are normalized, so
cosine similarity = a number between −1 and 1 (higher = more similar).

**Step 4 — Index in ChromaDB.** For each chunk Chroma stores: the vector, the `context_text` (as the
"document"), and metadata (`id`, `product`, `source_document`, `accessed` date, `needs_review`…).
Your shipped index (`embedding/chroma_db`, collection `bm_faq`) holds **240 Arabic chunks**.
English chunks exist (`chunking/chunks_en.json`) but are **not indexed yet**.

### Online part — answering one question

**Step 5 — Embed the question** with the *same* model (`search_chunks.py`).
**Step 6 — Nearest-neighbour search**: Chroma returns the `k` chunks whose vectors are closest.
**Step 7 — Threshold gate** (new, `retriever.py`): keep only chunks with score ≥ `MIN_SCORE`.
If none pass → **abstain** ("I don't know") instead of guessing.
**Step 8 — Generation**: hand the surviving chunks to the LLM, which writes the answer *from those chunks only*.

Steps 1–7 existed. **Step 8 is the missing piece this layer adds.**

---

## 3. What the LLM layer adds, file by file

All new code is in `version2/chatbot/`. Nothing in your existing folders was modified
(only `requirements.txt` was edited, to enable the LLM SDKs).

| File | Role |
|---|---|
| `config.py` | Paths, thresholds, model names, fixed Arabic texts. Reads `version2/.env`. |
| `retriever.py` | Wraps your `search_chunks.search()` and adds the **abstain gate**. |
| `llm_clients.py` | One interface, two adapters: `AnthropicLLM` (Claude) and `OpenAICompatLLM` (OpenAI **and Gemini** through Google's OpenAI-compatible endpoint). Switch with `LLM_PROVIDER`. |
| `agent_tools.py` | Defines the 4 tools the LLM can call: `search_faq` (new) + your existing `check_personal_loan_eligibility`, `check_car_loan_eligibility`, `estimate_payment` (reused from `eligibility/tools.py`). |
| `prompts.py` | The system prompt (the LLM's rulebook). |
| `agent.py` | The orchestrator: session memory, tool-use loop, code-level guards. |
| `cli.py` / `api.py` | Terminal chat / FastAPI backend (`/chat`, `/reset`, `/health`). |
| `smoke_test.py` | Live behavioural check (real model + real LLM). |
| `tests/` | 30 offline tests. |

### 3.1 The tool-use loop (the heart of the integration)

Modern LLMs can *request* a function call. You describe the functions (name, description, JSON schema);
the model replies "please run `search_faq` with `query=...`"; **your code runs it** and returns the result;
the model then continues. The model never touches Chroma or the rule engine directly.

For one customer message, `agent.py` does:

1. Clean the input (empty → polite prompt, no LLM call; >1000 chars → truncated).
2. Send `system prompt + history + message + tool list` to the LLM.
3. If the LLM answers with text → go to step 5.
4. If it requests tools → run them via `ToolRunner`, append the results, go back to step 2
   (max 5 rounds, so a confused model can't loop forever).
5. Apply the **guards** below, save the clean turn into session memory, return the reply.

Example — *"ايه مستندات قرض العربية؟"*:
`LLM → search_faq("المستندات المطلوبة لقرض السيارة")` → retriever returns chunk `CAR-003` wrapped as
`<source ref="S1" score="0.83">…</source>` → `LLM → "تحتاج … [S1]"` → code strips `[S1]` and appends
`المصادر: CAR-003 — قرض_السيارة.pdf (2026-10-22~)`.

Example — *"عندي 30 سنة ومرتبي 15000 وعايز قرض شخصي 200 ألف على 60 شهر"*:
`LLM → check_personal_loan_eligibility(age=30, net_monthly_income=15000, loan_amount=200000, …)` →
rule engine returns `LIKELY_ELIGIBLE, 5,495.39/month` → LLM words it → code appends the bank's own
disclaimer verbatim. If a field is missing the engine returns `NEED_MORE_INFO` and the LLM asks for exactly
those fields; the next customer message triggers a second call.

Why the LLM calls `search_faq` itself instead of us searching with the raw message: it **rewrites the
query as a standalone Arabic question**. That solves two known weak spots in one move — English questions
(the index is Arabic) and follow-ups like "وإيه المصاريف؟" (it resolves "it" from the conversation).

### 3.2 The rules enforced **in code**, not only in the prompt

A prompt is a request; code is a guarantee. These do not depend on the LLM behaving:

| Guard | What the code does |
|---|---|
| **Abstain** | If the model searched, nothing passed the threshold, and no calculator was used → the reply is replaced by the fixed fallback text, whatever the LLM wrote. |
| **Sources from metadata** | The LLM only *cites* markers like `[S1]`. The "المصادر" footer is built from the chunk's metadata. Invented markers are ignored. |
| **Disclaimer** | Any eligibility/estimate result → the bank's own disclaimer text (from the rules JSON) is appended verbatim. |
| **Prompt-injection defence** | `<` and `>` inside retrieved text are escaped so a chunk can't close the `<source>` tag or fake instructions; the prompt says source text is data. |
| **Errors** | Any exception → safe Arabic/English message, no stack trace, no key leakage. |
| **Bounded loop, bounded input, bounded memory** | 5 tool rounds, 1000 chars, last 12 messages per session. |
| **No personal data in logs** | Logs contain tool names, scores, latency, token counts — never message text. |

### 3.3 The retrieval threshold (important!)

`MIN_SCORE` (default **0.50**) decides "relevant enough". It is a **starting guess**. On your index,
chunks of *unrelated* products have a median cosine of 0.43 (90th percentile 0.53), so a low threshold would let
weakly related text through. Calibrate it with your own tool:

```bash
cd version2/embedding/eval_real
python eval_retrieval.py          # prints top-1 scores for in-scope vs out-of-scope questions
```
Pick a value that rejects the `out_of_scope` / `tool` questions but keeps the correct `faq` hits, then set
`BANKWISE_MIN_SCORE` in `.env`. Re-calibrate whenever the model, the chunks, or the data change.

---

## 4. Run it — step by step

All commands run from the `version2/` folder. Python 3.10+.

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # includes sentence-transformers, chromadb, anthropic, ...
pip install openai                                       # only if you use OpenAI or Gemini

# 2. choose ONE provider and put its key in .env
cp .env.example .env                                     # then edit .env
#   LLM_PROVIDER=anthropic   +  ANTHROPIC_API_KEY=...
#   LLM_PROVIDER=gemini      +  GEMINI_API_KEY=...       (needs: pip install openai)
#   LLM_PROVIDER=openai      +  OPENAI_API_KEY=...

# 3. offline tests (no key, no internet, no model download)       -> expect: 30 passed
python -m pytest chatbot/tests -q

# 4. live smoke test: real bge-m3 + real LLM + real Chroma        -> expect: 7/7 checks passed
python -m chatbot.smoke_test
#   FIRST RUN downloads BAAI/bge-m3 (~2 GB) from Hugging Face: needs internet and a few minutes.

# 5. chat in the terminal
python -m chatbot.cli --debug          # --debug shows tools used, retrieval score, latency, tokens

# 6. or start the API
uvicorn chatbot.api:app --reload
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"session_id":"u1","message":"ازاي اكلم خدمة العملاء؟"}'
```

Model names are in `chatbot/config.py → DEFAULT_MODELS` (or set `LLM_MODEL` in `.env`). Check your
provider's current model list; names change over time.

### How to read `--debug` output
- `tools=['search_faq']`, `top_score≥threshold` → answered from the FAQ.
- `abstained=True reason=no_knowledge` → nothing relevant found; fixed fallback sent.
- `tools=['check_personal_loan_eligibility']` → decision came from the rule engine.
- `reason=llm_error` → API/network problem; check the key, model name, and internet.

---

## 5. What was verified, and what was not

Verified while building this (sandbox without an API key and without access to Hugging Face):
- 30 offline tests pass; deliberately breaking the abstain guard, the disclaimer, and the tag escaping each made a test fail (so the tests really guard those rules).
- Both provider adapters were exercised through the **real Anthropic and OpenAI SDKs** against a mock HTTP server (request format, tool-call parsing, token counts).
- The retriever, Chroma query, metadata, rule engine, agent loop, guards, and `smoke_test.py` ran end-to-end on your **real index and real rule files**, using stored chunk vectors as query vectors and a scripted stand-in for the LLM.

**Not verified (could not be, here):** a real call to an LLM, the real `bge-m3` embedding of new questions, and
the answer quality / best threshold on real customer phrasing. Step 4 above (the smoke test) is how you verify these on your machine.

---

## 6. Known gaps and suggested next steps

1. **Calibrate `MIN_SCORE`** (section 3.3). Highest priority.
2. **English is not indexed.** Run `index_chunks.py` with `chunks_en.json` too, or rely on the LLM translating queries to Arabic (already in the tool description).
3. **Source labels.** The label uses `source_document`. Some entries carry a file name that looks unrelated to the topic (e.g. `CON-001` → `BM_services_guide_for_Egyptians_residing_abroad.pdf`). Worth checking the data; the footer will show whatever the metadata says. `accessed` dates marked `~` are estimates from `prepare_chunks.py`.
4. **No-tool answers.** If the LLM answers without calling any tool (greetings are fine), nothing forces grounding; the prompt tells it to always search first. If you want a hard guarantee, force a tool call on the first round (`tool_choice` = any/required) and handle small talk separately.
5. **Reranker / hybrid search** (your `instructions.txt` ideas 2–3) if hit@1 is low after calibration.
6. **PII masking** of incoming messages is not implemented (the prompt tells the bot not to ask for IDs/cards/OTP, and logs contain no text).
7. **Sessions are in memory.** Fine for one server process; use Redis/DB before running several workers.
8. **Rate limiting / auth** on the API (`slowapi` is already in requirements).
