#!/usr/bin/env python3
"""
prepare_chunks.py
-----------------
Merges the five Banque Misr FAQ files into chunk files ready for embedding.

One chunk = one question + its answer (structure-based chunking).
Answer text is NEVER modified: the script only reads, adds metadata and writes.

Outputs (JSON arrays, UTF-8):
  chunks_ar.json                      -> Arabic chunks to index (calculator entries excluded)
  chunks_en.json                      -> English chunks (only entries that have English text)
  excluded_calculator_entries.json    -> LOAN-calc-* / CAR-calc-* (numbers go to the eligibility tool)

Usage:
  python prepare_chunks.py --input-dir ./faq_files --output-dir ./out
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

# One unified `product` per source file (category/topic are NOT reliable for this,
# e.g. `retail_finance` is used for personal finance in one file and mortgage in another).
PRODUCT_FILES = {
    "faq_current_account.json": dict(product="current_account", ar="الحساب الجاري", en="Current account"),
    "faq_debit_cards.json":     dict(product="debit_cards",     ar="بطاقات الخصم", en="Debit cards"),
    "faq_personal_loans.json":  dict(product="personal_loans",  ar="القروض الشخصية", en="Personal loans"),
    "faq_car_loans.json":       dict(product="car_loans",       ar="قروض السيارات", en="Car loans"),
}

# The general file mixes many topics: product = its category, except where the
# category name is misleading after RET-001..010 moved to personal loans.
GENERAL_FILE = "banque_misr_faq_v1_1.json"
GENERAL_OVERRIDES = {
    "retail_finance": dict(product="mortgage_finance", ar="التمويل العقاري", en="Mortgage finance"),
}

# ESTIMATED collection dates, used ONLY when an entry's `accessed` is null.
# These are NOT facts: they are the closest known date (the earliest one, to be
# conservative about stale rates). Change them here once you know the real dates.
ESTIMATED_ACCESS = {
    "faq_current_account.json": "2026-10-22",
    "faq_personal_loans.json":  "2026-10-22",   # only RET-001..010 are null
    "faq_car_loans.json":       "2026-10-22",
    GENERAL_FILE:               "2026-10-22",
}

LANGS = ("ar", "en")

# Optional metadata copied through when present
PASSTHROUGH = [
    "program", "card_type", "source_document_date", "translated",
    "generated_from", "moved_from", "note", "review_note",
]

# ----------------------------------------------------------------------------


def find_file(input_dir: Path, name: str) -> Path:
    """Match 'faq_x.json' even if the file has an upload prefix like '1794..._faq_x.json'."""
    for p in sorted(input_dir.glob("*.json")):
        if p.name == name or p.name.endswith("_" + name):
            return p
    sys.exit(f"File not found in {input_dir}: {name}")


def build_chunks(entry, meta, lang, source_name):
    q = entry["question"].get(lang)
    a = entry["answer"].get(lang) if isinstance(entry["answer"], dict) else None
    if not q or not a:
        return None
    label = meta["ar"] if lang == "ar" else meta["en"]
    q_word, a_word = ("السؤال", "الإجابة") if lang == "ar" else ("Question", "Answer")

    accessed = entry.get("accessed")
    estimated = False
    if not accessed:
        accessed = ESTIMATED_ACCESS.get(source_name)
        estimated = accessed is not None

    chunk = {
        "chunk_id": f"{entry['id']}:{lang}",
        "id": entry["id"],
        "lang": lang,
        "product": meta["product"],
        "product_label": label,
        "category": entry.get("category"),
        "topic": entry.get("topic"),
        "scope": entry.get("scope"),
        "needs_review": bool(entry.get("needs_review")),
        "accessed": accessed,
        "accessed_estimated": estimated,
        "source_json": source_name,
        "source_document": entry.get("source_file"),
        "tags": entry.get("tags", []),
        # Text to embed: product label + question only (answer kept out so the
        # similarity is question-to-question). Switch to `context_text` to test
        # "question + answer" embedding.
        "embed_text": f"[{label}] {q}",
        # Text to hand to the LLM after retrieval (also usable as embed text, option 2)
        "context_text": f"[{label}]\n{q_word}: {q}\n{a_word}: {a}",
        "question": q,
        "answer": a,
    }
    for k in PASSTHROUGH:
        if entry.get(k) is not None:
            chunk[k] = entry[k]
    return {k: v for k, v in chunk.items() if v is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks_ar, chunks_en, excluded = [], [], []

    # ---- general file (product comes from each entry's category)
    gp = find_file(in_dir, GENERAL_FILE)
    gdata = json.loads(gp.read_text(encoding="utf-8"))
    cat_labels = {c["id"]: c["label"] for c in gdata["categories"]}
    sources = [(GENERAL_FILE, gdata, None)] + [
        (name, json.loads(find_file(in_dir, name).read_text(encoding="utf-8")), meta)
        for name, meta in PRODUCT_FILES.items()
    ]

    for source_name, data, fixed_meta in sources:
        for entry in data["faqs"]:
            if fixed_meta is not None:
                meta = fixed_meta
            elif entry["category"] in GENERAL_OVERRIDES:
                meta = GENERAL_OVERRIDES[entry["category"]]
            else:
                lab = cat_labels[entry["category"]]
                meta = dict(product=entry["category"], ar=lab["ar"], en=lab["en"])

            is_calc = bool(entry.get("generated_from"))
            for lang in LANGS:
                ch = build_chunks(entry, meta, lang, source_name)
                if ch is None:
                    continue
                if is_calc:
                    ch["excluded_reason"] = "calculator-generated numbers: answered by the eligibility tool, not RAG"
                    excluded.append(ch)
                elif lang == "ar":
                    chunks_ar.append(ch)
                else:
                    chunks_en.append(ch)

    # ---- validation (fail loudly instead of writing a broken file)
    all_chunks = chunks_ar + chunks_en + excluded
    ids = Counter(c["chunk_id"] for c in all_chunks)
    dups = [k for k, v in ids.items() if v > 1]
    assert not dups, f"duplicate chunk_ids: {dups}"
    assert all(c["embed_text"].strip() and c["answer"].strip() for c in all_chunks), "empty text found"

    for fname, data in (("chunks_ar.json", chunks_ar), ("chunks_en.json", chunks_en),
                        ("excluded_calculator_entries.json", excluded)):
        (out_dir / fname).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- summary
    print(f"chunks_ar.json:                   {len(chunks_ar)}")
    print(f"chunks_en.json:                   {len(chunks_en)}")
    print(f"excluded_calculator_entries.json: {len(excluded)}")
    print("\nArabic chunks per product:")
    for k, v in Counter(c["product"] for c in chunks_ar).most_common():
        print(f"  {k:<24}{v}")
    print("\nneeds_review (Arabic chunks, by file):")
    by_file = {}
    for c in chunks_ar:
        if c["needs_review"]:
            by_file.setdefault(c["source_json"], []).append(c["id"])
    for f, lst in by_file.items():
        print(f"  {f} ({len(lst)}): {', '.join(lst)}")
    print(f"  TOTAL: {sum(len(v) for v in by_file.values())}")
    est = sum(1 for c in chunks_ar if c["accessed_estimated"])
    print(f"\nChunks with ESTIMATED accessed date: {est}")


if __name__ == "__main__":
    main()
