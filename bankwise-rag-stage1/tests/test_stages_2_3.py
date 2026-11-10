"""
BankWise AI — Unit Test Suite for Stages 2 & 3
Run with `python -m pytest tests/test_stages_2_3.py -v`
"""

import pytest
from bankwise_rag.schema import SourceRecord
from bankwise_rag.chunking import chunk_record
from bankwise_rag.metadata import make_chunks, compute_content_hash


def test_faq_chunking_preserves_qa():
    rec = SourceRecord(
        record_id="CON-001",
        kind="faq",
        language="ar",
        category="contact_channels",
        product="general",
        source_file="BM_services_guide.pdf",
        question="كيف أتواصل مع خدمة العملاء؟",
        answer="يمكنك الاتصال بمركز الاتصال على 19888."
    )
    raw = chunk_record(rec)
    assert len(raw) == 1
    assert "السؤال: كيف أتواصل مع خدمة العملاء؟" in raw[0].text
    assert "الإجابة: يمكنك الاتصال بمركز الاتصال على 19888." in raw[0].text


def test_long_faq_repeats_question():
    long_answer = "تفاصيل الخدمة: " + ("هذا نص مكرر لاختبار تقسيم الإجابات الطويلة. " * 30)
    rec = SourceRecord(
        record_id="LONG-001",
        kind="faq",
        language="ar",
        category="general",
        product="test",
        source_file="test.pdf",
        question="ما هي الشروط والتفاصيل الكاملة؟",
        answer=long_answer
    )
    raw = chunk_record(rec)
    assert len(raw) > 1
    for chunk in raw:
        assert "السؤال: ما هي الشروط والتفاصيل الكاملة؟" in chunk.text


def test_metadata_scalar_types_and_hashing():
    rec = SourceRecord(
        record_id="ACC-001",
        kind="faq",
        language="ar",
        category="accounts",
        product="savings",
        source_file="BM_services_guide.pdf",
        source_url="https://www.banquemisr.com",
        retrieved_date="2026-01-15",
        question="ما مميزات حسابات التوفير؟",
        answer="تمنح عائداً تنافسياً."
    )
    raw = chunk_record(rec)
    chunks = make_chunks(rec, raw)
    
    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_id == "BM_services_guide::ACC-001::0"
    
    for key, val in c.metadata.items():
        assert isinstance(val, (str, int, float, bool)), f"Non-scalar metadata key: {key}"

    assert c.embed_text != ""
    assert "content_hash" in c.metadata


def test_content_hash_changes_on_metadata_edit():
    rec1 = SourceRecord(
        record_id="ACC-001", kind="faq", language="ar", category="accounts",
        product="savings", source_file="doc.pdf", question="سؤال؟", answer="إجابة."
    )
    rec2 = SourceRecord(
        record_id="ACC-001", kind="faq", language="ar", category="cards",
        product="savings", source_file="doc.pdf", question="سؤال؟", answer="إجابة."
    )
    raw1 = chunk_record(rec1)
    raw2 = chunk_record(rec2)
    c1 = make_chunks(rec1, raw1)[0]
    c2 = make_chunks(rec2, raw2)[0]
    
    assert c1.metadata["content_hash"] != c2.metadata["content_hash"]