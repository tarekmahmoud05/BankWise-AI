import json
from dataclasses import asdict

from bankwise_rag.cli import main
from bankwise_rag.config import ROOT
from bankwise_rag.loaders import load_directory
from conftest import write_json


def test_file_level_defaults_apply_and_record_overrides(tmp_path):
    write_json(tmp_path / "a.json", {
        "source_url": "https://x/1", "retrieved_date": "2026-01-01", "category": "loans", "language": "ar",
        "records": [{"id": "1", "question": "س", "answer": "ج"},
                    {"id": "2", "question": "س2", "answer": "ج2", "category": "fees"}]})
    recs, rep = load_directory(tmp_path)
    assert [r.category for r in recs] == ["loans", "fees"]
    assert recs[0].source_url == "https://x/1" and not rep.warnings


def test_plain_list_arabic_keys_and_aliases(tmp_path):
    write_json(tmp_path / "a.json", [{"id": "1", "السؤال": "ما الرسوم؟", "الإجابة": "مئة", "lang": "ar",
                                       "url": "u", "retrieved_at": "d"}])
    (r,), _ = load_directory(tmp_path)
    assert (r.question, r.answer, r.source_url, r.retrieved_date) == ("ما الرسوم؟", "مئة", "u", "d")


def test_jsonl_supported(tmp_path):
    (tmp_path / "a.jsonl").write_text(
        '{"id":"1","question":"q","answer":"a"}\n{"id":"2","question":"q2","answer":"a2"}\n', encoding="utf-8")
    recs, _ = load_directory(tmp_path)
    assert len(recs) == 2


def test_policy_record_with_body_and_with_sections(tmp_path):
    write_json(tmp_path / "a.json", [
        {"id": "p1", "title": "Policy", "content": "# A\ntext"},
        {"id": "p2", "title": "Policy 2", "sections": [{"heading": "H", "text": "t"}, "plain"]}])
    a, b = load_directory(tmp_path)[0]
    assert a.kind == b.kind == "policy" and a.body.startswith("# A")
    assert b.sections == [("H", "t"), ("", "plain")]


def test_missing_source_or_date_warns_k07(tmp_path):
    write_json(tmp_path / "a.json", [{"id": "1", "question": "q", "answer": "a"}])
    _, rep = load_directory(tmp_path)
    assert any("source_url" in w for w in rep.warnings) and any("retrieved_date" in w for w in rep.warnings)


def test_language_detected_when_absent(tmp_path):
    write_json(tmp_path / "a.json", [{"id": "1", "question": "ما الرسوم؟", "answer": "مئة"},
                                     {"id": "2", "question": "Fee?", "answer": "100"}])
    recs, _ = load_directory(tmp_path)
    assert [r.language for r in recs] == ["ar", "en"]


def test_missing_id_generated_and_stable(tmp_path):
    write_json(tmp_path / "a.json", [{"question": "q", "answer": "a"}])
    (r1,), _ = load_directory(tmp_path)
    (r2,), _ = load_directory(tmp_path)
    assert r1.record_id.startswith("auto-") and r1.record_id == r2.record_id


def test_duplicate_ids_are_renamed_not_overwritten(tmp_path):
    write_json(tmp_path / "a.json", [{"id": "1", "question": "q", "answer": "a"},
                                     {"id": "1", "question": "q2", "answer": "a2"}])
    recs, rep = load_directory(tmp_path)
    assert len({r.record_id for r in recs}) == 2 and any("duplicate" in w for w in rep.warnings)


def test_bad_json_is_reported_not_raised(tmp_path):
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    write_json(tmp_path / "ok.json", [{"id": "1", "question": "q", "answer": "a"}])
    recs, rep = load_directory(tmp_path)
    assert len(recs) == 1 and any("bad.json" in e for e in rep.errors)


def test_unusable_record_skipped_with_warning(tmp_path):
    write_json(tmp_path / "a.json", [{"id": "1", "foo": "bar"}, {"id": "2", "question": "q", "answer": "a"}])
    recs, rep = load_directory(tmp_path)
    assert len(recs) == 1 and any("skipped" in w for w in rep.warnings)


def test_empty_directory_is_an_error(tmp_path):
    _, rep = load_directory(tmp_path)
    assert rep.errors


def test_sample_data_loads_cleanly():
    recs, rep = load_directory(ROOT / "data" / "sample")
    assert len(recs) == 18 and not rep.warnings and not rep.errors
    assert all(r.source_url and r.retrieved_date for r in recs)


def test_cli_exports_jsonl_roundtrip(tmp_path):
    out = tmp_path / "docs.jsonl"
    assert main(["load", "--data", str(ROOT / "data" / "sample"), "--out", str(out)]) == 0
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 18 and {"record_id", "kind", "language", "source_url"} <= set(rows[0])
