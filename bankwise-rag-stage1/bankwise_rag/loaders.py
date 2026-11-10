"""Step 01 - Load & normalize.

Reads the team's JSON files and maps every record into SourceRecord. The loader is deliberately
tolerant about field names and file layout because the final JSON schema may still change:

  * a file may be a list of records, or an object {"records": [...], <file-level defaults>}
  * file-level defaults (source_url, retrieved_date, language, category, product) apply to every
    record unless the record sets its own value
  * field aliases are listed in FIELD_ALIASES (English and Arabic keys)
  * .json and .jsonl are both accepted
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .normalize import detect_language
from .schema import SourceRecord

FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "id": ("id", "record_id", "faq_id", "doc_id", "uid"),
    "question": ("question", "q", "السؤال", "سؤال"),
    "answer": ("answer", "a", "response", "الإجابة", "الاجابة", "جواب"),
    "title": ("title", "name", "heading", "العنوان"),
    "content": ("content", "text", "body", "description", "المحتوى", "النص"),
    "sections": ("sections", "parts"),
    "category": ("category", "topic", "الفئة", "التصنيف"),
    "product": ("product", "product_name", "loan_type", "المنتج"),
    "language": ("language", "lang"),
    "source_url": ("source_url", "url", "link", "source"),
    "retrieved_date": ("retrieved_date", "retrieval_date", "retrieved_at", "scraped_at"),
    "updated_at": ("updated_at", "last_updated", "update_date", "modified", "date"),
}
CONTAINER_KEYS = ("records", "faqs", "faq", "items", "data", "documents", "policies", "entries")
DEFAULT_KEYS = ("language", "category", "product", "source_url", "retrieved_date", "updated_at")


@dataclass
class LoadReport:
    files: int = 0
    records: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _pick(d: Dict[str, Any], name: str) -> str:
    for key in FIELD_ALIASES[name]:
        v = d.get(key)
        if v not in (None, "", []):
            return v if isinstance(v, (list, dict)) else str(v).strip()
    return ""


def _sections(raw: Any) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if not isinstance(raw, list):
        return out
    for s in raw:
        if isinstance(s, str) and s.strip():
            out.append(("", s.strip()))
        elif isinstance(s, dict):
            text = _pick(s, "content") or _pick(s, "answer")
            if text:
                out.append((_pick(s, "title"), text))
    return out


def _unwrap(data: Any) -> Tuple[List[dict], Dict[str, Any]]:
    """Return (records, file-level defaults)."""
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)], {}
    if isinstance(data, dict):
        for key in CONTAINER_KEYS:
            if isinstance(data.get(key), list):
                defaults = {k: v for k, v in data.items() if k != key}
                return [r for r in data[key] if isinstance(r, dict)], defaults
        return [data], {}                       # a single record
    return [], {}


def _read(path: Path) -> Any:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def load_file(path: Path, report: LoadReport) -> List[SourceRecord]:
    try:
        records, defaults = _unwrap(_read(path))
    except (OSError, ValueError) as exc:
        report.errors.append(f"{path.name}: cannot read ({exc})")
        return []

    default_vals = {k: _pick(defaults, k) for k in DEFAULT_KEYS}
    out: List[SourceRecord] = []
    seen: Dict[str, int] = {}

    for idx, raw in enumerate(records):
        q, a = _pick(raw, "question"), _pick(raw, "answer")
        title, content = _pick(raw, "title"), _pick(raw, "content")
        sections = _sections(raw.get("sections") or raw.get("parts"))
        if q and a:
            kind = "faq"
        elif title or content or sections:
            kind = "policy"
            if not content and not sections:
                report.warnings.append(f"{path.name}[{idx}]: policy record has no text, skipped")
                continue
        else:
            report.warnings.append(f"{path.name}[{idx}]: no question/answer/title/content, skipped")
            continue

        def get(name: str) -> str:
            return _pick(raw, name) or default_vals.get(name, "")

        rid = _pick(raw, "id")
        if not rid:
            rid = "auto-" + hashlib.sha1(f"{path.name}|{q or title}".encode()).hexdigest()[:10]
            report.warnings.append(f"{path.name}[{idx}]: no id, generated {rid}")
        if rid in seen:                          # never let two records share a chunk id
            seen[rid] += 1
            new = f"{rid}~{seen[rid]}"
            report.warnings.append(f"{path.name}: duplicate id '{rid}', renamed {new}")
            rid = new
        else:
            seen[rid] = 1

        rec = SourceRecord(
            record_id=rid, kind=kind,
            language=get("language") or detect_language(q + " " + a + " " + title + " " + content),
            category=get("category"), product=get("product"),
            source_file=path.name, source_url=get("source_url"),
            retrieved_date=get("retrieved_date"), updated_at=get("updated_at"),
            title=title or q, question=q, answer=a, body=content, sections=sections,
        )
        for missing in ("source_url", "retrieved_date"):       # KPI K-07
            if not getattr(rec, missing):
                report.warnings.append(f"{path.name}:{rid}: missing {missing} (KPI K-07)")
        out.append(rec)
    return out


def load_directory(data_dir: Path) -> Tuple[List[SourceRecord], LoadReport]:
    report = LoadReport()
    data_dir = Path(data_dir)
    files = sorted(p for p in data_dir.glob("*.json*") if p.suffix in (".json", ".jsonl"))
    if not files:
        report.errors.append(f"no .json/.jsonl files found in {data_dir}")
    records: List[SourceRecord] = []
    for p in files:
        report.files += 1
        records.extend(load_file(p, report))
    report.records = len(records)
    return records, report
