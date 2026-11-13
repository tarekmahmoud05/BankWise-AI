"""
BankWise AI — Stage 2: Prepare Chunks (chunking.py)

Transforms SourceRecord objects into RawChunk objects adhering to:
1. Q&A pair preservation for FAQs (repeating the question context if answers split).
2. Document title & heading context prefixing for policies (split on Markdown headings / sections).
3. Line, sentence and word-boundary splitting with overlap. No text is ever dropped.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from .schema import SourceRecord, RawChunk

MAX_CHUNK_CHARS = 900
OVERLAP_CHARS = 120

_SENTENCE_END = re.compile(r"(?<=[.!?؟؛])\s+")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")


def _hard_split(text: str, max_chars: int) -> List[str]:
    """Cut text into pieces <= max_chars, at a space when possible, else mid-word. Loses nothing."""
    out: List[str] = []
    t = text.strip()
    while len(t) > max_chars:
        cut = t.rfind(" ", 0, max_chars + 1)
        if cut <= 0:
            cut = max_chars
        out.append(t[:cut].rstrip())
        t = t[cut:].lstrip()
    if t:
        out.append(t)
    return out


def _split_long_line(line: str, max_chars: int) -> List[str]:
    """Break one oversized line: sentence boundaries first, then word boundary, then hard cut."""
    pieces: List[str] = []
    cur = ""
    for sent in _SENTENCE_END.split(line):
        if len(sent) > max_chars:
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.extend(_hard_split(sent, max_chars))
        elif not cur:
            cur = sent
        elif len(cur) + 1 + len(sent) <= max_chars:
            cur = f"{cur} {sent}"
        else:
            pieces.append(cur)
            cur = sent
    if cur:
        pieces.append(cur)
    return pieces


def _split_text_by_lines(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> List[str]:
    """
    Packs whole lines into chunks of <= max_chars with ~overlap chars carried into the next chunk.
    Lines stay on their own line (bullets / fee tables remain readable). A line longer than
    max_chars is first broken into sentences / words so nothing is truncated.
    """
    lines: List[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        lines.extend([line] if len(line) <= max_chars else _split_long_line(line, max_chars))
    if not lines:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for line in lines:
        added = len(line) + (1 if current else 0)
        if current_len + added <= max_chars:
            current.append(line)
            current_len += added
            continue

        chunks.append("\n".join(current))

        overlap_lines: List[str] = []
        for prev in reversed(current):
            candidate = [prev] + overlap_lines
            if len("\n".join(candidate)) <= overlap:
                overlap_lines = candidate
            else:
                break
        # the carried-over lines must never push the new chunk past the limit
        while overlap_lines and len("\n".join(overlap_lines)) + 1 + len(line) > max_chars:
            overlap_lines.pop(0)

        current = overlap_lines + [line]
        current_len = len("\n".join(current))

    if current:
        chunks.append("\n".join(current))
    return chunks


def _split_body_by_headings(body: str) -> List[Tuple[str, str]]:
    """Split Markdown-style text on '# heading' lines -> [(heading, text)]. No headings -> one section."""
    sections: List[Tuple[str, str]] = []
    heading = ""
    buf: List[str] = []
    for line in body.split("\n"):
        m = _MD_HEADING.match(line)
        if m:
            if "".join(buf).strip():
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((heading, "\n".join(buf).strip()))
    return sections


def _policy_chunks(doc_title: str, sections: List[Tuple[str, str]], default_heading: str = "") -> List[RawChunk]:
    out: List[RawChunk] = []
    for heading, content in sections:
        heading_clean = heading.strip()
        content = content.strip()
        if not content:
            continue
        prefix = f"{doc_title} — {heading_clean}" if doc_title and heading_clean else (doc_title or heading_clean)
        full_text = f"{prefix}\n{content}" if prefix else content
        label = heading_clean or default_heading

        if len(full_text) <= MAX_CHUNK_CHARS:
            out.append(RawChunk(text=full_text, heading=label))
        else:
            header_line = f"{prefix}\n" if prefix else ""
            available_chars = max(100, MAX_CHUNK_CHARS - len(header_line))
            for part in _split_text_by_lines(content, max_chars=available_chars, overlap=OVERLAP_CHARS):
                out.append(RawChunk(text=f"{header_line}{part}", heading=label))
    return out


def chunk_record(record: SourceRecord) -> List[RawChunk]:
    """
    Transforms a SourceRecord into a list of RawChunk objects.
    - FAQ: Keeps Question and Answer together. Repeats question if answer is split.
    - Policy: Splits sections (or Markdown headings in the body) with title/heading prefixes.
    """
    raw_chunks: List[RawChunk] = []

    if record.kind == "faq":
        q_prefix = "السؤال:" if record.language == "ar" else "Question:"
        a_prefix = "الإجابة:" if record.language == "ar" else "Answer:"
        question = record.question.strip()
        answer = (record.answer or "").strip()
        full_qa = f"{q_prefix} {question}\n{a_prefix} {answer}"

        if len(full_qa) <= MAX_CHUNK_CHARS:
            raw_chunks.append(RawChunk(text=full_qa, heading=question))
        else:
            prefix_header = f"{q_prefix} {question}\n{a_prefix} "
            available_chars = max(100, MAX_CHUNK_CHARS - len(prefix_header))
            for part in _split_text_by_lines(answer, max_chars=available_chars, overlap=OVERLAP_CHARS):
                raw_chunks.append(RawChunk(text=f"{prefix_header}{part}", heading=question))

    elif record.kind == "policy":
        doc_title = record.title.strip() if record.title else ""
        if record.sections:
            raw_chunks = _policy_chunks(doc_title, record.sections)
        else:
            body_text = (record.body or record.answer or "").strip()
            raw_chunks = _policy_chunks(doc_title, _split_body_by_headings(body_text),
                                        default_heading=doc_title or "General Policy")

    return raw_chunks
