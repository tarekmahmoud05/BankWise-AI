"""
BankWise AI — Stage 2: Prepare Chunks (chunking.py)

Transforms SourceRecord objects into RawChunk objects adhering to:
1. Q&A pair preservation for FAQs (repeating the question context if answers split).
2. Document title & heading context prefixing for policies.
3. Line and sentence boundary splitting with overlap.
"""

import re
from typing import List
from bankwise_rag.schema import SourceRecord, RawChunk

MAX_CHUNK_CHARS = 900
OVERLAP_CHARS = 120


def _split_text_by_lines(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> List[str]:
    """
    Splits long text into overlapping chunks, maintaining line structures.
    Falls back to sentence boundaries or hard character splits for long single lines.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return [text[:max_chars]] if text else []

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for line in lines:
        if current_len + len(line) + (1 if current_chunk else 0) <= max_chars:
            current_chunk.append(line)
            current_len += len(line) + (1 if len(current_chunk) > 1 else 0)
        else:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            overlap_lines: List[str] = []
            overlap_len = 0
            for prev_line in reversed(current_chunk):
                if overlap_len + len(prev_line) + 1 <= overlap:
                    overlap_lines.insert(0, prev_line)
                    overlap_len += len(prev_line) + 1
                else:
                    break
            current_chunk = overlap_lines + [line]
            current_len = sum(len(l) + 1 for l in current_chunk)

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    final_chunks: List[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            sentences = re.split(r'(?<=[.!?؟؛])\s+', chunk)
            sub_chunk = ""
            for s in sentences:
                if len(sub_chunk) + len(s) + (1 if sub_chunk else 0) <= max_chars:
                    sub_chunk = f"{sub_chunk} {s}".strip() if sub_chunk else s
                else:
                    if sub_chunk:
                        final_chunks.append(sub_chunk)
                    sub_chunk = s[:max_chars]
            if sub_chunk:
                final_chunks.append(sub_chunk)

    return final_chunks or [text[:max_chars]]


def chunk_record(record: SourceRecord) -> List[RawChunk]:
    """
    Transforms a SourceRecord into a list of RawChunk objects.
    - FAQ: Keeps Question and Answer together. Repeats question if answer is split.
    - Policy: Splits sections/body by headings with context prefixes.
    """
    raw_chunks: List[RawChunk] = []

    if record.kind == "faq":
        q_prefix = "السؤال:" if record.language == "ar" else "Question:"
        a_prefix = "الإجابة:" if record.language == "ar" else "Answer:"
        full_qa = f"{q_prefix} {record.question.strip()}\n{a_prefix} {record.answer.strip()}"

        if len(full_qa) <= MAX_CHUNK_CHARS:
            raw_chunks.append(RawChunk(text=full_qa, heading=record.question.strip()))
        else:
            prefix_header = f"{q_prefix} {record.question.strip()}\n{a_prefix} "
            available_chars = max(100, MAX_CHUNK_CHARS - len(prefix_header))
            answer_parts = _split_text_by_lines(record.answer.strip(), max_chars=available_chars, overlap=OVERLAP_CHARS)
            for part in answer_parts:
                chunk_text = f"{prefix_header}{part}"
                raw_chunks.append(RawChunk(text=chunk_text, heading=record.question.strip()))

    elif record.kind == "policy":
        doc_title = record.title.strip() if record.title else ""
        
        if record.sections:
            for heading, content in record.sections:
                heading_clean = heading.strip()
                prefix = f"{doc_title} — {heading_clean}" if doc_title and heading_clean else (doc_title or heading_clean)
                full_sec_text = f"{prefix}\n{content.strip()}" if prefix else content.strip()

                if len(full_sec_text) <= MAX_CHUNK_CHARS:
                    raw_chunks.append(RawChunk(text=full_sec_text, heading=heading_clean))
                else:
                    header_line = f"{prefix}\n" if prefix else ""
                    available_chars = max(100, MAX_CHUNK_CHARS - len(header_line))
                    parts = _split_text_by_lines(content.strip(), max_chars=available_chars, overlap=OVERLAP_CHARS)
                    for part in parts:
                        raw_chunks.append(RawChunk(text=f"{header_line}{part}", heading=heading_clean))
        else:
            body_text = (record.body or record.answer or "").strip()
            heading_clean = doc_title or "General Policy"
            prefix = doc_title if doc_title else ""
            
            if len(body_text) <= MAX_CHUNK_CHARS:
                full_text = f"{prefix}\n{body_text}" if prefix else body_text
                raw_chunks.append(RawChunk(text=full_text, heading=heading_clean))
            else:
                header_line = f"{prefix}\n" if prefix else ""
                available_chars = max(100, MAX_CHUNK_CHARS - len(header_line))
                parts = _split_text_by_lines(body_text, max_chars=available_chars, overlap=OVERLAP_CHARS)
                for part in parts:
                    raw_chunks.append(RawChunk(text=f"{header_line}{part}", heading=heading_clean))

    return raw_chunks