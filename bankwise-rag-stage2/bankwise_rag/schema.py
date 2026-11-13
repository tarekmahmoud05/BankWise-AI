"""Shared document structure across stages.

SourceRecord is the output contract of Stage 1.
RawChunk is the intermediate chunk representation of Stage 2.
Chunk is the final chunk representation of Stage 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class SourceRecord:
    """One JSON record after normalisation. FAQ records have question+answer;
    policy records have title + body and/or sections."""

    record_id: str
    kind: str                       # "faq" | "policy"
    language: str                   # "ar" | "en"
    category: str = ""
    product: str = ""
    source_file: str = ""
    source_url: str = ""            # KPI K-07: must be present
    retrieved_date: str = ""        # KPI K-07: must be present
    updated_at: str = ""
    title: str = ""                 # FAQ: the question
    question: str = ""
    answer: str | None = None
    body: str | None = None
    sections: list[tuple[str, str]] = field(default_factory=list)   # (heading, text)


@dataclass
class RawChunk:
    """Intermediate chunk produced by Stage 2 (chunking)."""
    text: str
    heading: str = ""


@dataclass
class Chunk:
    """Final chunk contract produced by Stage 3 (metadata & normalization)."""
    chunk_id: str
    text: str
    embed_text: str
    metadata: Dict[str, Any]

