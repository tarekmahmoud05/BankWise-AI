"""
BankWise AI — Stage 3: Add Metadata (metadata.py)

Constructs production Chunk objects with:
1. Normalized embedding text (embed_text) created via normalize_text().
2. Flat scalar metadata dictionary (compatible with vector stores like ChromaDB).
3. Deterministic chunk ID (<file_stem>::<record_id>::<index>).
4. MD5 content hash over text and metadata attributes.
"""

import hashlib
import json
from pathlib import Path
from typing import List
from bankwise_rag.schema import SourceRecord, RawChunk, Chunk
from bankwise_rag.normalize import normalize_text


def compute_content_hash(text: str, metadata: dict) -> str:
    """Computes deterministic MD5 hash over chunk text and metadata."""
    serialized = json.dumps({"text": text, "metadata": metadata}, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def make_chunks(record: SourceRecord, raw_chunks: List[RawChunk]) -> List[Chunk]:
    """Converts RawChunks and SourceRecord into final Chunk objects with metadata."""
    file_stem = Path(record.source_file).stem if record.source_file else "unknown"
    chunk_count = len(raw_chunks)
    chunks: List[Chunk] = []

    for idx, raw in enumerate(raw_chunks):
        chunk_id = f"{file_stem}::{record.record_id}::{idx}"
        
        # Enforce scalar metadata (str, int, float, bool only)
        metadata = {
            "record_id": str(record.record_id or ""),
            "kind": str(record.kind or ""),
            "language": str(record.language or ""),
            "category": str(record.category or ""),
            "product": str(record.product or ""),
            "source_file": str(record.source_file or ""),
            "source_url": str(record.source_url or ""),
            "retrieved_date": str(record.retrieved_date or ""),
            "updated_at": str(record.updated_at or ""),
            "title": str(record.title or ""),
            "heading": str(raw.heading or ""),
            "chunk_index": int(idx),
            "chunk_count": int(chunk_count)
        }

        # Calculate Hash after creating scalar dict
        metadata["content_hash"] = compute_content_hash(raw.text, metadata)

        # Normalize text for vector embedding representation
        embed_text = normalize_text(raw.text)

        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=raw.text,
            embed_text=embed_text,
            metadata=metadata
        ))

    return chunks