"""Stage 4 - Ingest pipeline: JSON files -> chunks -> embeddings -> Chroma (idempotent)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .chunking import chunk_record
from .embeddings import NormalizingEmbeddings
from .index import SyncReport, open_store, reset_index, sync_chunks
from .loaders import LoadReport, load_directory
from .metadata import make_chunks
from .schema import Chunk


class IngestError(RuntimeError):
    pass


@dataclass
class IngestReport:
    records: int
    chunks: int
    warnings: int
    sync: SyncReport
    reset: bool


def build_chunks(data_dir: Path) -> Tuple[List[Chunk], LoadReport]:
    records, rep = load_directory(data_dir)
    if rep.errors:
        raise IngestError("; ".join(rep.errors))
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(make_chunks(record, chunk_record(record)))
    if not chunks:
        raise IngestError(f"no chunks produced from {data_dir} (all records skipped?)")
    dups = [cid for cid, n in Counter(c.chunk_id for c in chunks).items() if n > 1]
    if dups:
        raise IngestError(f"{len(dups)} duplicate chunk_id(s), e.g. {dups[:3]}")
    return chunks, rep


def ingest(data_dir: Path, index_dir: Path, embeddings: NormalizingEmbeddings, reset: bool = False,
           progress: Optional[Callable[[str], None]] = None) -> IngestReport:
    chunks, rep = build_chunks(data_dir)          # fail before touching the index
    did_reset = reset_index(index_dir) if reset else False
    store = open_store(index_dir, embeddings)
    sync = sync_chunks(store, chunks, progress=progress)
    return IngestReport(records=rep.records, chunks=len(chunks), warnings=len(rep.warnings), sync=sync, reset=did_reset)
