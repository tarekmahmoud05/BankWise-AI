"""
Stage 4 - Chroma index through LangChain, with idempotent sync.

Rules (see RAG_STAGES_2-5_HANDOFF.md, section 6):
  1. Cosine space, persistent client, ONE collection.
  2. The embedder name is stored on the collection; opening it with another embedder raises
     EmbedderMismatchError (mixed vector spaces ruin retrieval silently). The fix is `ingest --reset`.
  3. Ingest is idempotent: unchanged chunks (same content_hash) are skipped, edited ones are replaced,
     chunks that disappeared from the data folder are deleted. Only changed chunks are embedded.
  4. Upsert in batches of 64.
Stored document = original chunk text. Vectors come from the normalised text (see embeddings.py).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma

from .config import COLLECTION_NAME
from .embeddings import NormalizingEmbeddings
from .schema import Chunk

BATCH_SIZE = 64


class EmbedderMismatchError(RuntimeError):
    pass


@dataclass
class SyncReport:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0

    @property
    def total(self) -> int:
        return self.added + self.updated + self.unchanged


def _client(index_dir: Path) -> "chromadb.ClientAPI":
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(index_dir), settings=Settings(anonymized_telemetry=False))


def _collection_names(client) -> List[str]:
    return [c if isinstance(c, str) else c.name for c in client.list_collections()]


def reset_index(index_dir: Path, collection: str = COLLECTION_NAME) -> bool:
    """Delete the collection through the client (no file deletion, so no Windows file-lock problems)."""
    client = _client(index_dir)
    if collection in _collection_names(client):
        client.delete_collection(collection)
        return True
    return False


def open_store(index_dir: Path, embeddings: NormalizingEmbeddings, collection: str = COLLECTION_NAME) -> Chroma:
    client = _client(index_dir)
    if collection in _collection_names(client):
        stored = (client.get_collection(collection).metadata or {}).get("embedder")
        if stored != embeddings.name:
            raise EmbedderMismatchError(
                f"index was built with embedder '{stored}' but you opened it with '{embeddings.name}'. "
                "Re-run ingest with --reset to rebuild it with the new embedder.")
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        client=client,
        collection_metadata={"embedder": embeddings.name},
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def _batches(items: List[Any], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def sync_chunks(store: Chroma, chunks: List[Chunk], batch_size: int = BATCH_SIZE,
                progress: Optional[Callable[[str], None]] = None) -> SyncReport:
    """Make the collection contain exactly `chunks` (matched by chunk_id), touching as little as possible."""
    say = progress or (lambda _msg: None)
    existing = store.get(include=["metadatas"])
    old: Dict[str, str] = {i: (m or {}).get("content_hash", "") for i, m in zip(existing["ids"], existing["metadatas"])}
    new: Dict[str, Chunk] = {c.chunk_id: c for c in chunks}

    stale = [i for i in old if i not in new]
    to_add = [c for i, c in new.items() if i not in old]
    to_update = [c for i, c in new.items() if i in old and old[i] != c.metadata["content_hash"]]
    report = SyncReport(added=len(to_add), updated=len(to_update), deleted=len(stale),
                        unchanged=len(new) - len(to_add) - len(to_update))

    for batch in _batches(stale, batch_size):
        store.delete(ids=batch)
    todo = to_add + to_update
    done = 0
    for batch in _batches(todo, batch_size):
        store.add_texts(texts=[c.text for c in batch], metadatas=[c.metadata for c in batch],
                        ids=[c.chunk_id for c in batch])           # upsert: replaces edited chunks
        done += len(batch)
        say(f"  embedded {done}/{len(todo)}")
    return report


def index_stats(index_dir: Path, collection: str = COLLECTION_NAME) -> Optional[Dict[str, Any]]:
    """Read-only summary. Does not load any embedding model. None if the collection does not exist."""
    client = _client(index_dir)
    if collection not in _collection_names(client):
        return None
    col = client.get_collection(collection)
    metas = col.get(include=["metadatas"])["metadatas"] or []
    return {
        "embedder": (col.metadata or {}).get("embedder"),
        "chunks": col.count(),
        "language": dict(Counter(m.get("language", "") for m in metas)),
        "kind": dict(Counter(m.get("kind", "") for m in metas)),
        "source_file": dict(Counter(m.get("source_file", "") for m in metas)),
        "missing_source_url": sum(1 for m in metas if not m.get("source_url")),
        "missing_retrieved_date": sum(1 for m in metas if not m.get("retrieved_date")),
    }


def search(store: Chroma, query: str, k: int = 4, language: Optional[str] = None) -> List[Tuple[Any, float]]:
    """Debug helper. Returns [(Document, similarity)], similarity = 1 - cosine distance. (Stage 5 owns real retrieval.)"""
    flt = {"language": language} if language else None
    hits = store.similarity_search_with_score(query, k=k, filter=flt)
    return [(doc, 1.0 - dist) for doc, dist in hits]
