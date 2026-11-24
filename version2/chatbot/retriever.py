"""RAG retrieval with an abstain gate.

    question --(bge-m3 embedding)--> vector --(Chroma cosine search)--> top-k chunks
             --(score >= MIN_SCORE?)--> hits  |  nothing above threshold -> abstain

This module wraps the existing `search_chunks.search()` (not modified) and adds:
  * an absolute DB path (the original uses a path relative to the current directory),
  * the threshold gate, with a clear reason when it abstains,
  * input hygiene (empty / very long queries),
  * an injectable `search_fn`, so tests can run without downloading the embedding model.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import config


@dataclass
class Hit:
    chunk_id: str
    id: str                      # FAQ entry id, e.g. "CAR-003"
    score: float                 # cosine similarity, higher = closer
    text: str                    # context_text: "[label]\nالسؤال: ...\nالإجابة: ..."
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    hits: list
    abstained: bool
    reason: str = ""             # "", "empty_query", "below_threshold", "no_results"
    top_score: float = 0.0
    threshold: float = 0.0


def _default_search_fn(query, k, product):
    from search_chunks import search          # resolved through sys.path set in config.py
    return search(query, k=k, product=product, min_score=0.0,
                  db_dir=str(config.CHROMA_DIR), collection=config.COLLECTION)


class Retriever:
    def __init__(self, search_fn: Optional[Callable] = None,
                 k: int = config.TOP_K, min_score: float = config.MIN_SCORE):
        self._search = search_fn or _default_search_fn
        self.k = k
        self.min_score = min_score

    def warmup(self):
        """Load the embedding model now (first load can take a while) instead of on the first customer."""
        self._search("مرحبا", 1, None)

    def retrieve(self, query: str, product: Optional[str] = None) -> RetrievalResult:
        query = (query or "").strip()[: config.MAX_USER_CHARS]
        if not query:
            return RetrievalResult([], True, "empty_query", 0.0, self.min_score)

        raw = self._search(query, self.k, product)          # list of dicts, best first
        if not raw:
            return RetrievalResult([], True, "no_results", 0.0, self.min_score)

        top = max(h["score"] for h in raw)
        hits = [Hit(h["chunk_id"], h["id"], h["score"], h["context_text"], h["metadata"])
                for h in raw if h["score"] >= self.min_score]
        if not hits:
            return RetrievalResult([], True, "below_threshold", top, self.min_score)
        return RetrievalResult(hits, False, "", top, self.min_score)
