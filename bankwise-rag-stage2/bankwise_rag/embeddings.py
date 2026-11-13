"""
Stage 4 - Embedders (LangChain `Embeddings` objects).

Every embedder returned by build_embeddings() is a NormalizingEmbeddings: it applies normalize_text()
to documents AND queries right before embedding. The Chroma index therefore stores the ORIGINAL text
(what the customer reads) while the vectors are computed from the normalised text (== Chunk.embed_text).

  bge-m3 : BAAI/bge-m3 through langchain-huggingface (dense vectors, normalised -> cosine)
  hash   : offline character-n-gram test double. LEXICAL, NOT SEMANTIC. For tests/CI only; never use it
           to report accuracy or to pick a score threshold.
"""
from __future__ import annotations

import hashlib
import math
from typing import List

from langchain_core.embeddings import Embeddings

from .config import EMBEDDING_MODEL
from .normalize import normalize_text


class NormalizingEmbeddings(Embeddings):
    """Wraps another Embeddings object; `name` identifies the vector space (stored on the collection)."""

    def __init__(self, inner: Embeddings, name: str):
        self.inner = inner
        self.name = name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.inner.embed_documents([normalize_text(t) for t in texts])

    def embed_query(self, text: str) -> List[float]:
        return self.inner.embed_query(normalize_text(text))


class HashingEmbeddings(Embeddings):
    """Deterministic character n-gram hashing embedder (signed hashing, L2-normalised). Lexical only."""

    def __init__(self, dims: int = 512, ngram: int = 3):
        self.dims = dims
        self.ngram = ngram

    def _embed(self, text: str) -> List[float]:
        t = f" {text} "
        vec = [0.0] * self.dims
        for i in range(max(1, len(t) - self.ngram + 1)):
            h = int.from_bytes(hashlib.blake2b(t[i:i + self.ngram].encode("utf-8"), digest_size=8).digest(), "big")
            vec[h % self.dims] += 1.0 if (h >> 20) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def build_embeddings(kind: str = "bge-m3", device: str | None = None, batch_size: int = 16) -> NormalizingEmbeddings:
    if kind == "hash":
        return NormalizingEmbeddings(HashingEmbeddings(), name="hash-ngram-v1")
    if kind != "bge-m3":
        raise ValueError(f"unknown embedder '{kind}' (use 'bge-m3' or 'hash')")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        import torch
    except ImportError as exc:                       # pragma: no cover - depends on the user's environment
        raise RuntimeError("bge-m3 needs: pip install -r requirements.txt (sentence-transformers, torch, "
                           f"langchain-huggingface). Original error: {exc}") from exc
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    inner = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},   # unit vectors -> cosine
    )
    return NormalizingEmbeddings(inner, name=f"hf:{EMBEDDING_MODEL}")
