"""Settings: where the JSON data lives (stage 1) and where the Chroma index / embedder come from (stage 4)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

EMBEDDING_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "bankwise_kb"

# Defaults used by the root scripts (ingest.py / eval_retrieval.py) so you don't have to type them.
DEFAULT_LANGUAGE = "ar"                                   # eval: search only Arabic records ("all" = no filter)
EVAL_QUESTIONS = ROOT / "eval" / "eval_questions_bankwise.json"


def resolve_data_dir(explicit: Optional[Path] = None) -> Path:
    """Explicit path > $BANKWISE_DATA_DIR > data/raw (if it has JSON) > data/sample (demo data)."""
    if explicit:
        return Path(explicit)
    if os.environ.get("BANKWISE_DATA_DIR"):
        return Path(os.environ["BANKWISE_DATA_DIR"])
    raw = ROOT / "data" / "raw"
    return raw if any(raw.glob("*.json*")) else ROOT / "data" / "sample"


def resolve_index_dir(explicit: Optional[Path] = None) -> Path:
    """Explicit path > $BANKWISE_INDEX_DIR > <project>/chroma_db."""
    if explicit:
        return Path(explicit)
    if os.environ.get("BANKWISE_INDEX_DIR"):
        return Path(os.environ["BANKWISE_INDEX_DIR"])
    return ROOT / "chroma_db"


def default_embedder() -> str:
    """'bge-m3' (real model) or 'hash' (offline test double). Override with $BANKWISE_EMBEDDER."""
    return os.environ.get("BANKWISE_EMBEDDER", "bge-m3")
