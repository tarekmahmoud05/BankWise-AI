"""Stage 1 settings: only where the JSON files live."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent


def resolve_data_dir(explicit: Optional[Path] = None) -> Path:
    """Explicit path > $BANKWISE_DATA_DIR > data/raw (if it has JSON) > data/sample (demo data)."""
    if explicit:
        return Path(explicit)
    if os.environ.get("BANKWISE_DATA_DIR"):
        return Path(os.environ["BANKWISE_DATA_DIR"])
    raw = ROOT / "data" / "raw"
    return raw if any(raw.glob("*.json*")) else ROOT / "data" / "sample"
