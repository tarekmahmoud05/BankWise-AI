"""Arabic-aware text normalisation (risks R-05 / R-06).

Used for EMBEDDING and for QUERIES only. The original text is always kept for display and
for the LLM context, so customers never see altered fee wording.
"""
from __future__ import annotations

import re
import unicodedata

_TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_INVISIBLE = re.compile(r"[​-‏‪-‮⁦-⁩﻿]")
_ALEF = re.compile(r"[آأإٱ]")          # آ أ إ ٱ -> ا
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_PUNCT = str.maketrans({"،": ",", "؛": ";", "؟": "?", "٫": ".", "٬": ","})
_WS = re.compile(r"\s+")

_AR_LETTER = re.compile(r"[؀-ۿ]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")


def normalize_text(text: str) -> str:
    """Light, retrieval-friendly normalisation. Keeps ة and the word order untouched."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = _INVISIBLE.sub("", t)
    t = _TASHKEEL.sub("", t)
    t = t.replace("ـ", "")                  # tatweel
    t = _ALEF.sub("ا", t)
    t = t.replace("ى", "ي")            # ى -> ي
    t = t.translate(_DIGITS).translate(_PUNCT)   # Arabic-Indic digits -> ASCII
    return _WS.sub(" ", t).strip()


def detect_language(text: str) -> str:
    """'ar' if Arabic letters dominate, else 'en'. Only a fallback when the JSON has no language."""
    ar = len(_AR_LETTER.findall(text or ""))
    la = len(_LATIN_LETTER.findall(text or ""))
    return "ar" if ar >= la and ar > 0 else "en"
