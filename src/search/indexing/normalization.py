"""Canonical Python text normalizer (cross-backend reuse only).

This module owns the canonical normalizer. OpenSearch's custom analyzers own search-time matching
(index and query side), so there is no indexed normalized field. The Python normalizer is kept *only*
for its cross-backend job — producing a single, versioned normalization that future embedding/Qdrant
and Neo4j paths can reuse, so the lexical and semantic halves stay consistent in spirit. It therefore
stays versioned even though OpenSearch no longer depends on it.
"""

from __future__ import annotations

import re
import string
import unicodedata

NORMALIZATION_PROFILE_ID = "qgraph_search_normalization"
NORMALIZATION_PROFILE_VERSION = "2026-06-22.v1"

_ARABIC_DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_ZERO_WIDTH_JOINER = "‌"
_PUNCTUATION_TABLE = str.maketrans(
    {character: " " for character in f"{string.punctuation}،؛؟«»“”‘’…"}
)
_WHITESPACE_RE = re.compile(r"\s+")

_ARABIC_TRANSLATION_TABLE = str.maketrans(
    {
        "آ": "ا",
        "أ": "ا",
        "إ": "ا",
        "ٱ": "ا",
        "ؤ": "و",
        "ئ": "ي",
        "ى": "ي",
        "ة": "ه",
        _TATWEEL: "",
    }
)

_PERSIAN_TRANSLATION_TABLE = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ۀ": "ه",
        "ة": "ه",
        _TATWEEL: "",
        _ZERO_WIDTH_JOINER: " ",
    }
)


def normalize_text(text: str, language_code: str) -> str:
    language = language_code.casefold()
    if language == "ar":
        return normalize_arabic(text)
    if language == "fa":
        return normalize_persian(text)
    if language == "en":
        return normalize_english(text)
    return normalize_generic(text)


def normalize_arabic(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _ARABIC_DIACRITICS_RE.sub("", normalized)
    normalized = normalized.translate(_ARABIC_TRANSLATION_TABLE)
    normalized = normalized.translate(_PUNCTUATION_TABLE)
    return _collapse_whitespace(normalized)


def normalize_persian(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _ARABIC_DIACRITICS_RE.sub("", normalized)
    normalized = normalized.translate(_PERSIAN_TRANSLATION_TABLE)
    normalized = normalized.translate(_PUNCTUATION_TABLE)
    return _collapse_whitespace(normalized)


def normalize_english(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.translate(_PUNCTUATION_TABLE)
    return _collapse_whitespace(normalized)


def normalize_generic(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.translate(_PUNCTUATION_TABLE)
    return _collapse_whitespace(normalized)


def _collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()
