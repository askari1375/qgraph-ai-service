"""Backwards-compatible re-export of the canonical normalizer.

The implementation moved to :mod:`src.search.indexing.normalization`, which is now its home (kept for
cross-backend reuse). This module re-exports the public names so the existing serving path keeps
importing from here unchanged; it is removed once that path is retired.
"""

from src.search.indexing.normalization import (
    NORMALIZATION_PROFILE_ID,
    NORMALIZATION_PROFILE_VERSION,
    normalize_arabic,
    normalize_english,
    normalize_generic,
    normalize_persian,
    normalize_text,
)

__all__ = [
    "NORMALIZATION_PROFILE_ID",
    "NORMALIZATION_PROFILE_VERSION",
    "normalize_arabic",
    "normalize_english",
    "normalize_generic",
    "normalize_persian",
    "normalize_text",
]
