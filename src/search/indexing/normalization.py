"""Canonical Python text normalizer (cross-backend reuse only).

This becomes the home of the normalizer currently in ``src/services/search_normalization.py``.

Important scope note: OpenSearch's custom analyzers own search-time matching, so there is no indexed
``normalized_text`` field. The Python normalizer is kept *only* for its cross-backend job — producing
a single, versioned normalization that future embedding/Qdrant and Neo4j paths can reuse, so the
lexical and semantic halves stay consistent in spirit. It therefore stays versioned
(``NORMALIZATION_PROFILE_VERSION``) even though OpenSearch no longer depends on it.
"""

from __future__ import annotations


def normalize_text(text: str, language_code: str) -> str:
    """Normalize ``text`` for the given language for cross-backend reuse.

    Not implemented yet: moved from ``services/search_normalization.py`` (Arabic/Persian/English
    normalization), carrying its profile id/version with it.
    """
    raise NotImplementedError("indexing.normalization.normalize_text is not implemented yet")
