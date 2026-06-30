"""Versioned embedding-input preparation (document and query text).

OpenSearch analyzers own lexical normalization, but Qdrant cannot call them, so the semantic side
prepares its input through the canonical Python normalizer in
:mod:`src.search.indexing.normalization`. Doing that reuse here — under its own version — keeps the
embedded text reproducible and records the rule that produced a vector.

Document and query text are normalized identically today; the document/query distinction is carried by
:class:`~src.search.embeddings.contracts.EmbeddingInputMode` and applied by the provider (for Cohere,
the API ``input_type``), not by a text prefix. If a future provider needs asymmetric prefixes, add
them here and bump :data:`EMBEDDING_INPUT_PROFILE_VERSION`.
"""

from __future__ import annotations

from src.search.embeddings.contracts import EmbeddingError
from src.search.indexing.normalization import normalize_text

EMBEDDING_INPUT_PROFILE_ID = "qgraph_embedding_input"
EMBEDDING_INPUT_PROFILE_VERSION = "2026-06-30.v1"


def prepare_embedding_input(
    text: str,
    language_code: str,
    *,
    max_chars: int | None = None,
) -> str:
    """Normalize ``text`` for embedding; raise ``EmbeddingError`` on empty or over-long input.

    Never truncates: an over-limit document is a build error to surface (with its id, by the caller),
    not something to silently shorten, because hidden truncation makes a vector unreproducible.
    """
    normalized = normalize_text(text, language_code)
    if not normalized:
        raise EmbeddingError(
            "embedding input is empty after normalization",
            reason="embedding_input_empty",
            detail={"language_code": language_code},
        )
    if max_chars is not None and len(normalized) > max_chars:
        raise EmbeddingError(
            f"embedding input has {len(normalized)} chars, exceeds max {max_chars}",
            reason="embedding_input_too_long",
            detail={"length": len(normalized), "max_chars": max_chars},
        )
    return normalized
