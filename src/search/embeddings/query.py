"""Compute the one query embedding a hybrid search reuses across every retrieval scope.

The query is embedded **once** here and the vector is then shared (via ``QueryContext.query_embedding``)
across the ayah and translation scopes, so a search never pays for the same embedding twice. Input
preparation goes through the same versioned normalizer as documents (``prepare_embedding_input``); the
provider's symmetric/asymmetric query handling is its own concern (OpenAI is symmetric, so ``embed_query``
is just the document path with one input).
"""

from __future__ import annotations

from src.search.contracts import QueryContext
from src.search.embeddings.contracts import EmbeddingProvider, validate_embedding_vectors
from src.search.embeddings.input import prepare_embedding_input


def embed_query_for_search(provider: EmbeddingProvider, query_context: QueryContext) -> list[float]:
    """Normalize and embed the query once; validate the vector before it is reused across scopes.

    ``detected_language`` is a soft hint that picks the normalization dialect; an unknown/empty value
    falls back to the generic normalizer (``normalize_text``), so any-language queries still embed.
    """
    normalized = prepare_embedding_input(
        query_context.raw_query, query_context.detected_language or ""
    )
    vector = provider.embed_query(normalized)
    validate_embedding_vectors([vector], expected_count=1, dimensions=provider.profile.dimensions)
    return vector
