"""Embedding contracts and input preparation for semantic retrieval.

The project-owned seam for turning text into vectors. This package defines *contracts* and *input
preparation* only — no provider talks to a network here. The real Cohere adapter and a deterministic
test provider both implement :class:`EmbeddingProvider`; the input rules are versioned so an embedded
vector is reproducible.
"""

from __future__ import annotations

from src.search.embeddings.contracts import (
    EmbeddingError,
    EmbeddingInputMode,
    EmbeddingProvider,
    EmbeddingProviderProfile,
    validate_embedding_vectors,
)
from src.search.embeddings.input import (
    EMBEDDING_INPUT_PROFILE_ID,
    EMBEDDING_INPUT_PROFILE_VERSION,
    prepare_embedding_input,
)

__all__ = [
    "EMBEDDING_INPUT_PROFILE_ID",
    "EMBEDDING_INPUT_PROFILE_VERSION",
    "EmbeddingError",
    "EmbeddingInputMode",
    "EmbeddingProvider",
    "EmbeddingProviderProfile",
    "prepare_embedding_input",
    "validate_embedding_vectors",
]
