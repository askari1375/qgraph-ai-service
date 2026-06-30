"""Embedding provider contract and the immutable provider profile.

This is the project-owned seam for turning text into vectors — a small typed contract, deliberately
not a third-party embedding framework. The compatibility facts that protect the vector store from
silent model/dimension drift (provider, model, dimensions) live in *our* types so they can be stamped
onto a semantic index and checked before retrieval. The real provider adapter (Cohere) implements this
protocol later; tests and developer experiments inject a deterministic provider through the same
protocol. There is no runtime fake fallback.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingInputMode(str, Enum):
    """Whether text is prepared/embedded as an indexed document or as a query.

    Asymmetric providers embed the two differently. For Cohere this maps to the API ``input_type``
    (``search_document`` / ``search_query``); the project keeps neutral names so the contract does not
    bake in one provider's vocabulary.
    """

    DOCUMENT = "document"
    QUERY = "query"


class EmbeddingProviderProfile(BaseModel):
    """Immutable compatibility facts about one embedding provider/model.

    These are the fields that, if they silently changed, would corrupt a vector collection: a new
    model or dimension count makes existing vectors meaningless. They get stamped onto the semantic
    index profile later and checked before retrieval, so the profile is ``frozen`` — a built profile is
    a fact, not a mutable config object.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    dimensions: int = Field(gt=0)
    #: True if the provider embeds documents and queries differently (e.g. Cohere ``input_type``).
    distinguishes_input_modes: bool = False


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into dense vectors. Mirrors the ``Retriever`` protocol style.

    The real Cohere adapter and the deterministic test provider both implement this exact shape, so
    nothing downstream branches on which provider produced a vector.
    """

    @property
    def profile(self) -> EmbeddingProviderProfile: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed indexed documents (``EmbeddingInputMode.DOCUMENT``); one vector per input, in order."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query (``EmbeddingInputMode.QUERY``)."""
        ...


class EmbeddingError(Exception):
    """Typed embedding-domain failure, mirroring ``OpenSearchError``'s reason/detail shape."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        detail: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.reason = reason
        self.detail = detail or {}


def validate_embedding_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    expected_count: int,
    dimensions: int,
) -> None:
    """Validate a provider response batch; raise ``EmbeddingError`` on any inconsistency.

    Guards the invariants the real adapter must enforce (and the build relies on): one vector per
    input, every vector of the configured dimension, every value finite. No silent coercion.
    """
    if len(vectors) != expected_count:
        raise EmbeddingError(
            f"expected {expected_count} vectors, got {len(vectors)}",
            reason="embedding_response_invalid",
            detail={"expected_count": expected_count, "actual_count": len(vectors)},
        )
    for index, vector in enumerate(vectors):
        if len(vector) != dimensions:
            raise EmbeddingError(
                f"vector {index} has {len(vector)} dims, expected {dimensions}",
                reason="embedding_response_invalid",
                detail={
                    "index": index,
                    "expected_dimensions": dimensions,
                    "actual_dimensions": len(vector),
                },
            )
        if any(not math.isfinite(value) for value in vector):
            raise EmbeddingError(
                f"vector {index} contains a non-finite value",
                reason="embedding_response_invalid",
                detail={"index": index},
            )
