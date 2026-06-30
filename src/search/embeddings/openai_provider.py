"""OpenAI embedding provider — the first production ``EmbeddingProvider`` adapter.

Wraps the official ``openai`` SDK behind the project's narrow embedding seam. The SDK owns transport,
timeout, and retry/backoff; this adapter owns the *project* contract: it exposes the immutable
compatibility profile, reassembles vectors in input order, validates the response, and maps SDK
failures to the typed ``EmbeddingError`` reasons the build/retrieval paths already understand. Tests
inject a fake through ``client``; no network call happens unless a real client is built.

OpenAI ``text-embedding-3-large`` is symmetric (no query/document ``input_type``) and L2-normalized, so
``distinguishes_input_modes=False`` and cosine distance ranks identically. V1 uses the model's native
dimension with no reduction: the API ``dimensions`` parameter is intentionally not sent, so a profile
that claims a different dimension fails ``validate_embedding_vectors`` loudly instead of silently
reducing.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import openai

from src.search.embeddings.contracts import (
    EmbeddingError,
    EmbeddingProviderProfile,
    validate_embedding_vectors,
)

_PROVIDER_NAME = "openai"


class OpenAIEmbeddingProvider:
    """Production ``EmbeddingProvider`` backed by the OpenAI embeddings API."""

    def __init__(
        self,
        *,
        model: str,
        dimensions: int,
        api_key: str = "",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        client: openai.OpenAI | None = None,
    ):
        if not model:
            raise EmbeddingError(
                "embedding model is not configured",
                reason="embedding_provider_not_configured",
            )
        if dimensions <= 0:
            raise EmbeddingError(
                "embedding dimensions must be positive",
                reason="embedding_provider_not_configured",
                detail={"dimensions": dimensions},
            )
        self._dimensions = dimensions
        self._profile = EmbeddingProviderProfile(
            provider=_PROVIDER_NAME,
            model=model,
            dimensions=dimensions,
            distinguishes_input_modes=False,
            vectors_normalized=True,
        )
        # Injected client (tests) wins; otherwise the SDK owns timeout/retry. The factory has already
        # checked the api key is present, so constructing a real client here will not fail on auth.
        self._client = client or openai.OpenAI(
            api_key=api_key or None,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    @property
    def profile(self) -> EmbeddingProviderProfile:
        return self._profile

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents; one vector per input, in input order."""
        return self._embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query. OpenAI is symmetric, so this is the document path with one input."""
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with _translate_errors():
            response = self._client.embeddings.create(model=self._profile.model, input=texts)
        vectors = _ordered_vectors(response, expected_count=len(texts))
        validate_embedding_vectors(vectors, expected_count=len(texts), dimensions=self._dimensions)
        return vectors


def _ordered_vectors(response: Any, *, expected_count: int) -> list[list[float]]:
    """Extract embeddings sorted by their response ``index`` so order matches the input batch."""
    data = getattr(response, "data", None)
    if data is None or len(data) != expected_count:
        raise EmbeddingError(
            f"expected {expected_count} embeddings, got {0 if data is None else len(data)}",
            reason="embedding_response_invalid",
            detail={"expected_count": expected_count},
        )
    try:
        ordered = sorted(data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]
    except (AttributeError, TypeError) as exc:
        raise EmbeddingError(
            "embedding response is malformed",
            reason="embedding_response_invalid",
        ) from exc


@contextmanager
def _translate_errors() -> Iterator[None]:
    """Map any OpenAI SDK failure (timeout, connection, rate limit, API status) to ``EmbeddingError``.

    ``openai.APIError`` is the base of every request failure the SDK raises, including the
    timeout/connection/rate-limit subclasses, so one catch covers them all without a silent fallback.
    """
    try:
        yield
    except openai.APIError as exc:
        detail: dict[str, Any] = {"error_type": type(exc).__name__}
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            detail["status_code"] = status_code
        raise EmbeddingError(
            f"OpenAI embeddings request failed: {exc}",
            reason="embedding_provider_unavailable",
            detail=detail,
        ) from exc
