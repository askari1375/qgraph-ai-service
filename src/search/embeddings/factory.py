"""Resolve the production embedding provider from settings.

The single provider-construction seam, named like ``build_qdrant_store`` / ``build_opensearch_adapter``
— one production resolution point, not a provider registry. Branch on the configured provider name and
construct its adapter; an unset or unknown provider raises ``embedding_provider_not_configured``. The
deterministic test provider is never reachable through this path: test fakes arrive only by direct
dependency injection, never through runtime configuration.
"""

from __future__ import annotations

from src.config import Settings
from src.search.embeddings.contracts import EmbeddingError, EmbeddingProvider
from src.search.embeddings.openai_provider import OpenAIEmbeddingProvider


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Return the configured production embedding provider, or raise if none is configured."""
    if settings.embedding_provider == "openai":
        has_config = (
            settings.embedding_model
            and settings.embedding_dimensions > 0
            and settings.embedding_api_key
        )
        if not has_config:
            raise EmbeddingError(
                "openai embedding provider is selected but model, dimensions, or api key is missing",
                reason="embedding_provider_not_configured",
                detail={
                    "has_model": bool(settings.embedding_model),
                    "has_dimensions": settings.embedding_dimensions > 0,
                    "has_api_key": bool(settings.embedding_api_key),
                },
            )
        return OpenAIEmbeddingProvider(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            api_key=settings.embedding_api_key,
            timeout_seconds=settings.embedding_timeout_seconds,
            max_retries=settings.embedding_max_retries,
        )
    raise EmbeddingError(
        "no production embedding provider is configured",
        reason="embedding_provider_not_configured",
        detail={"embedding_provider": settings.embedding_provider},
    )
