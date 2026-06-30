"""Resolve the production embedding provider from settings.

The single provider-construction seam, named like ``build_qdrant_store`` / ``build_opensearch_adapter``
— one production resolution point, not a provider registry. No real provider is wired yet, so this
raises ``embedding_provider_not_configured``; the concrete provider adapter fills in the branch here.
The deterministic test provider is never reachable through this path: test fakes arrive only by direct
dependency injection, never through runtime configuration.
"""

from __future__ import annotations

from src.config import Settings
from src.search.embeddings.contracts import EmbeddingError, EmbeddingProvider


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Return the configured production embedding provider, or raise if none is configured."""
    raise EmbeddingError(
        "no production embedding provider is configured",
        reason="embedding_provider_not_configured",
    )
