"""Deterministic embedding provider for tests and developer experiments.

Implements :class:`~src.search.embeddings.contracts.EmbeddingProvider` with stable, finite, pure-stdlib
vectors (no network, no numpy). The same text always yields the same vector, and documents vs queries
yield different vectors (so the provider behaves like an asymmetric model). It lives under ``tests/`` on
purpose: it must never be selectable as a production runtime provider.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from src.search.embeddings.contracts import EmbeddingInputMode, EmbeddingProviderProfile

_DEFAULT_DIMENSIONS = 8


class DeterministicEmbeddingProvider:
    """A reproducible, dependency-free embedding provider for tests."""

    def __init__(
        self,
        *,
        dimensions: int = _DEFAULT_DIMENSIONS,
        provider: str = "deterministic-test",
        model: str = "deterministic-test",
    ):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions
        self._profile = EmbeddingProviderProfile(
            provider=provider,
            model=model,
            dimensions=dimensions,
            distinguishes_input_modes=True,
        )

    @property
    def profile(self) -> EmbeddingProviderProfile:
        return self._profile

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text, EmbeddingInputMode.DOCUMENT) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text, EmbeddingInputMode.QUERY)

    def _vector(self, text: str, mode: EmbeddingInputMode) -> list[float]:
        seed = f"{mode.value}:{text}"
        values: list[float] = []
        counter = 0
        while len(values) < self._dimensions:
            digest = hashlib.sha256(f"{seed}:{counter}".encode()).digest()
            for offset in range(0, len(digest), 4):
                if len(values) >= self._dimensions:
                    break
                integer = int.from_bytes(digest[offset : offset + 4], "big")
                values.append(integer / 0xFFFFFFFF * 2.0 - 1.0)
            counter += 1
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            return values
        return [value / norm for value in values]
