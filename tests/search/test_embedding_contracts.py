"""Embedding provider contract, profile immutability, and response validation."""

import math

import pytest
from pydantic import ValidationError

from src.search.embeddings.contracts import (
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingProviderProfile,
    validate_embedding_vectors,
)
from tests.support.embeddings import DeterministicEmbeddingProvider


def test_profile_forbids_extra_fields():
    with pytest.raises(ValidationError):
        EmbeddingProviderProfile(provider="p", model="m", dimensions=4, unexpected="x")


def test_profile_is_frozen():
    profile = EmbeddingProviderProfile(provider="p", model="m", dimensions=4)
    with pytest.raises(ValidationError):
        profile.dimensions = 8


def test_profile_dimensions_must_be_positive():
    with pytest.raises(ValidationError):
        EmbeddingProviderProfile(provider="p", model="m", dimensions=0)


def test_deterministic_provider_satisfies_protocol():
    assert isinstance(DeterministicEmbeddingProvider(), EmbeddingProvider)


def test_embed_documents_preserves_cardinality_and_shape():
    provider = DeterministicEmbeddingProvider(dimensions=6)
    vectors = provider.embed_documents(["alpha", "beta", "gamma"])
    assert len(vectors) == 3
    assert all(len(vector) == 6 for vector in vectors)
    assert all(math.isfinite(value) for vector in vectors for value in vector)


def test_embeddings_are_deterministic_and_mode_aware():
    provider = DeterministicEmbeddingProvider(dimensions=6)
    assert provider.embed_query("light") == provider.embed_query("light")
    # Document and query embeddings of the same text differ (asymmetric provider).
    assert provider.embed_documents(["light"])[0] != provider.embed_query("light")


def test_validate_embedding_vectors_accepts_valid_batch():
    validate_embedding_vectors([[0.1, 0.2], [0.3, 0.4]], expected_count=2, dimensions=2)


def test_validate_embedding_vectors_rejects_wrong_count():
    with pytest.raises(EmbeddingError) as excinfo:
        validate_embedding_vectors([[0.1, 0.2]], expected_count=2, dimensions=2)
    assert excinfo.value.reason == "embedding_response_invalid"


def test_validate_embedding_vectors_rejects_wrong_dimensions():
    with pytest.raises(EmbeddingError) as excinfo:
        validate_embedding_vectors([[0.1, 0.2, 0.3]], expected_count=1, dimensions=2)
    assert excinfo.value.reason == "embedding_response_invalid"


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_validate_embedding_vectors_rejects_non_finite(bad_value):
    with pytest.raises(EmbeddingError) as excinfo:
        validate_embedding_vectors([[0.1, bad_value]], expected_count=1, dimensions=2)
    assert excinfo.value.reason == "embedding_response_invalid"
