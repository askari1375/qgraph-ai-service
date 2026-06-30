"""Versioned embedding-input preparation: reuse, validation, no silent truncation."""

import pytest

from src.search.embeddings.contracts import EmbeddingError
from src.search.embeddings.input import (
    EMBEDDING_INPUT_PROFILE_ID,
    EMBEDDING_INPUT_PROFILE_VERSION,
    prepare_embedding_input,
)
from src.search.indexing.normalization import normalize_text


@pytest.mark.parametrize(
    ("text", "language_code"),
    [
        ("الرَّحْمَٰن", "ar"),
        ("خداوند بخشنده", "fa"),
        ("In the Name!", "en"),
    ],
)
def test_prepare_reuses_canonical_normalizer(text, language_code):
    assert prepare_embedding_input(text, language_code) == normalize_text(text, language_code)


@pytest.mark.parametrize("text", ["", "   ", "!!!"])
def test_empty_after_normalization_raises(text):
    with pytest.raises(EmbeddingError) as excinfo:
        prepare_embedding_input(text, "en")
    assert excinfo.value.reason == "embedding_input_empty"


def test_over_max_chars_raises_without_truncating():
    with pytest.raises(EmbeddingError) as excinfo:
        prepare_embedding_input("in the name of god", "en", max_chars=5)
    assert excinfo.value.reason == "embedding_input_too_long"
    # The error reports the real length — nothing was silently shortened to fit.
    assert excinfo.value.detail["length"] > 5


def test_within_max_chars_returns_normalized():
    assert prepare_embedding_input("Light", "en", max_chars=10) == "light"


def test_profile_identity_is_stable():
    assert EMBEDDING_INPUT_PROFILE_ID == "qgraph_embedding_input"
    assert EMBEDDING_INPUT_PROFILE_VERSION == "2026-06-30.v1"
