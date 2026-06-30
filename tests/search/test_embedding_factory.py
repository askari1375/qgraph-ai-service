"""build_embedding_provider resolves a configured provider, or fails closed."""

import pytest

from src.config import Settings
from src.search.embeddings.contracts import EmbeddingError
from src.search.embeddings.factory import build_embedding_provider
from src.search.embeddings.openai_provider import OpenAIEmbeddingProvider


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-large",
        "embedding_dimensions": 3072,
        "embedding_api_key": "sk-test",
    }
    base.update(overrides)
    return Settings(**base)


def test_resolves_openai_provider():
    provider = build_embedding_provider(_settings())
    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider.profile.provider == "openai"
    assert provider.profile.dimensions == 3072


@pytest.mark.parametrize(
    "overrides",
    [
        {"embedding_api_key": ""},
        {"embedding_model": ""},
        {"embedding_dimensions": 0},
    ],
)
def test_openai_with_incomplete_config_is_not_configured(overrides: dict[str, object]):
    with pytest.raises(EmbeddingError) as excinfo:
        build_embedding_provider(_settings(**overrides))
    assert excinfo.value.reason == "embedding_provider_not_configured"


@pytest.mark.parametrize("provider_name", ["", "cohere"])
def test_unset_or_unknown_provider_is_not_configured(provider_name: str):
    with pytest.raises(EmbeddingError) as excinfo:
        build_embedding_provider(_settings(embedding_provider=provider_name))
    assert excinfo.value.reason == "embedding_provider_not_configured"
