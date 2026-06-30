"""Guards that the test suite never inherits the developer's local .env.

Regression test for the hermeticity bug where a populated `.env` (live OpenSearch URL, `hybrid_v1`, a
real API key) leaked into supposedly isolated unit tests and made them hit localhost. conftest disables
`.env` loading at collection time; this asserts the result.
"""

from src.config import Settings


def test_dev_env_is_not_loaded_in_tests():
    settings = Settings()
    assert settings.opensearch_url == ""
    assert settings.qdrant_url == ""
    assert settings.search_retrieval_policy == "lexical_v1"
    assert settings.embedding_provider == ""
    assert settings.embedding_api_key == ""
    assert settings.django_internal_base_url == ""


def test_invalid_retrieval_policy_is_rejected():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(search_retrieval_policy="hybird_v1")  # typo must fail, not silently run lexical
