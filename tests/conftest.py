from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.config import Settings, get_settings
from src.main import create_app

# Hermetic tests: never read the developer's local `.env`. It may point at live backends, select
# `hybrid_v1`, or carry a real API key — none of which a unit test should pick up. After this, Settings
# come only from explicit kwargs + code defaults, so "normal tests need neither live Qdrant nor paid
# calls" is enforced by isolation rather than assumed. Runs at collection time, before any fixture.
Settings.model_config["env_file"] = None
get_settings.cache_clear()


@pytest.fixture(scope="session")
def app():
    return create_app()


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def search_plan_payload():
    return {
        "query": "verses about patience",
        "filters": {"surah_ids": [2]},
        "output_preferences": {"include_summary": True, "include_statistics": True},
    }


@pytest.fixture
def search_execute_payload():
    return {
        "query": "verses about patience",
        "filters": {"surah_ids": [2]},
        "output_preferences": {
            "include_summary": True,
            "include_statistics": True,
            "include_explanation": False,
        },
        "context": {"query_id": 123, "execution_id": 456},
    }


@pytest.fixture
def search_job_create_payload():
    idempotency_key = f"search-exec-{uuid4().hex[:12]}"
    return {
        "query": "verses about patience",
        "filters": {"surah_ids": [2]},
        "output_preferences": {
            "include_summary": True,
            "include_statistics": True,
            "include_explanation": False,
        },
        "context": {"query_id": 123, "execution_id": 456},
        "idempotency_key": idempotency_key,
        "client_ref": {"query_id": 123, "execution_id": 456},
    }


@pytest.fixture
def segmentation_generate_payload():
    return {
        "surah_id": 2,
        "ayahs": [
            {
                "id": 8,
                "number_in_surah": 1,
                "text_ar": "placeholder",
                "translations": [{"lang": "en", "text": "placeholder"}],
            },
            {
                "id": 9,
                "number_in_surah": 2,
                "text_ar": "placeholder",
                "translations": [{"lang": "en", "text": "placeholder"}],
            },
        ],
        "options": {
            "granularity": "medium",
            "max_segments": 20,
            "include_tags": True,
            "include_summaries": True,
        },
        "context": {
            "workspace_slug": "my-workspace",
            "requested_by_user_id": 42,
        },
    }
