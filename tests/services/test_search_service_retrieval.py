from typing import Any

import pytest

from src.api.schemas.search import SearchExecuteRequest
from src.config import Settings
from src.search.indexing.documents import DOCUMENT_SCHEMA_VERSION
from src.search.indexing.mapping import ANALYSIS_PROFILE_VERSION
from src.search.indexing.normalization import (
    NORMALIZATION_PROFILE_ID,
    NORMALIZATION_PROFILE_VERSION,
)
from src.services.search_service import SearchRetrievalError, build_search_execute_response


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "search_lexical_backend_mode": "opensearch",
        "opensearch_url": "http://opensearch:9200",
    }
    values.update(overrides)
    return Settings(**values)


def _profile(**overrides: Any) -> dict[str, Any]:
    profile = {
        "index_id": "qgraph-ayah-lexical-20260625-001",
        "corpus_snapshot_id": "snapshot-001",
        "corpus_snapshot_hash": "sha256:abc123",
        "document_schema_version": DOCUMENT_SCHEMA_VERSION,
        "normalization_profile_id": NORMALIZATION_PROFILE_ID,
        "normalization_profile_version": NORMALIZATION_PROFILE_VERSION,
        "analysis_profile_version": ANALYSIS_PROFILE_VERSION,
    }
    profile.update(overrides)
    return profile


class _Resp:
    def __init__(self, status_code: int = 200, payload: Any = None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = ""

    def json(self) -> Any:
        return self._payload


class _FakeAdapter:
    def __init__(self, *, profile: dict[str, Any] | None = None, hits=None, get_status: int = 200):
        self._profile = profile if profile is not None else _profile()
        self._hits = hits if hits is not None else []
        self._get_status = get_status

    def get(self, path: str) -> _Resp:
        if self._get_status != 200:
            return _Resp(self._get_status)
        return _Resp(
            200,
            {
                "qgraph-ayah-lexical-20260625-001": {
                    "mappings": {"_meta": {"qgraph_index_profile": self._profile}}
                }
            },
        )

    def post(self, path: str, *, json_payload=None, content=None, headers=None) -> _Resp:
        return _Resp(200, {"hits": {"hits": self._hits}})

    def put(self, path: str, *, json_payload) -> _Resp:  # pragma: no cover
        return _Resp(200, {})

    def delete(self, path: str) -> _Resp:  # pragma: no cover
        return _Resp(200, {})


def _hit(score: float) -> dict[str, Any]:
    return {
        "_id": "ayah:1:1:translation:en-sahih",
        "_score": score,
        "_source": {
            "id": "ayah:1:1:translation:en-sahih",
            "canonical_content_id": "ayah:1:1",
            "content_en": "In the name of Allah, the Entirely Merciful",
            "metadata": {
                "content_type": "translation",
                "surah_number": 1,
                "ayah_number": 1,
                "ayah_global_number": 1,
                "language_code": "en",
                "source_id": "en-sahih",
                "source_name": "Sahih International",
            },
        },
        "highlight": {"content_en": ["In the name of <mark>Allah</mark>"]},
    }


def test_retrieval_mode_returns_django_compatible_blocks_and_items():
    response = build_search_execute_response(
        SearchExecuteRequest(
            query="mercy", filters={"surahs": [1]}, output_preferences={"top_k": 5}
        ),
        settings=_settings(),
        adapter=_FakeAdapter(hits=[_hit(7.5)]),
    )

    assert response.render_schema_version == "v1"
    assert response.metadata["mock"] is False
    assert response.metadata["backend"] == "open_search"
    assert response.metadata["corpus_snapshot_id"] == "snapshot-001"
    assert response.metadata["analysis_profile_version"] == ANALYSIS_PROFILE_VERSION

    block = response.blocks[0]
    assert block.block_type == "results"
    assert block.payload == {"query": "mercy", "result_count": 1, "top_k": 5}
    item = block.items[0]
    assert item.rank == 1
    assert item.result_type == "ayah"
    assert item.score == 1.0
    assert item.title == "Surah 1, Ayah 1"
    assert item.provenance["backend"] == "open_search"
    assert item.provenance["lexical_score"] == 7.5
    assert item.match_metadata["document_id"] == "ayah:1:1:translation:en-sahih"
    assert item.match_metadata["canonical_content_id"] == "ayah:1:1"
    assert item.match_metadata["content_type"] == "translation"


def test_retrieval_confidence_reflects_absolute_score():
    request = SearchExecuteRequest(query="mercy")
    weak = build_search_execute_response(
        request, settings=_settings(), adapter=_FakeAdapter(hits=[_hit(0.5)])
    )
    strong = build_search_execute_response(
        request, settings=_settings(), adapter=_FakeAdapter(hits=[_hit(40.0)])
    )
    assert weak.blocks[0].items[0].score == 1.0
    assert 0.0 < weak.overall_confidence < 0.1
    assert weak.overall_confidence < strong.overall_confidence < 1.0


def test_retrieval_empty_results_warns():
    response = build_search_execute_response(
        SearchExecuteRequest(query="zzz"), settings=_settings(), adapter=_FakeAdapter(hits=[])
    )
    assert response.blocks[0].items == []
    assert response.blocks[0].warning_text == "No lexical matches were returned."


def test_retrieval_maps_missing_index_to_clear_error():
    with pytest.raises(SearchRetrievalError) as exc_info:
        build_search_execute_response(
            SearchExecuteRequest(query="mercy"),
            settings=_settings(),
            adapter=_FakeAdapter(get_status=404),
        )
    assert exc_info.value.reason == "index_not_found"
    assert exc_info.value.status_code == 404


def test_retrieval_maps_incompatible_index_to_clear_error():
    with pytest.raises(SearchRetrievalError) as exc_info:
        build_search_execute_response(
            SearchExecuteRequest(query="mercy"),
            settings=_settings(),
            adapter=_FakeAdapter(profile=_profile(analysis_profile_version="stale")),
        )
    assert exc_info.value.reason == "index_profile_mismatch"


def test_retrieval_requires_configured_opensearch_url():
    with pytest.raises(SearchRetrievalError) as exc_info:
        build_search_execute_response(
            SearchExecuteRequest(query="mercy"),
            settings=_settings(opensearch_url=""),
        )
    assert exc_info.value.reason == "opensearch_not_configured"
