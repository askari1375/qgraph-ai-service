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
    values: dict[str, Any] = {"opensearch_url": "http://opensearch:9200"}
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


def _body_content_types(body: dict[str, Any]) -> set[str]:
    clauses = body.get("query", {}).get("bool", {}).get("filter", [])
    found: set[str] = set()
    for clause in clauses:
        terms = clause.get("terms", {}) if isinstance(clause, dict) else {}
        if "metadata.content_type" in terms:
            found.update(terms["metadata.content_type"])
    return found


class _FakeAdapter:
    """Routes each scoped query by its content_type filter (Arabic vs translation vs aggregation)."""

    def __init__(
        self,
        *,
        profile: dict[str, Any] | None = None,
        ayah_hits=None,
        translation_hits=None,
        aggregations=None,
        get_status: int = 200,
    ):
        self._profile = profile if profile is not None else _profile()
        self._ayah_hits = ayah_hits or []
        self._translation_hits = translation_hits or []
        self._aggregations = aggregations or {}
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
        body = json_payload or {}
        if "aggs" in body:
            return _Resp(200, {"hits": {"hits": []}, "aggregations": self._aggregations})
        content_types = _body_content_types(body)
        if "translation" in content_types and "quran_ayah" not in content_types:
            return _Resp(200, {"hits": {"hits": self._translation_hits}})
        return _Resp(200, {"hits": {"hits": self._ayah_hits}})

    def put(self, path: str, *, json_payload) -> _Resp:  # pragma: no cover
        return _Resp(200, {})

    def delete(self, path: str) -> _Resp:  # pragma: no cover
        return _Resp(200, {})


def _arabic_hit(score: float) -> dict[str, Any]:
    return {
        "_id": "ayah:1:1:ar",
        "_score": score,
        "_source": {
            "id": "ayah:1:1:ar",
            "canonical_content_id": "ayah:1:1",
            "content_ar": "بسم الله الرحمن الرحيم",
            "metadata": {
                "content_type": "quran_ayah",
                "surah_number": 1,
                "ayah_number": 1,
                "ayah_global_number": 1,
                "language_code": "ar",
                "source_name": "Quran Arabic",
            },
        },
        "highlight": {"content_ar": ["بسم الله <mark>الرحمن</mark> الرحيم"]},
    }


def _translation_hit(score: float, language_code: str = "en") -> dict[str, Any]:
    return {
        "_id": f"ayah:1:1:translation:{language_code}-sahih",
        "_score": score,
        "_source": {
            "id": f"ayah:1:1:translation:{language_code}-sahih",
            "canonical_content_id": "ayah:1:1",
            "content_en": "In the name of Allah, the Entirely Merciful",
            "metadata": {
                "content_type": "translation",
                "surah_number": 1,
                "ayah_number": 1,
                "ayah_global_number": 1,
                "language_code": language_code,
                "source_id": f"{language_code}-sahih",
                "source_name": "Sahih International",
            },
        },
        "highlight": {"content_en": ["In the name of <mark>Allah</mark>"]},
    }


def test_retrieval_groups_chart_arabic_and_translation_blocks():
    response = build_search_execute_response(
        SearchExecuteRequest(
            query="mercy", filters={"surahs": [1]}, output_preferences={"top_k": 5}
        ),
        settings=_settings(),
        adapter=_FakeAdapter(
            ayah_hits=[_arabic_hit(9.0)],
            translation_hits=[_translation_hit(7.5)],
            aggregations={"surahs": {"buckets": [{"key": 1, "doc_count": 3}]}},
        ),
    )

    assert response.metadata["backend"] == "open_search"
    assert response.metadata["analysis_profile_version"] == ANALYSIS_PROFILE_VERSION

    assert [b.block_type for b in response.blocks] == [
        "surah_distribution",
        "ayah_results",
        "ayah_results",
    ]
    assert [b.title for b in response.blocks] == [
        "Where this appears",
        "Quran",
        "English translations",
    ]
    assert response.blocks[0].payload["values"] == [{"surah": 1, "value": 3}]

    arabic_item = response.blocks[1].items[0]
    assert arabic_item.match_metadata["content_type"] == "quran_ayah"
    assert "<mark>" in arabic_item.highlighted_text

    english_item = response.blocks[2].items[0]
    assert english_item.match_metadata["content_type"] == "translation"
    assert english_item.match_metadata["document_id"] == "ayah:1:1:translation:en-sahih"


def test_include_translations_false_omits_translation_block():
    response = build_search_execute_response(
        SearchExecuteRequest(query="mercy", filters={"include_translations": False}),
        settings=_settings(),
        adapter=_FakeAdapter(
            ayah_hits=[_arabic_hit(9.0)], translation_hits=[_translation_hit(7.5)]
        ),
    )
    assert [b.block_type for b in response.blocks] == ["ayah_results"]
    assert response.blocks[0].title == "Quran"


def test_retrieval_confidence_reflects_absolute_score():
    request = SearchExecuteRequest(query="mercy")
    weak = build_search_execute_response(
        request, settings=_settings(), adapter=_FakeAdapter(ayah_hits=[_arabic_hit(0.5)])
    )
    strong = build_search_execute_response(
        request, settings=_settings(), adapter=_FakeAdapter(ayah_hits=[_arabic_hit(40.0)])
    )
    assert 0.0 < weak.overall_confidence < 0.1
    assert weak.overall_confidence < strong.overall_confidence < 1.0


def test_retrieval_empty_results_warns():
    response = build_search_execute_response(
        SearchExecuteRequest(query="zzz"), settings=_settings(), adapter=_FakeAdapter()
    )
    block = response.blocks[0]
    assert block.block_type == "ayah_results"
    assert block.items == []
    assert block.warning_text == "No lexical matches were returned."


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
