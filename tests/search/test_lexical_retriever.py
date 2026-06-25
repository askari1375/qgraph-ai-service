from typing import Any

import pytest

from src.search.contracts import ContentType, QueryContext, SearchFilters
from src.search.indexing.documents import DOCUMENT_SCHEMA_VERSION
from src.search.indexing.mapping import ANALYSIS_PROFILE_VERSION
from src.search.indexing.normalization import NORMALIZATION_PROFILE_VERSION
from src.search.opensearch_client import OpenSearchError
from src.search.retrievers.lexical_opensearch import (
    LexicalRetriever,
    aggregate_surah_distribution,
    build_search_body,
    build_surah_distribution_body,
)


def _compatible_profile(**overrides: Any) -> dict[str, Any]:
    profile = {
        "document_schema_version": DOCUMENT_SCHEMA_VERSION,
        "normalization_profile_version": NORMALIZATION_PROFILE_VERSION,
        "analysis_profile_version": ANALYSIS_PROFILE_VERSION,
    }
    profile.update(overrides)
    return profile


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: Any = None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = ""

    def json(self) -> Any:
        return self._payload


class _FakeAdapter:
    def __init__(self, *, profile: dict[str, Any], search_payload: dict[str, Any]):
        self._profile = profile
        self._search_payload = search_payload
        self.search_bodies: list[dict[str, Any]] = []

    def get(self, path: str) -> _FakeResponse:
        return _FakeResponse(
            200,
            {
                "qgraph-ayah-lexical-20260625-001": {
                    "mappings": {"_meta": {"qgraph_index_profile": self._profile}}
                }
            },
        )

    def post(self, path: str, *, json_payload=None, content=None, headers=None) -> _FakeResponse:
        self.search_bodies.append(json_payload)
        return _FakeResponse(200, self._search_payload)

    def put(self, path: str, *, json_payload) -> _FakeResponse:  # pragma: no cover - unused
        return _FakeResponse(200, {})

    def delete(self, path: str) -> _FakeResponse:  # pragma: no cover - unused
        return _FakeResponse(200, {})


def _query_context(**overrides: Any) -> QueryContext:
    values: dict[str, Any] = {"raw_query": "الرحمن", "top_k": 5}
    values.update(overrides)
    return QueryContext(**values)


def test_build_search_body_searches_all_language_fields():
    body = build_search_body(_query_context())
    primary = body["query"]["bool"]["should"][0]["multi_match"]["fields"]
    assert primary == ["content_ar^3", "content_fa^2", "content_en^2", "content_general"]
    # A lower-boost recall clause over the stemmed/exact sub-fields.
    recall = body["query"]["bool"]["should"][1]["multi_match"]
    assert recall["boost"] < 1.0
    assert "content_ar.stemmed" in recall["fields"]


def test_build_search_body_collapse_is_caller_controlled():
    assert build_search_body(_query_context(collapse=True))["collapse"] == {
        "field": "canonical_content_id"
    }
    assert "collapse" not in build_search_body(_query_context(collapse=False))


def test_build_search_body_applies_filters_and_highlight_tags():
    context = _query_context(
        filters=SearchFilters.from_request_filters({"content_types": ["surah_name"]})
    )
    body = build_search_body(context)
    assert {"terms": {"metadata.content_type": ["surah_name"]}} in body["query"]["bool"]["filter"]
    assert body["highlight"]["pre_tags"] == ["<mark>"]


def test_retrieve_maps_hits_to_candidates():
    adapter = _FakeAdapter(
        profile=_compatible_profile(),
        search_payload={
            "hits": {
                "hits": [
                    {
                        "_id": "ayah:1:1:ar",
                        "_score": 9.0,
                        "_source": {
                            "id": "ayah:1:1:ar",
                            "canonical_content_id": "ayah:1:1",
                            "content_ar": "بسم الله الرحمن الرحيم",
                            "metadata": {
                                "content_type": "quran_ayah",
                                "surah_number": 1,
                                "ayah_number": 1,
                                "language_code": "ar",
                            },
                        },
                        "highlight": {"content_ar": ["بسم الله <mark>الرحمن</mark> الرحيم"]},
                    }
                ]
            }
        },
    )
    candidates = LexicalRetriever(adapter, "qgraph-ayah-lexical-active").retrieve(_query_context())

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.rank == 1
    assert candidate.score == 9.0
    assert candidate.document_id == "ayah:1:1:ar"
    assert candidate.canonical_content_id == "ayah:1:1"
    assert candidate.content_type == ContentType.QURAN_AYAH
    assert candidate.retriever == "opensearch_lexical"
    assert candidate.text == "بسم الله الرحمن الرحيم"
    assert "<mark>" in candidate.highlighted_text
    assert candidate.matched_fields == ["content_ar"]


def test_retrieve_refuses_an_incompatible_index():
    adapter = _FakeAdapter(
        profile=_compatible_profile(analysis_profile_version="something-old"),
        search_payload={"hits": {"hits": []}},
    )
    with pytest.raises(OpenSearchError) as exc_info:
        LexicalRetriever(adapter, "qgraph-ayah-lexical-active").retrieve(_query_context())
    assert exc_info.value.reason == "index_profile_mismatch"


def test_retrieve_rejects_unknown_content_type():
    adapter = _FakeAdapter(
        profile=_compatible_profile(),
        search_payload={
            "hits": {
                "hits": [
                    {
                        "_id": "x",
                        "_score": 1.0,
                        "_source": {
                            "id": "x",
                            "canonical_content_id": "x",
                            "metadata": {"content_type": "bogus"},
                        },
                    }
                ]
            }
        },
    )
    with pytest.raises(OpenSearchError) as exc_info:
        LexicalRetriever(adapter, "qgraph-ayah-lexical-active").retrieve(_query_context())
    assert exc_info.value.reason == "unexpected_content_type"


def test_build_surah_distribution_body_is_size_zero_terms_agg():
    context = _query_context(
        filters=SearchFilters.from_request_filters({"content_types": ["quran_ayah"]})
    )
    body = build_surah_distribution_body(context, size=15)
    assert body["size"] == 0
    assert body["aggs"]["surahs"]["terms"] == {"field": "metadata.surah_number", "size": 15}
    assert {"terms": {"metadata.content_type": ["quran_ayah"]}} in body["query"]["bool"]["filter"]


def test_aggregate_surah_distribution_parses_and_sorts_buckets():
    adapter = _FakeAdapter(
        profile=_compatible_profile(),
        search_payload={
            "aggregations": {
                "surahs": {
                    "buckets": [
                        {"key": 2, "doc_count": 17},
                        {"key": 1, "doc_count": 3},
                    ]
                }
            }
        },
    )
    distribution = aggregate_surah_distribution(
        adapter, "qgraph-ayah-lexical-active", _query_context()
    )
    assert distribution == [{"surah": 1, "value": 3}, {"surah": 2, "value": 17}]


def test_aggregate_surah_distribution_empty_when_no_aggregations():
    adapter = _FakeAdapter(profile=_compatible_profile(), search_payload={"hits": {"hits": []}})
    assert (
        aggregate_surah_distribution(adapter, "qgraph-ayah-lexical-active", _query_context()) == []
    )
