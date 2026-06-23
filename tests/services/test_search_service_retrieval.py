from datetime import datetime, timezone
from typing import Any

import pytest

from src.api.schemas.search import SearchExecuteRequest
from src.config import Settings
from src.services.opensearch_lexical import (
    LEXICAL_INDEX_PROFILE_SCHEMA_VERSION,
    OPEN_SEARCH_BACKEND_NAME,
    LexicalIndexProfile,
    LexicalSearchBackendError,
    LexicalSearchHit,
    LexicalSearchResult,
)
from src.services.search_documents import DOCUMENT_SCHEMA_VERSION
from src.services.search_normalization import (
    NORMALIZATION_PROFILE_ID,
    NORMALIZATION_PROFILE_VERSION,
)
from src.services.search_service import SearchRetrievalError, build_search_execute_response


class _FakeLexicalBackend:
    def __init__(
        self,
        *,
        result: LexicalSearchResult | None = None,
        error: LexicalSearchBackendError | None = None,
    ):
        self.result = result
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def search_with_profile(
        self,
        *,
        query: str,
        filters: dict[str, Any],
        top_k: int = 10,
        expected_corpus_snapshot_id: str = "",
        expected_corpus_snapshot_hash: str = "",
        expected_ranker_profile_id: str = "",
    ) -> LexicalSearchResult:
        self.calls.append(
            {
                "query": query,
                "filters": filters,
                "top_k": top_k,
                "expected_corpus_snapshot_id": expected_corpus_snapshot_id,
                "expected_corpus_snapshot_hash": expected_corpus_snapshot_hash,
                "expected_ranker_profile_id": expected_ranker_profile_id,
            }
        )
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _settings(**overrides) -> Settings:
    values = {
        "search_lexical_backend_mode": "opensearch",
        "search_active_corpus_snapshot_id": "snapshot-001",
        "search_active_corpus_snapshot_hash": "sha256:abc123",
        "search_ranker_profile_id": "lexical_bm25_v1",
    }
    values.update(overrides)
    return Settings(**values)


def _profile() -> LexicalIndexProfile:
    return LexicalIndexProfile(
        index_id="qgraph-ayah-lexical-v1",
        schema_version=LEXICAL_INDEX_PROFILE_SCHEMA_VERSION,
        backend=OPEN_SEARCH_BACKEND_NAME,
        corpus_snapshot_id="snapshot-001",
        corpus_snapshot_hash="sha256:abc123",
        document_schema_version=DOCUMENT_SCHEMA_VERSION,
        normalization_profile_id=NORMALIZATION_PROFILE_ID,
        normalization_profile_version=NORMALIZATION_PROFILE_VERSION,
        ranker_profile_id="lexical_bm25_v1",
        created_at=datetime.now(timezone.utc),
        document_count=2,
        included_languages=["ar", "en"],
        source_ids=["en-sahih", "quran-arabic"],
    )


def test_search_execute_retrieval_mode_returns_django_compatible_blocks_and_items():
    backend = _FakeLexicalBackend(
        result=LexicalSearchResult(
            profile=_profile(),
            hits=[
                LexicalSearchHit(
                    document_id="ayah:1:1:translation:en-sahih",
                    score=7.5,
                    text="In the name of Allah, the Entirely Merciful",
                    highlighted_text="In the name of Allah",
                    metadata={
                        "surah_number": 1,
                        "ayah_number": 1,
                        "ayah_global_number": 1,
                        "language_code": "en",
                        "source_id": "en-sahih",
                        "source_name": "Sahih International",
                    },
                )
            ],
        )
    )
    request = SearchExecuteRequest(
        query="mercy",
        filters={"surahs": [1]},
        output_preferences={"top_k": 5},
        context={"query_id": 123},
    )

    response = build_search_execute_response(
        request,
        settings=_settings(),
        lexical_backend=backend,
    )

    assert response.render_schema_version == "v1"
    assert response.metadata["mock"] is False
    assert response.metadata["backend"] == "open_search"
    assert response.metadata["corpus_snapshot_id"] == "snapshot-001"
    assert backend.calls == [
        {
            "query": "mercy",
            "filters": {"surahs": [1]},
            "top_k": 5,
            "expected_corpus_snapshot_id": "snapshot-001",
            "expected_corpus_snapshot_hash": "sha256:abc123",
            "expected_ranker_profile_id": "lexical_bm25_v1",
        }
    ]

    block = response.blocks[0]
    assert block.order == 0
    assert block.block_type == "results"
    assert block.payload == {"query": "mercy", "result_count": 1, "top_k": 5}
    assert block.items[0].rank == 1
    assert block.items[0].result_type == "ayah"
    assert block.items[0].score == 1.0
    assert block.items[0].title == "Surah 1, Ayah 1"
    assert block.items[0].provenance["backend"] == "open_search"
    assert block.items[0].provenance["lexical_score"] == 7.5
    assert block.items[0].match_metadata["document_id"] == "ayah:1:1:translation:en-sahih"


def _hit(score: float) -> LexicalSearchHit:
    return LexicalSearchHit(
        document_id="ayah:1:1:translation:en-sahih",
        score=score,
        text="In the name of Allah",
        highlighted_text="In the name of Allah",
        metadata={"surah_number": 1, "ayah_number": 1, "language_code": "en"},
    )


def test_search_execute_retrieval_confidence_reflects_absolute_score():
    request = SearchExecuteRequest(query="mercy", filters={}, output_preferences={}, context={})

    weak = build_search_execute_response(
        request,
        settings=_settings(),
        lexical_backend=_FakeLexicalBackend(
            result=LexicalSearchResult(profile=_profile(), hits=[_hit(0.5)])
        ),
    )
    strong = build_search_execute_response(
        request,
        settings=_settings(),
        lexical_backend=_FakeLexicalBackend(
            result=LexicalSearchResult(profile=_profile(), hits=[_hit(40.0)])
        ),
    )

    # Top item score stays relative (1.0) but overall confidence must reflect the
    # absolute match strength, not always be 1.0.
    assert weak.blocks[0].items[0].score == 1.0
    assert 0.0 < weak.overall_confidence < 0.1
    assert strong.overall_confidence > weak.overall_confidence
    assert strong.overall_confidence < 1.0


def test_search_execute_retrieval_mode_maps_missing_index_to_clear_error():
    backend = _FakeLexicalBackend(
        error=LexicalSearchBackendError(
            "OpenSearch lexical index is not available",
            reason="index_not_found",
            status_code=404,
        )
    )

    with pytest.raises(SearchRetrievalError) as exc_info:
        build_search_execute_response(
            SearchExecuteRequest(
                query="mercy",
                filters={},
                output_preferences={},
                context={},
            ),
            settings=_settings(),
            lexical_backend=backend,
        )

    assert exc_info.value.message == "OpenSearch lexical index is not available"
    assert exc_info.value.reason == "index_not_found"
    assert exc_info.value.status_code == 404


def test_search_execute_retrieval_mode_maps_stale_index_to_clear_error():
    backend = _FakeLexicalBackend(
        error=LexicalSearchBackendError(
            "OpenSearch lexical index profile does not match active retrieval configuration",
            reason="index_profile_mismatch",
            detail={
                "mismatches": {
                    "corpus_snapshot_hash": {
                        "expected": "sha256:abc123",
                        "actual": "sha256:stale",
                    }
                }
            },
        )
    )

    with pytest.raises(SearchRetrievalError) as exc_info:
        build_search_execute_response(
            SearchExecuteRequest(
                query="mercy",
                filters={},
                output_preferences={},
                context={},
            ),
            settings=_settings(),
            lexical_backend=backend,
        )

    assert exc_info.value.message == (
        "OpenSearch lexical index profile does not match active retrieval configuration"
    )
    assert exc_info.value.reason == "index_profile_mismatch"
    assert exc_info.value.detail["mismatches"]["corpus_snapshot_hash"] == {
        "expected": "sha256:abc123",
        "actual": "sha256:stale",
    }


def test_search_execute_retrieval_mode_requires_configured_opensearch_url():
    with pytest.raises(SearchRetrievalError) as exc_info:
        build_search_execute_response(
            SearchExecuteRequest(
                query="mercy",
                filters={},
                output_preferences={},
                context={},
            ),
            settings=_settings(opensearch_url=""),
        )

    assert exc_info.value.message == "OpenSearch lexical backend is not configured"
    assert exc_info.value.reason == "opensearch_not_configured"


def test_search_execute_retrieval_mode_requires_active_corpus_snapshot_config():
    with pytest.raises(SearchRetrievalError) as exc_info:
        build_search_execute_response(
            SearchExecuteRequest(
                query="mercy",
                filters={},
                output_preferences={},
                context={},
            ),
            settings=_settings(
                opensearch_url="http://opensearch:9200",
                search_active_corpus_snapshot_id="",
                search_active_corpus_snapshot_hash="",
            ),
        )

    assert exc_info.value.reason == "opensearch_active_snapshot_not_configured"
