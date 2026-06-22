from typing import Any

import pytest

from src.api.schemas.corpus import QuranCorpusSnapshot
from src.services.opensearch_lexical import (
    LexicalSearchBackendError,
    OpenSearchLexicalBackend,
    build_lexical_index_profile,
)
from src.services.search_documents import build_search_documents

INDEX_NAME = "qgraph-ayah-lexical-v1"


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: Any = None, text: str = ""):
        self.status_code = status_code
        self.payload = payload if payload is not None else {}
        self.text = text

    def json(self) -> Any:
        return self.payload


class _FakeAdapter:
    def __init__(
        self,
        *,
        get_response: _FakeResponse | None = None,
        put_response: _FakeResponse | None = None,
        post_responses: list[_FakeResponse] | None = None,
    ):
        self.get_response = get_response or _FakeResponse()
        self.put_response = put_response or _FakeResponse()
        self.post_responses = post_responses or [_FakeResponse()]
        self.calls: list[dict[str, Any]] = []

    def get(self, path: str) -> _FakeResponse:
        self.calls.append({"method": "GET", "path": path})
        return self.get_response

    def put(self, path: str, *, json_payload: dict[str, Any]) -> _FakeResponse:
        self.calls.append({"method": "PUT", "path": path, "json": json_payload})
        return self.put_response

    def post(
        self,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        content: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> _FakeResponse:
        self.calls.append(
            {
                "method": "POST",
                "path": path,
                "json": json_payload,
                "content": content,
                "headers": headers or {},
            }
        )
        return self.post_responses.pop(0)


def _snapshot() -> QuranCorpusSnapshot:
    return QuranCorpusSnapshot.model_validate(
        {
            "schema_version": "quran-corpus-snapshot.v1",
            "corpus_snapshot_id": "snapshot-001",
            "corpus_snapshot_hash": "sha256:abc123",
            "produced_at": "2026-06-22T10:00:00Z",
            "filters": {},
            "counts": {"ayahs": 1, "translations": 1},
            "translation_sources": [],
            "surahs": [],
            "ayahs": [
                {
                    "surah_number": 1,
                    "ayah_number": 1,
                    "ayah_global_number": 1,
                    "text_ar": "بسم الله الرحمن الرحيم",
                    "translations": [
                        {
                            "language_code": "en",
                            "source_id": "en-sahih",
                            "source_name": "Sahih International",
                            "text": "In the name of Allah, the Entirely Merciful",
                        }
                    ],
                }
            ],
        }
    )


def _documents():
    return build_search_documents(_snapshot())


def _profile():
    return build_lexical_index_profile(
        index_name=INDEX_NAME,
        documents=_documents(),
        ranker_profile_id="lexical_bm25_v1",
    )


def test_opensearch_backend_builds_index_mapping_and_bulk_request():
    documents = _documents()
    profile = _profile()
    adapter = _FakeAdapter(
        put_response=_FakeResponse(200, {"acknowledged": True}),
        post_responses=[_FakeResponse(200, {"errors": False})],
    )
    backend = OpenSearchLexicalBackend(index_name=INDEX_NAME, adapter=adapter)

    backend.index_documents(documents, profile)

    put_call = adapter.calls[0]
    assert put_call["method"] == "PUT"
    assert put_call["path"] == f"/{INDEX_NAME}"
    mapping = put_call["json"]["mappings"]
    assert mapping["_meta"]["qgraph_index_profile"]["corpus_snapshot_id"] == "snapshot-001"
    assert mapping["properties"]["text_ar"]["analyzer"] == "arabic"
    assert mapping["properties"]["text_fa"]["analyzer"] == "persian"
    assert mapping["properties"]["text_en"]["analyzer"] == "english"

    bulk_call = adapter.calls[1]
    assert bulk_call["path"] == "/_bulk"
    assert bulk_call["headers"] == {"Content-Type": "application/x-ndjson"}
    assert '"_id": "ayah:1:1:ar"' in bulk_call["content"]
    assert '"_id": "ayah:1:1:translation:en-sahih"' in bulk_call["content"]
    assert '"text_ar": "بسم الله الرحمن الرحيم"' in bulk_call["content"]
    assert '"text_en": "In the name of Allah, the Entirely Merciful"' in bulk_call["content"]


def test_opensearch_backend_builds_search_request_and_returns_hits():
    profile = _profile()
    adapter = _FakeAdapter(
        get_response=_FakeResponse(
            200,
            {
                INDEX_NAME: {
                    "mappings": {"_meta": {"qgraph_index_profile": profile.model_dump(mode="json")}}
                }
            },
        ),
        post_responses=[
            _FakeResponse(
                200,
                {
                    "hits": {
                        "hits": [
                            {
                                "_id": "ayah:1:1:translation:en-sahih",
                                "_score": 7.5,
                                "_source": {
                                    "id": "ayah:1:1:translation:en-sahih",
                                    "text": "In the name of Allah, the Entirely Merciful",
                                    "metadata": {
                                        "surah_number": 1,
                                        "ayah_number": 1,
                                        "ayah_global_number": 1,
                                        "language_code": "en",
                                        "source_id": "en-sahih",
                                        "corpus_snapshot_id": "snapshot-001",
                                        "corpus_snapshot_hash": "sha256:abc123",
                                    },
                                },
                                "highlight": {"text_en": ["In the name of Allah"]},
                            }
                        ]
                    }
                },
            )
        ],
    )
    backend = OpenSearchLexicalBackend(index_name=INDEX_NAME, adapter=adapter)

    hits = backend.search(
        query="mercy",
        filters={"surahs": [1, True, 999], "languages": ["en"]},
        top_k=5,
        expected_corpus_snapshot_id="snapshot-001",
        expected_corpus_snapshot_hash="sha256:abc123",
    )

    search_call = adapter.calls[1]
    assert search_call["path"] == f"/{INDEX_NAME}/_search"
    assert search_call["json"]["size"] == 5
    assert search_call["json"]["query"]["bool"]["filter"] == [
        {"terms": {"metadata.surah_number": [1]}},
        {"terms": {"metadata.language_code": ["en"]}},
    ]
    assert hits[0].document_id == "ayah:1:1:translation:en-sahih"
    assert hits[0].score == 7.5
    assert hits[0].highlighted_text == "In the name of Allah"
    assert hits[0].metadata["corpus_snapshot_hash"] == "sha256:abc123"


def test_opensearch_backend_missing_index_is_clear_error():
    backend = OpenSearchLexicalBackend(
        index_name=INDEX_NAME,
        adapter=_FakeAdapter(get_response=_FakeResponse(404, text="missing")),
    )

    with pytest.raises(LexicalSearchBackendError) as exc_info:
        backend.search(query="mercy", filters={}, top_k=5)

    assert exc_info.value.message == "OpenSearch lexical index is not available"
    assert exc_info.value.reason == "index_not_found"
    assert exc_info.value.status_code == 404
