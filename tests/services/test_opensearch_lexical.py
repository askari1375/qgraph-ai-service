from typing import Any

import httpx
import pytest

from src.api.schemas.corpus import QuranCorpusSnapshot
from src.services.opensearch_lexical import (
    LexicalSearchBackendError,
    OpenSearchHTTPAdapter,
    OpenSearchLexicalBackend,
    build_lexical_index_profile,
    build_search_request,
    resolve_opensearch_auth,
    resolve_opensearch_verify,
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
        delete_response: _FakeResponse | None = None,
    ):
        self.get_response = get_response or _FakeResponse()
        self.put_response = put_response or _FakeResponse()
        self.post_responses = post_responses or [_FakeResponse()]
        self.delete_response = delete_response or _FakeResponse()
        self.calls: list[dict[str, Any]] = []

    def get(self, path: str) -> _FakeResponse:
        self.calls.append({"method": "GET", "path": path})
        return self.get_response

    def delete(self, path: str) -> _FakeResponse:
        self.calls.append({"method": "DELETE", "path": path})
        return self.delete_response

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
            "schema_version": "qgraph-corpus-snapshot-v1",
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


def _documents_with_count(count: int):
    base_document = _documents()[0]
    return [
        base_document.model_copy(update={"id": f"ayah:1:{index}:ar"})
        for index in range(1, count + 1)
    ]


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


def test_opensearch_backend_sends_multiple_bulk_batches_when_document_limit_is_exceeded():
    documents = _documents_with_count(5)
    profile = build_lexical_index_profile(
        index_name=INDEX_NAME,
        documents=documents,
        ranker_profile_id="lexical_bm25_v1",
    )
    adapter = _FakeAdapter(
        put_response=_FakeResponse(200, {"acknowledged": True}),
        post_responses=[
            _FakeResponse(200, {"errors": False}),
            _FakeResponse(200, {"errors": False}),
            _FakeResponse(200, {"errors": False}),
        ],
    )
    backend = OpenSearchLexicalBackend(index_name=INDEX_NAME, adapter=adapter)

    backend.index_documents(
        documents,
        profile,
        bulk_batch_document_count=2,
        bulk_batch_max_bytes=1024 * 1024,
    )

    bulk_calls = [call for call in adapter.calls if call["path"] == "/_bulk"]
    assert len(bulk_calls) == 3
    assert [call["content"].count('"index"') for call in bulk_calls] == [2, 2, 1]
    for call in bulk_calls:
        assert call["headers"] == {"Content-Type": "application/x-ndjson"}
        assert call["content"].endswith("\n")
        lines = call["content"].splitlines()
        assert len(lines) % 2 == 0
        assert all(lines)


def test_opensearch_backend_bulk_status_failure_includes_batch_context():
    documents = _documents_with_count(3)
    profile = build_lexical_index_profile(
        index_name=INDEX_NAME,
        documents=documents,
        ranker_profile_id="lexical_bm25_v1",
    )
    adapter = _FakeAdapter(
        put_response=_FakeResponse(200, {"acknowledged": True}),
        post_responses=[
            _FakeResponse(200, {"errors": False}),
            _FakeResponse(413, text="request entity too large"),
        ],
    )
    backend = OpenSearchLexicalBackend(index_name=INDEX_NAME, adapter=adapter)

    with pytest.raises(LexicalSearchBackendError) as exc_info:
        backend.index_documents(
            documents,
            profile,
            bulk_batch_document_count=2,
            bulk_batch_max_bytes=1024 * 1024,
        )

    assert exc_info.value.message == "Failed to bulk index OpenSearch lexical documents"
    assert exc_info.value.reason == "bulk_index_failed"
    assert exc_info.value.status_code == 413
    assert exc_info.value.detail["body"] == "request entity too large"
    assert exc_info.value.detail["batch_number"] == 2
    assert exc_info.value.detail["document_count"] == 1
    assert exc_info.value.detail["first_document_id"] == "ayah:1:3:ar"
    assert exc_info.value.detail["last_document_id"] == "ayah:1:3:ar"
    assert exc_info.value.detail["byte_size"] > 0


def test_opensearch_backend_recreate_deletes_then_creates_index():
    documents = _documents()
    profile = _profile()
    adapter = _FakeAdapter(
        delete_response=_FakeResponse(200, {"acknowledged": True}),
        put_response=_FakeResponse(200, {"acknowledged": True}),
        post_responses=[_FakeResponse(200, {"errors": False})],
    )
    backend = OpenSearchLexicalBackend(index_name=INDEX_NAME, adapter=adapter)

    backend.index_documents(documents, profile, recreate=True)

    methods = [call["method"] for call in adapter.calls]
    assert methods[0] == "DELETE"
    assert adapter.calls[0]["path"] == f"/{INDEX_NAME}"
    assert methods[1] == "PUT"


def test_opensearch_backend_recreate_tolerates_missing_index():
    adapter = _FakeAdapter(
        delete_response=_FakeResponse(404, text="index_not_found"),
        put_response=_FakeResponse(200, {"acknowledged": True}),
        post_responses=[_FakeResponse(200, {"errors": False})],
    )
    backend = OpenSearchLexicalBackend(index_name=INDEX_NAME, adapter=adapter)

    backend.index_documents(_documents(), _profile(), recreate=True)

    assert [call["method"] for call in adapter.calls][:2] == ["DELETE", "PUT"]


def test_opensearch_backend_create_on_existing_index_surfaces_clear_error():
    adapter = _FakeAdapter(
        put_response=_FakeResponse(400, text="resource_already_exists_exception"),
    )
    backend = OpenSearchLexicalBackend(index_name=INDEX_NAME, adapter=adapter)

    with pytest.raises(LexicalSearchBackendError) as exc_info:
        backend.index_documents(_documents(), _profile())

    assert exc_info.value.reason == "index_create_failed"
    assert exc_info.value.status_code == 400


def test_iter_bulk_index_batches_rejects_oversized_single_document():
    documents = _documents()
    profile = _profile()
    adapter = _FakeAdapter(put_response=_FakeResponse(200, {"acknowledged": True}))
    backend = OpenSearchLexicalBackend(index_name=INDEX_NAME, adapter=adapter)

    with pytest.raises(LexicalSearchBackendError) as exc_info:
        backend.index_documents(documents, profile, bulk_batch_max_bytes=10)

    assert exc_info.value.reason == "bulk_document_too_large"
    assert exc_info.value.detail["max_bytes"] == 10
    assert exc_info.value.detail["document_id"]


def test_source_id_filter_preserves_case_while_languages_are_lowercased():
    request = build_search_request(
        query="mercy",
        filters={"source_ids": ["EN-Sahih"], "languages": ["EN"]},
        top_k=10,
    )
    filter_clauses = request["query"]["bool"]["filter"]
    terms = {
        next(iter(clause["terms"])): next(iter(clause["terms"].values()))
        for clause in filter_clauses
    }
    assert terms["metadata.source_id"] == ["EN-Sahih"]
    assert terms["metadata.language_code"] == ["en"]


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


def test_opensearch_http_adapter_maps_connection_errors_to_clear_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 61] Connection refused", request=request)

    backend = OpenSearchLexicalBackend(
        index_name=INDEX_NAME,
        adapter=OpenSearchHTTPAdapter(
            base_url="http://127.0.0.1:9200",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
    )

    with pytest.raises(LexicalSearchBackendError) as exc_info:
        backend.search(query="mercy", filters={}, top_k=5)

    assert exc_info.value.message == "Failed to reach OpenSearch lexical backend"
    assert exc_info.value.reason == "opensearch_request_failed"
    assert exc_info.value.detail["method"] == "GET"
    assert exc_info.value.detail["path"] == f"/{INDEX_NAME}"
    assert exc_info.value.detail["base_url"] == "http://127.0.0.1:9200"
    assert exc_info.value.detail["error_type"] == "ConnectError"
    assert "Connection refused" in exc_info.value.detail["error"]


def test_resolve_opensearch_auth_returns_tuple_only_when_username_set():
    assert resolve_opensearch_auth("svc", "secret") == ("svc", "secret")
    assert resolve_opensearch_auth("", "secret") is None
    assert resolve_opensearch_auth("", "") is None


def test_resolve_opensearch_verify_prefers_ca_path_then_falls_back_to_flag():
    assert resolve_opensearch_verify(True, "/etc/ssl/ca.pem") == "/etc/ssl/ca.pem"
    assert resolve_opensearch_verify(False, "/etc/ssl/ca.pem") == "/etc/ssl/ca.pem"
    assert resolve_opensearch_verify(True, "") is True
    assert resolve_opensearch_verify(False, "") is False
