from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Iterator
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from src.services.search_documents import (
    DOCUMENT_SCHEMA_VERSION,
    SearchIndexDocument,
)
from src.services.search_normalization import (
    NORMALIZATION_PROFILE_ID,
    NORMALIZATION_PROFILE_VERSION,
    normalize_text,
)

OPEN_SEARCH_BACKEND_NAME = "open_search"
LEXICAL_INDEX_PROFILE_SCHEMA_VERSION = "qgraph_lexical_index_profile.v1"
DEFAULT_BULK_BATCH_DOCUMENT_COUNT = 1000
DEFAULT_BULK_BATCH_MAX_BYTES = 8 * 1024 * 1024


class LexicalSearchBackendError(Exception):
    def __init__(
        self,
        message: str,
        *,
        reason: str,
        status_code: int | None = None,
        detail: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.reason = reason
        self.status_code = status_code
        self.detail = detail or {}


class OpenSearchResponse(Protocol):
    status_code: int
    text: str

    def json(self) -> Any: ...


class OpenSearchAdapter(Protocol):
    def get(self, path: str) -> OpenSearchResponse: ...

    def delete(self, path: str) -> OpenSearchResponse: ...

    def put(self, path: str, *, json_payload: dict[str, Any]) -> OpenSearchResponse: ...

    def post(
        self,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        content: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> OpenSearchResponse: ...


class OpenSearchHTTPAdapter:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 10.0,
        http_client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._http_client = http_client or httpx.Client(timeout=timeout_seconds)

    def get(self, path: str) -> httpx.Response:
        return self._request("GET", path)

    def delete(self, path: str) -> httpx.Response:
        return self._request("DELETE", path)

    def put(self, path: str, *, json_payload: dict[str, Any]) -> httpx.Response:
        return self._request("PUT", path, json=json_payload)

    def post(
        self,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        content: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return self._request(
            "POST",
            path,
            json=json_payload,
            content=content,
            headers=headers,
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._http_client.request(method, f"{self.base_url}{path}", **kwargs)
        except httpx.RequestError as exc:
            raise LexicalSearchBackendError(
                "Failed to reach OpenSearch lexical backend",
                reason="opensearch_request_failed",
                detail={
                    "method": method,
                    "path": path,
                    "base_url": self.base_url,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            ) from exc


class LexicalIndexProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    corpus_snapshot_id: str = Field(min_length=1)
    corpus_snapshot_hash: str = Field(min_length=1)
    document_schema_version: str = Field(min_length=1)
    normalization_profile_id: str = Field(min_length=1)
    normalization_profile_version: str = Field(min_length=1)
    ranker_profile_id: str = Field(min_length=1)
    created_at: datetime
    document_count: int = Field(ge=0)
    included_languages: list[str]
    source_ids: list[str]


class LexicalSearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    score: float = Field(ge=0.0)
    text: str = ""
    highlighted_text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class LexicalSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: LexicalIndexProfile
    hits: list[LexicalSearchHit]


@dataclass(frozen=True)
class BulkIndexBatch:
    number: int
    documents: list[SearchIndexDocument]
    body: str

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def byte_size(self) -> int:
        return len(self.body.encode("utf-8"))

    @property
    def first_document_id(self) -> str:
        return self.documents[0].id

    @property
    def last_document_id(self) -> str:
        return self.documents[-1].id

    def context(self) -> dict[str, Any]:
        return {
            "batch_number": self.number,
            "document_count": self.document_count,
            "byte_size": self.byte_size,
            "first_document_id": self.first_document_id,
            "last_document_id": self.last_document_id,
        }


class OpenSearchLexicalBackend:
    def __init__(self, *, index_name: str, adapter: OpenSearchAdapter):
        self.index_name = index_name
        self.adapter = adapter

    def delete_index(self) -> None:
        response = self.adapter.delete(f"/{self.index_name}")
        if response.status_code == 404:
            return
        _raise_for_opensearch_error(
            response,
            message="Failed to delete OpenSearch lexical index",
            reason="index_delete_failed",
        )

    def index_documents(
        self,
        documents: list[SearchIndexDocument],
        profile: LexicalIndexProfile,
        *,
        recreate: bool = False,
        bulk_batch_document_count: int = DEFAULT_BULK_BATCH_DOCUMENT_COUNT,
        bulk_batch_max_bytes: int = DEFAULT_BULK_BATCH_MAX_BYTES,
    ) -> None:
        if recreate:
            self.delete_index()
        create_response = self.adapter.put(
            f"/{self.index_name}",
            json_payload=build_opensearch_index_config(profile),
        )
        _raise_for_opensearch_error(
            create_response,
            message="Failed to create OpenSearch lexical index",
            reason="index_create_failed",
        )

        for batch in iter_bulk_index_batches(
            self.index_name,
            documents,
            max_documents=bulk_batch_document_count,
            max_bytes=bulk_batch_max_bytes,
        ):
            bulk_response = self.adapter.post(
                "/_bulk",
                content=batch.body,
                headers={"Content-Type": "application/x-ndjson"},
            )
            _raise_for_opensearch_error(
                bulk_response,
                message="Failed to bulk index OpenSearch lexical documents",
                reason="bulk_index_failed",
                detail=batch.context(),
            )
            bulk_payload = _response_json(bulk_response)
            if isinstance(bulk_payload, dict) and bulk_payload.get("errors"):
                raise LexicalSearchBackendError(
                    "OpenSearch bulk indexing reported document errors",
                    reason="bulk_index_document_errors",
                    status_code=bulk_response.status_code,
                    detail={
                        **batch.context(),
                        "items": bulk_payload.get("items", [])[:5],
                    },
                )

    def search(
        self,
        *,
        query: str,
        filters: dict[str, Any],
        top_k: int = 10,
        expected_corpus_snapshot_id: str = "",
        expected_corpus_snapshot_hash: str = "",
        expected_ranker_profile_id: str = "",
    ) -> list[LexicalSearchHit]:
        return self.search_with_profile(
            query=query,
            filters=filters,
            top_k=top_k,
            expected_corpus_snapshot_id=expected_corpus_snapshot_id,
            expected_corpus_snapshot_hash=expected_corpus_snapshot_hash,
            expected_ranker_profile_id=expected_ranker_profile_id,
        ).hits

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
        profile = self.get_index_profile()
        _validate_index_profile(
            profile,
            expected_corpus_snapshot_id=expected_corpus_snapshot_id,
            expected_corpus_snapshot_hash=expected_corpus_snapshot_hash,
            expected_ranker_profile_id=expected_ranker_profile_id,
        )

        response = self.adapter.post(
            f"/{self.index_name}/_search",
            json_payload=build_search_request(query=query, filters=filters, top_k=top_k),
        )
        if response.status_code == 404:
            raise LexicalSearchBackendError(
                "OpenSearch lexical index is not available",
                reason="index_not_found",
                status_code=response.status_code,
            )
        _raise_for_opensearch_error(
            response,
            message="OpenSearch lexical search failed",
            reason="search_failed",
        )
        return LexicalSearchResult(
            profile=profile, hits=_parse_search_hits(_response_json(response))
        )

    def get_index_profile(self) -> LexicalIndexProfile:
        response = self.adapter.get(f"/{self.index_name}")
        if response.status_code == 404:
            raise LexicalSearchBackendError(
                "OpenSearch lexical index is not available",
                reason="index_not_found",
                status_code=response.status_code,
            )
        _raise_for_opensearch_error(
            response,
            message="Failed to inspect OpenSearch lexical index",
            reason="index_inspect_failed",
        )

        payload = _response_json(response)
        if not isinstance(payload, dict):
            raise LexicalSearchBackendError(
                "OpenSearch lexical index inspection returned malformed JSON",
                reason="index_profile_malformed",
            )

        index_payload = payload.get(self.index_name)
        if not isinstance(index_payload, dict):
            raise LexicalSearchBackendError(
                "OpenSearch lexical index profile is missing",
                reason="index_profile_missing",
            )
        meta = index_payload.get("mappings", {}).get("_meta", {})
        raw_profile = meta.get("qgraph_index_profile")
        if not isinstance(raw_profile, dict):
            raise LexicalSearchBackendError(
                "OpenSearch lexical index profile is missing",
                reason="index_profile_missing",
            )
        return LexicalIndexProfile.model_validate(raw_profile)


def build_lexical_index_profile(
    *,
    index_name: str,
    documents: list[SearchIndexDocument],
    ranker_profile_id: str,
) -> LexicalIndexProfile:
    if not documents:
        raise ValueError("documents must not be empty")

    corpus_snapshot_ids = {document.metadata.corpus_snapshot_id for document in documents}
    corpus_snapshot_hashes = {document.metadata.corpus_snapshot_hash for document in documents}
    document_schema_versions = {document.metadata.document_schema_version for document in documents}
    normalization_profile_ids = {
        document.metadata.normalization_profile_id for document in documents
    }
    normalization_profile_versions = {
        document.metadata.normalization_profile_version for document in documents
    }
    if len(corpus_snapshot_ids) != 1 or len(corpus_snapshot_hashes) != 1:
        raise ValueError("documents must come from a single corpus snapshot")
    if (
        len(document_schema_versions) != 1
        or len(normalization_profile_ids) != 1
        or len(normalization_profile_versions) != 1
    ):
        raise ValueError("documents must share document and normalization profiles")

    return LexicalIndexProfile(
        index_id=index_name,
        schema_version=LEXICAL_INDEX_PROFILE_SCHEMA_VERSION,
        backend=OPEN_SEARCH_BACKEND_NAME,
        corpus_snapshot_id=next(iter(corpus_snapshot_ids)),
        corpus_snapshot_hash=next(iter(corpus_snapshot_hashes)),
        document_schema_version=next(iter(document_schema_versions)),
        normalization_profile_id=next(iter(normalization_profile_ids)),
        normalization_profile_version=next(iter(normalization_profile_versions)),
        ranker_profile_id=ranker_profile_id,
        created_at=datetime.now(timezone.utc),
        document_count=len(documents),
        included_languages=sorted({document.metadata.language_code for document in documents}),
        source_ids=sorted({document.metadata.source_id for document in documents}),
    )


def build_opensearch_index_config(profile: LexicalIndexProfile) -> dict[str, Any]:
    return {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "_meta": {"qgraph_index_profile": profile.model_dump(mode="json")},
            "dynamic": "strict",
            "properties": {
                "id": {"type": "keyword"},
                "text": {"type": "text"},
                "text_ar": {"type": "text", "analyzer": "arabic"},
                "text_fa": {"type": "text", "analyzer": "persian"},
                "text_en": {"type": "text", "analyzer": "english"},
                "text_general": {"type": "text", "analyzer": "standard"},
                "normalized_text": {"type": "text", "analyzer": "standard"},
                "metadata": {
                    "properties": {
                        "corpus_snapshot_id": {"type": "keyword"},
                        "corpus_snapshot_hash": {"type": "keyword"},
                        "document_schema_version": {"type": "keyword"},
                        "normalization_profile_id": {"type": "keyword"},
                        "normalization_profile_version": {"type": "keyword"},
                        "surah_number": {"type": "integer"},
                        "ayah_number": {"type": "integer"},
                        "ayah_global_number": {"type": "integer"},
                        "language_code": {"type": "keyword"},
                        "source_id": {"type": "keyword"},
                        "source_name": {"type": "keyword"},
                        "document_kind": {"type": "keyword"},
                    }
                },
            },
        },
    }


def build_search_request(
    *,
    query: str,
    filters: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    normalized_queries = _normalized_query_variants(query)
    should_clauses: list[dict[str, Any]] = [
        {
            "multi_match": {
                "query": query,
                "fields": ["text_ar^3", "text_fa^2", "text_en^2", "text_general"],
                "type": "best_fields",
            }
        }
    ]
    should_clauses.extend(
        {"match": {"normalized_text": {"query": normalized_query}}}
        for normalized_query in normalized_queries
    )

    bool_query: dict[str, Any] = {
        "should": should_clauses,
        "minimum_should_match": 1,
    }
    filter_clauses = _build_filter_clauses(filters)
    if filter_clauses:
        bool_query["filter"] = filter_clauses

    return {
        "size": top_k,
        "query": {"bool": bool_query},
        "highlight": {
            "fields": {
                "text_ar": {},
                "text_fa": {},
                "text_en": {},
                "text_general": {},
            }
        },
    }


def _validate_index_profile(
    profile: LexicalIndexProfile,
    *,
    expected_corpus_snapshot_id: str,
    expected_corpus_snapshot_hash: str,
    expected_ranker_profile_id: str = "",
) -> None:
    mismatches: dict[str, dict[str, str]] = {}
    expected_values = {
        "backend": OPEN_SEARCH_BACKEND_NAME,
        "document_schema_version": DOCUMENT_SCHEMA_VERSION,
        "normalization_profile_id": NORMALIZATION_PROFILE_ID,
        "normalization_profile_version": NORMALIZATION_PROFILE_VERSION,
    }
    if expected_corpus_snapshot_id:
        expected_values["corpus_snapshot_id"] = expected_corpus_snapshot_id
    if expected_corpus_snapshot_hash:
        expected_values["corpus_snapshot_hash"] = expected_corpus_snapshot_hash
    if expected_ranker_profile_id:
        expected_values["ranker_profile_id"] = expected_ranker_profile_id

    profile_values = profile.model_dump(mode="json")
    for field_name, expected_value in expected_values.items():
        actual_value = profile_values.get(field_name)
        if actual_value != expected_value:
            mismatches[field_name] = {
                "expected": expected_value,
                "actual": str(actual_value),
            }

    if mismatches:
        raise LexicalSearchBackendError(
            "OpenSearch lexical index profile does not match active retrieval configuration",
            reason="index_profile_mismatch",
            detail={"mismatches": mismatches},
        )


def _build_filter_clauses(filters: dict[str, Any]) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = []
    surah_numbers = _coerce_int_filter(filters, "surahs", fallback_key="surah_ids", low=1, high=114)
    if surah_numbers:
        clauses.append({"terms": {"metadata.surah_number": surah_numbers}})

    language_codes = _coerce_string_list_filter(
        filters, "languages", fallback_key="language_codes", casefold=True
    )
    if language_codes:
        clauses.append({"terms": {"metadata.language_code": language_codes}})

    # source_id is stored verbatim (a stable external_id) on the keyword field, so
    # the filter must preserve case to match exactly.
    source_ids = _coerce_string_list_filter(filters, "source_ids", casefold=False)
    if source_ids:
        clauses.append({"terms": {"metadata.source_id": source_ids}})

    return clauses


def _coerce_int_filter(
    filters: dict[str, Any],
    key: str,
    *,
    fallback_key: str | None = None,
    low: int,
    high: int,
) -> list[int]:
    raw_values = filters.get(key)
    if raw_values is None and fallback_key is not None:
        raw_values = filters.get(fallback_key)
    if not isinstance(raw_values, list):
        return []

    values: list[int] = []
    seen: set[int] = set()
    for raw_value in raw_values:
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            continue
        if raw_value < low or raw_value > high or raw_value in seen:
            continue
        values.append(raw_value)
        seen.add(raw_value)
    return values


def _coerce_string_list_filter(
    filters: dict[str, Any],
    key: str,
    *,
    fallback_key: str | None = None,
    casefold: bool = True,
) -> list[str]:
    raw_values = filters.get(key)
    if raw_values is None and fallback_key is not None:
        raw_values = filters.get(fallback_key)
    if not isinstance(raw_values, list):
        return []

    values: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        if isinstance(raw_value, bool) or raw_value is None:
            continue
        value = str(raw_value).strip()
        if casefold:
            value = value.casefold()
        if not value or value in seen:
            continue
        values.append(value)
        seen.add(value)
    return values


def _normalized_query_variants(query: str) -> list[str]:
    variants: list[str] = []
    seen: set[str] = set()
    for language_code in ("ar", "fa", "en"):
        normalized = normalize_text(query, language_code)
        if normalized and normalized not in seen:
            variants.append(normalized)
            seen.add(normalized)
    return variants


def iter_bulk_index_batches(
    index_name: str,
    documents: list[SearchIndexDocument],
    *,
    max_documents: int = DEFAULT_BULK_BATCH_DOCUMENT_COUNT,
    max_bytes: int = DEFAULT_BULK_BATCH_MAX_BYTES,
) -> Iterator[BulkIndexBatch]:
    if max_documents < 1:
        raise ValueError("max_documents must be at least 1")
    if max_bytes < 1:
        raise ValueError("max_bytes must be at least 1")

    batch_documents: list[SearchIndexDocument] = []
    batch_lines: list[str] = []
    batch_bytes = 0
    batch_number = 1

    for document in documents:
        document_lines = _bulk_document_lines(index_name, document)
        document_bytes = sum(len(f"{line}\n".encode("utf-8")) for line in document_lines)
        if document_bytes > max_bytes:
            raise LexicalSearchBackendError(
                "A single document exceeds the OpenSearch bulk byte limit and "
                "cannot be split into a smaller batch",
                reason="bulk_document_too_large",
                detail={
                    "document_id": document.id,
                    "document_bytes": document_bytes,
                    "max_bytes": max_bytes,
                },
            )
        would_exceed_count = len(batch_documents) >= max_documents
        would_exceed_bytes = batch_documents and batch_bytes + document_bytes > max_bytes

        if would_exceed_count or would_exceed_bytes:
            yield BulkIndexBatch(
                number=batch_number,
                documents=batch_documents,
                body=_join_ndjson_lines(batch_lines),
            )
            batch_number += 1
            batch_documents = []
            batch_lines = []
            batch_bytes = 0

        batch_documents.append(document)
        batch_lines.extend(document_lines)
        batch_bytes += document_bytes

    if batch_documents:
        yield BulkIndexBatch(
            number=batch_number,
            documents=batch_documents,
            body=_join_ndjson_lines(batch_lines),
        )


def _build_bulk_body(index_name: str, documents: list[SearchIndexDocument]) -> str:
    lines: list[str] = []
    for document in documents:
        lines.extend(_bulk_document_lines(index_name, document))
    return _join_ndjson_lines(lines)


def _bulk_document_lines(index_name: str, document: SearchIndexDocument) -> list[str]:
    return [
        json.dumps(
            {"index": {"_index": index_name, "_id": document.id}},
            ensure_ascii=False,
        ),
        json.dumps(_document_source(document), ensure_ascii=False),
    ]


def _join_ndjson_lines(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"


def _document_source(document: SearchIndexDocument) -> dict[str, Any]:
    source = {
        "id": document.id,
        "text": document.text,
        "normalized_text": document.normalized_text,
        "metadata": document.metadata.model_dump(mode="json"),
    }
    language_code = document.metadata.language_code
    if language_code == "ar":
        source["text_ar"] = document.text
    elif language_code == "fa":
        source["text_fa"] = document.text
    elif language_code == "en":
        source["text_en"] = document.text
    else:
        source["text_general"] = document.text
    return source


def _parse_search_hits(payload: Any) -> list[LexicalSearchHit]:
    if not isinstance(payload, dict):
        raise LexicalSearchBackendError(
            "OpenSearch lexical search returned malformed JSON",
            reason="search_response_malformed",
        )
    raw_hits = payload.get("hits", {}).get("hits", [])
    if not isinstance(raw_hits, list):
        raise LexicalSearchBackendError(
            "OpenSearch lexical search returned malformed hits",
            reason="search_response_malformed",
        )

    hits: list[LexicalSearchHit] = []
    for raw_hit in raw_hits:
        if not isinstance(raw_hit, dict):
            continue
        source = raw_hit.get("_source", {})
        if not isinstance(source, dict):
            source = {}
        document_id = str(raw_hit.get("_id") or source.get("id") or "").strip()
        if not document_id:
            continue
        hits.append(
            LexicalSearchHit(
                document_id=document_id,
                score=float(raw_hit.get("_score") or 0.0),
                text=str(source.get("text") or ""),
                highlighted_text=_extract_highlight(raw_hit) or str(source.get("text") or ""),
                metadata=source.get("metadata") if isinstance(source.get("metadata"), dict) else {},
            )
        )
    return hits


_HIGHLIGHT_FIELD_PREFERENCE = ("text_ar", "text_fa", "text_en", "text_general")


def _extract_highlight(raw_hit: dict[str, Any]) -> str:
    highlight = raw_hit.get("highlight")
    if not isinstance(highlight, dict):
        return ""
    # Prefer the human-readable language fields in a stable order; never surface
    # the normalized (punctuation-stripped/casefolded) variant.
    for field_name in _HIGHLIGHT_FIELD_PREFERENCE:
        values = highlight.get(field_name)
        if isinstance(values, list) and values:
            return str(values[0])
    return ""


def _raise_for_opensearch_error(
    response: OpenSearchResponse,
    *,
    message: str,
    reason: str,
    detail: dict[str, Any] | None = None,
) -> None:
    if response.status_code < 400:
        return
    error_detail = {"body": response.text}
    if detail:
        error_detail.update(detail)
    raise LexicalSearchBackendError(
        message,
        reason=reason,
        status_code=response.status_code,
        detail=error_detail,
    )


def _response_json(response: OpenSearchResponse) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise LexicalSearchBackendError(
            "OpenSearch returned invalid JSON",
            reason="opensearch_invalid_json",
            status_code=response.status_code,
            detail={"message": str(exc)},
        ) from exc
