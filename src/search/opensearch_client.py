"""Reusable OpenSearch HTTP boundary.

The low-level transport (an httpx-backed adapter, the request/response protocols, the error type, and
the small response helpers) lives here so both the indexing path (build/activate) and the query path
(retrieval) share one client. Higher-level operations (create index, bulk, alias swap, search, read
the index ``_meta`` profile) are layered on top.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any, Protocol

import httpx

DEFAULT_BULK_BATCH_DOCUMENT_COUNT = 1000
DEFAULT_BULK_BATCH_MAX_BYTES = 8 * 1024 * 1024


class OpenSearchError(Exception):
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
        auth: tuple[str, str] | None = None,
        verify: bool | str = True,
        http_client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._http_client = http_client or httpx.Client(
            timeout=timeout_seconds, auth=auth, verify=verify
        )

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
            raise OpenSearchError(
                "Failed to reach OpenSearch",
                reason="opensearch_request_failed",
                detail={
                    "method": method,
                    "path": path,
                    "base_url": self.base_url,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            ) from exc


def raise_for_opensearch_error(
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
    raise OpenSearchError(
        message,
        reason=reason,
        status_code=response.status_code,
        detail=error_detail,
    )


def response_json(response: OpenSearchResponse) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise OpenSearchError(
            "OpenSearch returned invalid JSON",
            reason="opensearch_invalid_json",
            status_code=response.status_code,
            detail={"message": str(exc)},
        ) from exc


def search(adapter: OpenSearchAdapter, target: str, body: dict[str, Any]) -> dict[str, Any]:
    """Run a ``_search`` against an index or alias and return the parsed response."""
    response = adapter.post(f"/{target}/_search", json_payload=body)
    if response.status_code == 404:
        raise OpenSearchError(
            "OpenSearch index or alias is not available",
            reason="index_not_found",
            status_code=response.status_code,
        )
    raise_for_opensearch_error(response, message="OpenSearch search failed", reason="search_failed")
    payload = response_json(response)
    if not isinstance(payload, dict):
        raise OpenSearchError(
            "OpenSearch search returned malformed JSON", reason="search_response_malformed"
        )
    return payload


def read_index_profile(adapter: OpenSearchAdapter, target: str) -> dict[str, Any]:
    """Read ``mappings._meta.qgraph_index_profile`` from an index or alias.

    A ``GET`` on an alias returns ``{concrete_index_name: {...}}``; the single entry is used, so this
    works for both the serving alias and a concrete physical index.
    """
    response = adapter.get(f"/{target}")
    if response.status_code == 404:
        raise OpenSearchError(
            "OpenSearch index or alias is not available",
            reason="index_not_found",
            status_code=response.status_code,
        )
    raise_for_opensearch_error(
        response, message="Failed to inspect OpenSearch index", reason="index_inspect_failed"
    )
    payload = response_json(response)
    if not isinstance(payload, dict) or not payload:
        raise OpenSearchError("OpenSearch index profile is missing", reason="index_profile_missing")
    index_payload = next(iter(payload.values()))
    profile = None
    if isinstance(index_payload, dict):
        meta = index_payload.get("mappings", {}).get("_meta", {})
        profile = meta.get("qgraph_index_profile")
    if not isinstance(profile, dict):
        raise OpenSearchError("OpenSearch index profile is missing", reason="index_profile_missing")
    return profile


def create_index(adapter: OpenSearchAdapter, name: str, body: dict[str, Any]) -> None:
    response = adapter.put(f"/{name}", json_payload=body)
    raise_for_opensearch_error(
        response, message="Failed to create OpenSearch index", reason="index_create_failed"
    )


def delete_index(adapter: OpenSearchAdapter, name: str) -> None:
    response = adapter.delete(f"/{name}")
    if response.status_code == 404:
        return
    raise_for_opensearch_error(
        response, message="Failed to delete OpenSearch index", reason="index_delete_failed"
    )


def refresh(adapter: OpenSearchAdapter, name: str) -> None:
    """Make recently indexed documents searchable (used right before build validation)."""
    response = adapter.post(f"/{name}/_refresh")
    raise_for_opensearch_error(
        response, message="Failed to refresh OpenSearch index", reason="index_refresh_failed"
    )


def bulk_index(
    adapter: OpenSearchAdapter,
    name: str,
    sources: Iterable[tuple[str, dict[str, Any]]],
    *,
    batch_document_count: int = DEFAULT_BULK_BATCH_DOCUMENT_COUNT,
    batch_max_bytes: int = DEFAULT_BULK_BATCH_MAX_BYTES,
) -> int:
    """Bulk-index ``(document_id, source)`` pairs; returns the number indexed."""
    total = 0
    for body, count in _iter_bulk_bodies(name, sources, batch_document_count, batch_max_bytes):
        response = adapter.post(
            "/_bulk", content=body, headers={"Content-Type": "application/x-ndjson"}
        )
        raise_for_opensearch_error(
            response,
            message="Failed to bulk index OpenSearch documents",
            reason="bulk_index_failed",
        )
        payload = response_json(response)
        if isinstance(payload, dict) and payload.get("errors"):
            raise OpenSearchError(
                "OpenSearch bulk indexing reported document errors",
                reason="bulk_index_document_errors",
                detail={"items": payload.get("items", [])[:5]},
            )
        total += count
    return total


def list_index_names(adapter: OpenSearchAdapter, pattern: str) -> list[str]:
    response = adapter.get(f"/_cat/indices/{pattern}?format=json&h=index")
    if response.status_code == 404:
        return []
    raise_for_opensearch_error(
        response, message="Failed to list OpenSearch indices", reason="index_list_failed"
    )
    payload = response_json(response)
    if not isinstance(payload, list):
        return []
    names = [
        str(row.get("index", "")).strip()
        for row in payload
        if isinstance(row, dict) and row.get("index")
    ]
    return sorted(name for name in names if name)


def get_alias_targets(adapter: OpenSearchAdapter, alias: str) -> list[str]:
    response = adapter.get(f"/_alias/{alias}")
    if response.status_code == 404:
        return []
    raise_for_opensearch_error(
        response, message="Failed to read OpenSearch alias", reason="alias_read_failed"
    )
    payload = response_json(response)
    if not isinstance(payload, dict):
        return []
    return sorted(payload.keys())


def swap_alias(
    adapter: OpenSearchAdapter,
    alias: str,
    new_index: str,
    *,
    remove_indices: Iterable[str] = (),
) -> None:
    """Atomically repoint ``alias`` at ``new_index`` (remove old targets + add new in one call)."""
    actions: list[dict[str, Any]] = [
        {"remove": {"index": index, "alias": alias}} for index in remove_indices
    ]
    actions.append({"add": {"index": new_index, "alias": alias}})
    response = adapter.post("/_aliases", json_payload={"actions": actions})
    raise_for_opensearch_error(
        response, message="Failed to swap OpenSearch alias", reason="alias_swap_failed"
    )


def _iter_bulk_bodies(
    name: str,
    sources: Iterable[tuple[str, dict[str, Any]]],
    max_documents: int,
    max_bytes: int,
) -> Iterator[tuple[str, int]]:
    lines: list[str] = []
    count = 0
    size = 0
    for document_id, source in sources:
        action = json.dumps({"index": {"_index": name, "_id": document_id}}, ensure_ascii=False)
        document = json.dumps(source, ensure_ascii=False)
        pair_bytes = len(f"{action}\n{document}\n".encode())
        if count and (count >= max_documents or size + pair_bytes > max_bytes):
            yield "\n".join(lines) + "\n", count
            lines, count, size = [], 0, 0
        lines.append(action)
        lines.append(document)
        count += 1
        size += pair_bytes
    if count:
        yield "\n".join(lines) + "\n", count


def build_opensearch_adapter(
    *,
    url: str,
    timeout_seconds: float = 10.0,
    username: str = "",
    password: str = "",
    verify: bool | str = True,
) -> OpenSearchHTTPAdapter:
    if not url:
        raise OpenSearchError(
            "OpenSearch URL is not configured", reason="opensearch_not_configured"
        )
    return OpenSearchHTTPAdapter(
        base_url=url,
        timeout_seconds=timeout_seconds,
        auth=(username, password),
        verify=verify,
    )
