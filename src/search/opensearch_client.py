"""Reusable OpenSearch HTTP boundary.

The low-level transport (an httpx-backed adapter, the request/response protocols, the error type, and
the small response helpers) lives here so both the indexing path (build/activate) and the query path
(retrieval) share one client. Higher-level operations (create index, bulk, alias swap, search, read
the index ``_meta`` profile) are layered on top.
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx


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
