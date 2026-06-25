from typing import Any

import pytest

from src.search.opensearch_client import (
    OpenSearchError,
    bulk_index,
    get_alias_targets,
    list_index_names,
    read_index_profile,
    search,
    swap_alias,
)


class _Resp:
    def __init__(self, status_code: int = 200, payload: Any = None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = ""

    def json(self) -> Any:
        return self._payload


class _RecordingAdapter:
    def __init__(self, *, get_payloads=None, post_payload=None):
        self._get_payloads = get_payloads or {}
        self._post_payload = post_payload if post_payload is not None else {"errors": False}
        self.posts: list[dict[str, Any]] = []

    def get(self, path: str) -> _Resp:
        if path in self._get_payloads:
            return _Resp(200, self._get_payloads[path])
        return _Resp(404)

    def post(self, path: str, *, json_payload=None, content=None, headers=None) -> _Resp:
        self.posts.append({"path": path, "json": json_payload, "content": content})
        return _Resp(200, self._post_payload)

    def put(self, path: str, *, json_payload) -> _Resp:  # pragma: no cover
        return _Resp(200, {})

    def delete(self, path: str) -> _Resp:  # pragma: no cover
        return _Resp(200, {})


def test_bulk_index_batches_by_document_count():
    adapter = _RecordingAdapter()
    sources = [(f"ayah:1:{n}:ar", {"id": f"ayah:1:{n}:ar", "content_ar": "x"}) for n in range(1, 6)]
    total = bulk_index(adapter, "idx", sources, batch_document_count=2)
    assert total == 5
    assert len(adapter.posts) == 3  # 2 + 2 + 1
    first = adapter.posts[0]["content"]
    assert '"_index": "idx"' in first
    assert first.endswith("\n")


def test_bulk_index_raises_on_document_errors():
    adapter = _RecordingAdapter(
        post_payload={"errors": True, "items": [{"index": {"status": 400}}]}
    )
    with pytest.raises(OpenSearchError) as exc_info:
        bulk_index(adapter, "idx", [("a", {"id": "a"})])
    assert exc_info.value.reason == "bulk_index_document_errors"


def test_swap_alias_builds_atomic_remove_then_add_actions():
    adapter = _RecordingAdapter()
    swap_alias(adapter, "alias", "new-index", remove_indices=["old-index"])
    actions = adapter.posts[0]["json"]["actions"]
    assert actions == [
        {"remove": {"index": "old-index", "alias": "alias"}},
        {"add": {"index": "new-index", "alias": "alias"}},
    ]


def test_get_alias_targets_returns_empty_when_missing():
    assert get_alias_targets(_RecordingAdapter(), "absent") == []


def test_get_alias_targets_lists_indices():
    adapter = _RecordingAdapter(get_payloads={"/_alias/active": {"idx-a": {}, "idx-b": {}}})
    assert get_alias_targets(adapter, "active") == ["idx-a", "idx-b"]


def test_list_index_names_parses_cat_rows():
    adapter = _RecordingAdapter(
        get_payloads={"/_cat/indices/p-*?format=json&h=index": [{"index": "p-2"}, {"index": "p-1"}]}
    )
    assert list_index_names(adapter, "p-*") == ["p-1", "p-2"]


def test_read_index_profile_resolves_alias_payload():
    adapter = _RecordingAdapter(
        get_payloads={
            "/active": {
                "concrete-index": {
                    "mappings": {
                        "_meta": {"qgraph_index_profile": {"document_schema_version": "v2"}}
                    }
                }
            }
        }
    )
    assert read_index_profile(adapter, "active") == {"document_schema_version": "v2"}


def test_search_missing_target_maps_to_index_not_found():
    class _NotFound(_RecordingAdapter):
        def post(self, path, *, json_payload=None, content=None, headers=None):
            return _Resp(404)

    with pytest.raises(OpenSearchError) as exc_info:
        search(_NotFound(), "absent", {"query": {}})
    assert exc_info.value.reason == "index_not_found"
