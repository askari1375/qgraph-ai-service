from typing import Any

from src.config import Settings
from src.search.indexing.documents import DOCUMENT_SCHEMA_VERSION
from src.search.indexing.mapping import ANALYSIS_PROFILE_VERSION
from src.search.indexing.normalization import NORMALIZATION_PROFILE_VERSION
from src.services.search_service import check_search_readiness


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {"opensearch_url": "http://opensearch:9200"}
    values.update(overrides)
    return Settings(**values)


def _profile(**overrides: Any) -> dict[str, Any]:
    profile = {
        "document_schema_version": DOCUMENT_SCHEMA_VERSION,
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
    def __init__(self, *, targets, profile=None, hits=None):
        self._targets = targets
        self._profile = profile if profile is not None else _profile()
        self._hits = hits if hits is not None else [{"_id": "ayah:1:1:ar"}]

    def get(self, path: str) -> _Resp:
        if path.startswith("/_alias/"):
            if not self._targets:
                return _Resp(404)
            return _Resp(200, {target: {} for target in self._targets})
        return _Resp(
            200,
            {self._targets[0]: {"mappings": {"_meta": {"qgraph_index_profile": self._profile}}}},
        )

    def post(self, path: str, *, json_payload=None, content=None, headers=None) -> _Resp:
        return _Resp(200, {"hits": {"hits": self._hits}})

    def put(self, path: str, *, json_payload) -> _Resp:  # pragma: no cover - unused
        return _Resp(200, {})

    def delete(self, path: str) -> _Resp:  # pragma: no cover - unused
        return _Resp(200, {})


def _check_names(readiness) -> dict[str, bool]:
    return {check.name: check.ok for check in readiness.checks}


def test_readiness_ok_when_alias_compatible_and_smoke_query_hits():
    readiness = check_search_readiness(_settings(), _FakeAdapter(targets=["idx-001"]))
    assert readiness.ready is True
    assert readiness.active_index == "idx-001"
    assert _check_names(readiness) == {
        "alias_single_target": True,
        "index_profile_compatible": True,
        "smoke_query": True,
    }


def test_readiness_not_ready_when_opensearch_url_unset():
    readiness = check_search_readiness(_settings(opensearch_url=""))
    assert readiness.ready is False
    assert _check_names(readiness) == {"opensearch_configured": False}


def test_readiness_not_ready_when_alias_has_no_target():
    readiness = check_search_readiness(_settings(), _FakeAdapter(targets=[]))
    assert readiness.ready is False
    assert _check_names(readiness)["alias_single_target"] is False


def test_readiness_not_ready_when_alias_has_multiple_targets():
    readiness = check_search_readiness(_settings(), _FakeAdapter(targets=["idx-a", "idx-b"]))
    assert readiness.ready is False
    assert _check_names(readiness)["alias_single_target"] is False


def test_readiness_not_ready_when_index_profile_incompatible():
    adapter = _FakeAdapter(targets=["idx-001"], profile=_profile(analysis_profile_version="stale"))
    readiness = check_search_readiness(_settings(), adapter)
    assert readiness.ready is False
    assert _check_names(readiness)["index_profile_compatible"] is False


def test_readiness_not_ready_when_smoke_query_returns_no_hits():
    readiness = check_search_readiness(_settings(), _FakeAdapter(targets=["idx-001"], hits=[]))
    assert readiness.ready is False
    assert _check_names(readiness)["smoke_query"] is False
