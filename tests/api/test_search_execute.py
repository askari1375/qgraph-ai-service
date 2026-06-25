from src.api.schemas.search import SearchExecuteResponse, SearchResponseBlock
from src.services.search_service import SearchRetrievalError

_MARKDOWN = "| # | Reference | Match | Source |\n| ---: | --- | --- | --- |\n| 1 | Surah 1, Ayah 1 | بسم الله | Quran Arabic |"


def _canned_response(_payload) -> SearchExecuteResponse:
    return SearchExecuteResponse(
        title="Search results for mercy",
        overall_confidence=0.5,
        render_schema_version="v1",
        metadata={"backend": "open_search", "corpus_snapshot_id": "snapshot-001"},
        blocks=[
            SearchResponseBlock(
                order=0,
                block_type="markdown",
                title="Lexical matches",
                payload={"headline": "1 result(s)", "content": _MARKDOWN},
                confidence=0.5,
                provenance={"backend": "open_search"},
                warning_text="",
                items=[],
            )
        ],
    )


def test_search_execute_endpoint_returns_schema_shape(client, search_execute_payload, monkeypatch):
    monkeypatch.setattr("src.api.search.build_search_execute_response", _canned_response)
    response = client.post("/v1/search/execute", json=search_execute_payload)
    assert response.status_code == 200

    payload = response.json()
    assert set(payload) == {
        "title",
        "overall_confidence",
        "render_schema_version",
        "metadata",
        "blocks",
    }
    assert payload["render_schema_version"] == "v1"
    assert payload["metadata"]["backend"] == "open_search"
    assert payload["blocks"][0]["block_type"] == "markdown"
    assert isinstance(payload["blocks"][0]["payload"]["content"], str)


def test_search_execute_block_orders_are_unique(client, search_execute_payload, monkeypatch):
    monkeypatch.setattr("src.api.search.build_search_execute_response", _canned_response)
    response = client.post("/v1/search/execute", json=search_execute_payload)
    assert response.status_code == 200

    orders = [block["order"] for block in response.json()["blocks"]]
    assert len(orders) == len(set(orders))


def test_search_execute_retrieval_error_returns_service_error(
    client,
    search_execute_payload,
    monkeypatch,
):
    def raise_retrieval_error(_payload):
        raise SearchRetrievalError(
            "OpenSearch lexical index is not available",
            reason="index_not_found",
            status_code=404,
        )

    monkeypatch.setattr("src.api.search.build_search_execute_response", raise_retrieval_error)

    response = client.post("/v1/search/execute", json=search_execute_payload)

    assert response.status_code == 503
    payload = response.json()
    assert payload["error"] == "http_error"
    assert payload["detail"] == {
        "message": "OpenSearch lexical index is not available",
        "reason": "index_not_found",
        "backend_status_code": 404,
    }
