from src.api.schemas.search import SearchReadinessCheck, SearchReadinessResponse


def test_readiness_endpoint_returns_503_when_not_ready(client):
    # No OpenSearch URL is configured in the test environment, so search is not ready.
    response = client.get("/v1/search/readiness")
    assert response.status_code == 503

    payload = response.json()
    assert payload["ready"] is False
    assert payload["alias"]
    # The body is returned unwrapped (not via the generic http_error envelope) for monitoring.
    assert "checks" in payload


def test_readiness_endpoint_returns_200_when_ready(client, monkeypatch):
    ready = SearchReadinessResponse(
        ready=True,
        alias="qgraph-ayah-lexical-active",
        active_index="qgraph-ayah-lexical-20260627-001",
        checks=[SearchReadinessCheck(name="smoke_query", ok=True, detail={"hit_count": 3})],
    )
    monkeypatch.setattr("src.api.search.check_search_readiness", lambda *args, **kwargs: ready)

    response = client.get("/v1/search/readiness")
    assert response.status_code == 200
    assert response.json()["ready"] is True
