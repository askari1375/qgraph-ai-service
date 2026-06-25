import pytest

from src.api.schemas.search import SearchExecuteResponse, SearchResponseBlock


def _canned_response(_request, settings=None) -> SearchExecuteResponse:
    return SearchExecuteResponse(
        title="Search results",
        overall_confidence=0.5,
        render_schema_version="v1",
        metadata={"backend": "open_search"},
        blocks=[
            SearchResponseBlock(
                order=0,
                block_type="markdown",
                title="Lexical matches",
                payload={
                    "headline": "1 result(s)",
                    "content": "| # | Reference | Match | Source |",
                },
                confidence=0.5,
                provenance={"backend": "open_search"},
                warning_text="",
                items=[],
            )
        ],
    )


@pytest.fixture(autouse=True)
def _stub_search_builder(monkeypatch):
    # Jobs precompute the execute response at create time; stub retrieval so the lifecycle tests
    # don't need a live OpenSearch.
    monkeypatch.setattr("src.services.search_jobs.build_search_execute_response", _canned_response)


def test_search_job_create_endpoint_returns_expected_shape(client, search_job_create_payload):
    response = client.post("/v1/search/jobs", json=search_job_create_payload)
    assert response.status_code == 202

    payload = response.json()
    assert set(payload) == {"job_id", "status", "created_at", "poll_after_seconds"}
    assert payload["job_id"].startswith("job_")
    assert payload["status"] == "queued"
    assert payload["poll_after_seconds"] >= 0


def test_search_job_create_endpoint_reuses_active_job_for_same_idempotency_key(
    client,
    search_job_create_payload,
):
    first_response = client.post("/v1/search/jobs", json=search_job_create_payload)
    second_response = client.post("/v1/search/jobs", json=search_job_create_payload)
    assert first_response.status_code == 202
    assert second_response.status_code == 202
    assert second_response.json()["job_id"] == first_response.json()["job_id"]


def test_search_job_status_endpoint_returns_expected_shape(client, search_job_create_payload):
    create_response = client.post("/v1/search/jobs", json=search_job_create_payload)
    job_id = create_response.json()["job_id"]

    response = client.get(f"/v1/search/jobs/{job_id}")
    assert response.status_code == 200
    response = client.get(f"/v1/search/jobs/{job_id}")
    assert response.status_code == 200

    payload = response.json()
    assert set(payload) == {
        "job_id",
        "status",
        "created_at",
        "started_at",
        "completed_at",
        "poll_after_seconds",
        "result_available",
        "error",
        "progress",
    }
    assert payload["status"] in {"queued", "running", "succeeded", "failed", "canceled"}
    assert {"stage", "percent"} <= set(payload["progress"])


def test_search_job_result_endpoint_returns_409_while_job_is_not_ready(
    client,
    search_job_create_payload,
):
    create_response = client.post("/v1/search/jobs", json=search_job_create_payload)
    job_id = create_response.json()["job_id"]

    response = client.get(f"/v1/search/jobs/{job_id}/result")
    assert response.status_code == 409
    payload = response.json()
    assert payload["error"] == "http_error"
    assert payload["detail"]["message"] == "Search job result not ready"
    assert payload["detail"]["job_id"] == job_id


def test_search_job_result_endpoint_returns_execute_payload_when_job_is_ready(
    client,
    search_job_create_payload,
):
    create_response = client.post("/v1/search/jobs", json=search_job_create_payload)
    job_id = create_response.json()["job_id"]

    for _ in range(4):
        response = client.get(f"/v1/search/jobs/{job_id}")
        assert response.status_code == 200

    result_response = client.get(f"/v1/search/jobs/{job_id}/result")
    assert result_response.status_code == 200
    payload = result_response.json()
    assert set(payload) == {
        "title",
        "overall_confidence",
        "render_schema_version",
        "metadata",
        "blocks",
    }
    assert payload["render_schema_version"] == "v1"
    assert [block["order"] for block in payload["blocks"]] == list(range(len(payload["blocks"])))
    assert payload["blocks"][0]["block_type"] == "markdown"


def test_search_job_endpoints_return_404_for_unknown_job(client):
    status_response = client.get("/v1/search/jobs/job_missing")
    assert status_response.status_code == 404

    result_response = client.get("/v1/search/jobs/job_missing/result")
    assert result_response.status_code == 404
