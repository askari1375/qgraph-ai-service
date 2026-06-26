from src.services.search_jobs import ASYNC_SEARCH_NOT_IMPLEMENTED_REASON


def test_search_job_create_fails_loud_with_501(client, search_job_create_payload):
    response = client.post("/v1/search/jobs", json=search_job_create_payload)
    assert response.status_code == 501

    payload = response.json()
    assert payload["error"] == "http_error"
    assert payload["detail"]["reason"] == ASYNC_SEARCH_NOT_IMPLEMENTED_REASON


def test_search_job_status_fails_loud_with_501(client):
    response = client.get("/v1/search/jobs/job_anything")
    assert response.status_code == 501
    assert response.json()["detail"]["reason"] == ASYNC_SEARCH_NOT_IMPLEMENTED_REASON


def test_search_job_result_fails_loud_with_501(client):
    response = client.get("/v1/search/jobs/job_anything/result")
    assert response.status_code == 501
    assert response.json()["detail"]["reason"] == ASYNC_SEARCH_NOT_IMPLEMENTED_REASON
