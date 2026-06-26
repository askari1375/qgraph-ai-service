def test_search_plan_endpoint_returns_expected_keys(client, search_plan_payload):
    response = client.post("/v1/search/plan", json=search_plan_payload)
    assert response.status_code == 200

    payload = response.json()
    assert set(payload) == {
        "mode",
        "policy_label",
        "policy_snapshot",
        "routing_metadata",
        "backend_name",
        "backend_version",
    }
    assert payload["mode"] in {"sync", "async"}


def test_search_plan_endpoint_returns_json_objects(client, search_plan_payload):
    response = client.post("/v1/search/plan", json=search_plan_payload)
    assert response.status_code == 200

    payload = response.json()
    assert isinstance(payload["policy_snapshot"], dict)
    assert isinstance(payload["routing_metadata"], dict)


def test_search_plan_endpoint_returns_sync_retrieval_policy(client, search_plan_payload):
    response = client.post("/v1/search/plan", json=search_plan_payload)
    assert response.status_code == 200

    payload = response.json()
    assert payload["mode"] == "sync"
    assert payload["policy_label"] == "retrieval_sync_v1"
    assert payload["backend_name"] == "qgraph-ai-service"
    assert payload["policy_snapshot"]["requester"] == {
        "is_authenticated": False,
        "is_guest": True,
    }


def test_search_plan_endpoint_reflects_authenticated_requester(client, search_plan_payload):
    response = client.post(
        "/v1/search/plan",
        json={
            **search_plan_payload,
            "context": {"requester": {"is_authenticated": True, "is_guest": False}},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    # Authenticated users still run synchronous retrieval in the V1 policy.
    assert payload["mode"] == "sync"
    assert payload["policy_snapshot"]["requester"]["is_authenticated"] is True
