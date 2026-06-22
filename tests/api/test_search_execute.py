from src.services.search_service import SearchRetrievalError


def test_search_execute_endpoint_returns_schema_shape(client, search_execute_payload):
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
    assert isinstance(payload["metadata"], dict)
    assert isinstance(payload["blocks"], list)
    assert payload["blocks"]
    assert payload["render_schema_version"] == "v1"
    assert payload["metadata"] == {"mock": True}


def test_search_execute_block_orders_are_unique(client, search_execute_payload):
    response = client.post("/v1/search/execute", json=search_execute_payload)
    assert response.status_code == 200

    payload = response.json()
    orders = [block["order"] for block in payload["blocks"]]
    assert len(orders) == len(set(orders))
    assert orders == list(range(len(orders)))


def test_search_execute_returns_frontend_v1_mock_blocks(client, search_execute_payload):
    response = client.post("/v1/search/execute", json=search_execute_payload)
    assert response.status_code == 200

    payload = response.json()
    blocks = payload["blocks"]
    assert blocks[0]["block_type"] == "text"
    assert {block["block_type"] for block in blocks} <= {
        "text",
        "markdown",
        "surah_distribution",
    }
    for block in blocks:
        assert isinstance(block["payload"], dict)
        assert isinstance(block["provenance"], dict)
        assert block["items"] == []
        assert 0 <= block["confidence"] <= 1

    text_payload = blocks[0]["payload"]
    assert isinstance(text_payload["details"], str)
    assert "<" not in text_payload["details"]
    assert ">" not in text_payload["details"]


def test_search_execute_returns_text_and_markdown_by_default(client, search_execute_payload):
    payload = {
        **search_execute_payload,
        "query": "light",
        "filters": {},
    }

    response = client.post("/v1/search/execute", json=payload)
    assert response.status_code == 200

    blocks = response.json()["blocks"]
    assert [block["order"] for block in blocks] == [0, 1]
    assert [block["block_type"] for block in blocks] == ["text", "markdown"]


def test_search_execute_returns_markdown_block(client, search_execute_payload):
    payload = {
        **search_execute_payload,
        "query": "mercy",
        "filters": {},
    }

    response = client.post("/v1/search/execute", json=payload)
    assert response.status_code == 200

    blocks = response.json()["blocks"]
    assert [block["order"] for block in blocks] == [0, 1]
    assert [block["block_type"] for block in blocks] == ["text", "markdown"]

    markdown_payload = blocks[1]["payload"]
    assert isinstance(markdown_payload, dict)
    assert isinstance(markdown_payload["content"], str)
    assert "## Why this theme matters" in markdown_payload["content"]
    assert "### Mock observations" in markdown_payload["content"]
    assert "| Surah | Mock mentions | Note |" in markdown_payload["content"]
    assert "`r-h-m`" in markdown_payload["content"]


def test_search_execute_returns_full_markdown_content(client, search_execute_payload):
    payload = {
        **search_execute_payload,
        "query": "light",
        "filters": {},
    }

    response = client.post("/v1/search/execute", json=payload)
    assert response.status_code == 200

    markdown_blocks = [
        block for block in response.json()["blocks"] if block["block_type"] == "markdown"
    ]
    assert len(markdown_blocks) == 1

    content = markdown_blocks[0]["payload"]["content"]
    assert "## Why this theme matters" in content
    assert "### Mock observations" in content
    assert "| Surah | Mock mentions | Note |" in content
    assert "```json" in content
    assert "> This block is synthetic" in content
    assert "[Quran Corpus](https://corpus.quran.com)" in content


def test_search_execute_can_return_distribution_variant_without_filters(
    client,
    search_execute_payload,
):
    payload = {
        **search_execute_payload,
        "query": "justice",
        "filters": {},
    }

    response = client.post("/v1/search/execute", json=payload)
    assert response.status_code == 200

    blocks = response.json()["blocks"]
    assert [block["order"] for block in blocks] == [0, 1, 2]
    assert [block["block_type"] for block in blocks] == [
        "text",
        "markdown",
        "surah_distribution",
    ]


def test_search_execute_surahs_filter_controls_distribution_values(client, search_execute_payload):
    payload = {
        **search_execute_payload,
        "filters": {"surahs": [1, 2, 7, "bad", 999, True]},
    }

    response = client.post("/v1/search/execute", json=payload)
    assert response.status_code == 200

    distribution_blocks = [
        block for block in response.json()["blocks"] if block["block_type"] == "surah_distribution"
    ]
    assert len(distribution_blocks) == 1
    distribution = distribution_blocks[0]["payload"]
    assert distribution["values"] == [
        {"surah": 1, "value": 3},
        {"surah": 2, "value": 17},
        {"surah": 7, "value": 5},
    ]
    assert distribution["y_label"] == "Mock mentions"


def test_search_execute_item_ranks_are_unique_within_each_block(client, search_execute_payload):
    response = client.post("/v1/search/execute", json=search_execute_payload)
    assert response.status_code == 200

    payload = response.json()
    for block in payload["blocks"]:
        items = block["items"]
        ranks = [item["rank"] for item in items]
        assert len(ranks) == len(set(ranks))


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

    monkeypatch.setattr(
        "src.api.search.build_search_execute_response",
        raise_retrieval_error,
    )

    response = client.post("/v1/search/execute", json=search_execute_payload)

    assert response.status_code == 503
    payload = response.json()
    assert payload["error"] == "http_error"
    assert payload["detail"] == {
        "message": "OpenSearch lexical index is not available",
        "reason": "index_not_found",
        "backend_status_code": 404,
    }
