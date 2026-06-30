"""OpenAI embedding provider adapter: ordering, validation, and SDK error mapping.

All tests inject a fake client, so nothing here touches the network or makes a paid call.
"""

from collections.abc import Callable, Sequence
from typing import Any

import httpx
import openai
import pytest

from src.search.embeddings.contracts import EmbeddingError, EmbeddingProvider
from src.search.embeddings.openai_provider import OpenAIEmbeddingProvider


class _Item:
    def __init__(self, index: int, embedding: list[float]):
        self.index = index
        self.embedding = embedding


class _Response:
    def __init__(self, data: list[Any]):
        self.data = data


class _FakeEmbeddings:
    def __init__(self, handler: Callable[[str, Sequence[str]], Any]):
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    def create(self, *, model: str, input: Sequence[str]) -> Any:
        self.calls.append({"model": model, "input": list(input)})
        return self._handler(model, input)


class _FakeClient:
    def __init__(self, handler: Callable[[str, Sequence[str]], Any]):
        self.embeddings = _FakeEmbeddings(handler)


def _provider(
    handler: Callable[[str, Sequence[str]], Any],
    *,
    dimensions: int = 2,
    model: str = "text-embedding-3-large",
) -> tuple[OpenAIEmbeddingProvider, _FakeClient]:
    client = _FakeClient(handler)
    provider = OpenAIEmbeddingProvider(model=model, dimensions=dimensions, client=client)
    return provider, client


def _vectors_in_index_order(*vectors: list[float]) -> Callable[[str, Sequence[str]], _Response]:
    def handler(_model: str, _input: Sequence[str]) -> _Response:
        return _Response([_Item(index=i, embedding=vec) for i, vec in enumerate(vectors)])

    return handler


def test_satisfies_protocol_and_profile():
    provider, _ = _provider(_vectors_in_index_order([1.0, 0.0]))
    assert isinstance(provider, EmbeddingProvider)
    profile = provider.profile
    assert profile.provider == "openai"
    assert profile.model == "text-embedding-3-large"
    assert profile.dimensions == 2
    assert profile.distinguishes_input_modes is False


def test_embed_documents_preserves_cardinality():
    provider, client = _provider(_vectors_in_index_order([1.0, 0.0], [0.0, 1.0], [0.5, 0.5]))
    vectors = provider.embed_documents(["a", "b", "c"])
    assert vectors == [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    assert client.embeddings.calls[0]["input"] == ["a", "b", "c"]


def test_embed_documents_reassembles_out_of_order_response():
    def handler(_model: str, _input: Sequence[str]) -> _Response:
        # Provider returns the batch shuffled; only the index makes the mapping correct.
        return _Response(
            [
                _Item(index=2, embedding=[3.0, 3.0]),
                _Item(index=0, embedding=[1.0, 1.0]),
                _Item(index=1, embedding=[2.0, 2.0]),
            ]
        )

    provider, _ = _provider(handler)
    assert provider.embed_documents(["a", "b", "c"]) == [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]


def test_duplicate_response_index_is_invalid():
    # Right count, but index 1 is missing and 0 is duplicated: sorting would silently map a vector to
    # the wrong document, so the permutation check must reject it.
    def handler(_model: str, _input: Sequence[str]) -> _Response:
        return _Response(
            [
                _Item(index=0, embedding=[1.0, 1.0]),
                _Item(index=0, embedding=[2.0, 2.0]),
            ]
        )

    provider, _ = _provider(handler)
    with pytest.raises(EmbeddingError) as excinfo:
        provider.embed_documents(["a", "b"])
    assert excinfo.value.reason == "embedding_response_invalid"


def test_embed_query_returns_single_vector():
    provider, _ = _provider(_vectors_in_index_order([0.6, 0.8]))
    assert provider.embed_query("light") == [0.6, 0.8]


def test_empty_input_skips_the_api_call():
    called = False

    def handler(_model: str, _input: Sequence[str]) -> _Response:
        nonlocal called
        called = True
        return _Response([])

    provider, client = _provider(handler)
    assert provider.embed_documents([]) == []
    assert client.embeddings.calls == []
    assert called is False


def test_wrong_dimension_response_is_invalid():
    provider, _ = _provider(_vectors_in_index_order([1.0, 2.0, 3.0]), dimensions=2)
    with pytest.raises(EmbeddingError) as excinfo:
        provider.embed_documents(["a"])
    assert excinfo.value.reason == "embedding_response_invalid"


def test_count_mismatch_response_is_invalid():
    provider, _ = _provider(_vectors_in_index_order([1.0, 0.0]))
    with pytest.raises(EmbeddingError) as excinfo:
        provider.embed_documents(["a", "b"])
    assert excinfo.value.reason == "embedding_response_invalid"


def test_malformed_item_is_invalid():
    def handler(_model: str, _input: Sequence[str]) -> _Response:
        return _Response([object()])  # no .index / .embedding

    provider, _ = _provider(handler)
    with pytest.raises(EmbeddingError) as excinfo:
        provider.embed_documents(["a"])
    assert excinfo.value.reason == "embedding_response_invalid"


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/embeddings")


def _response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code=status_code, request=_request())


@pytest.mark.parametrize(
    "error",
    [
        openai.APITimeoutError(_request()),
        openai.APIConnectionError(request=_request()),
        openai.RateLimitError("rate limited", response=_response(429), body=None),
        openai.APIStatusError("server error", response=_response(500), body=None),
    ],
)
def test_sdk_errors_map_to_provider_unavailable(error: openai.APIError):
    def handler(_model: str, _input: Sequence[str]) -> Any:
        raise error

    provider, _ = _provider(handler)
    with pytest.raises(EmbeddingError) as excinfo:
        provider.embed_documents(["a"])
    assert excinfo.value.reason == "embedding_provider_unavailable"
    assert excinfo.value.detail["error_type"] == type(error).__name__


def test_status_error_carries_status_code_detail():
    def handler(_model: str, _input: Sequence[str]) -> Any:
        raise openai.APIStatusError("server error", response=_response(503), body=None)

    provider, _ = _provider(handler)
    with pytest.raises(EmbeddingError) as excinfo:
        provider.embed_documents(["a"])
    assert excinfo.value.detail["status_code"] == 503


@pytest.mark.parametrize("bad", [{"model": "", "dimensions": 2}, {"model": "m", "dimensions": 0}])
def test_constructor_rejects_incomplete_config(bad: dict[str, Any]):
    with pytest.raises(EmbeddingError) as excinfo:
        OpenAIEmbeddingProvider(client=_FakeClient(_vectors_in_index_order()), **bad)
    assert excinfo.value.reason == "embedding_provider_not_configured"
