import httpx
import pytest

from src.services.corpus_client import DjangoCorpusClient, DjangoCorpusClientError


def _snapshot_payload() -> dict:
    return {
        "schema_version": "qgraph-corpus-snapshot-v1",
        "corpus_snapshot_id": "snapshot-001",
        "corpus_snapshot_hash": "sha256:abc123",
        "produced_at": "2026-06-22T10:00:00Z",
        "filters": {"translation_languages": ["en", "fa"], "surah_numbers": [1]},
        "counts": {"ayahs": 1, "translations": 2},
        "translation_sources": [
            {"source_id": "en-sahih", "language_code": "en"},
            {"source_id": "fa-fooladvand", "language_code": "fa"},
        ],
        "surahs": [{"surah_number": 1, "name_ar": "Al-Fatihah"}],
        "ayahs": [
            {
                "surah_number": 1,
                "ayah_number": 1,
                "ayah_global_number": 1,
                "text_ar": "بسم الله الرحمن الرحيم",
                "translations": [
                    {
                        "language_code": "en",
                        "source_id": "en-sahih",
                        "source_name": "Sahih International",
                        "text": "In the name of Allah, the Entirely Merciful",
                    }
                ],
            }
        ],
    }


def test_django_corpus_client_sends_internal_header_and_query_params():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["query"] = request.url.query.decode()
        seen["token"] = request.headers["X-QGraph-Internal-Token"]
        seen["forwarded_proto"] = request.headers["X-Forwarded-Proto"]
        return httpx.Response(200, json=_snapshot_payload())

    client = DjangoCorpusClient(
        base_url="http://django.test",
        internal_token="secret-token",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    snapshot = client.fetch_quran_snapshot(
        translation_languages=["en", "fa"],
        surah_numbers=[1, 2],
    )

    assert seen == {
        "path": "/api/internal/ai/corpus-snapshots/quran",
        "query": "translation_languages=en%2Cfa&surah_numbers=1%2C2",
        "token": "secret-token",
        "forwarded_proto": "https",
    }
    assert snapshot.corpus_snapshot_id == "snapshot-001"
    assert snapshot.corpus_snapshot_hash == "sha256:abc123"
    assert snapshot.ayahs[0].translations[0].source_id == "en-sahih"


def test_django_corpus_client_rejects_malformed_payload():
    malformed = _snapshot_payload()
    malformed.pop("corpus_snapshot_hash")

    client = DjangoCorpusClient(
        base_url="http://django.test",
        internal_token="secret-token",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=malformed))
        ),
    )

    with pytest.raises(DjangoCorpusClientError) as exc_info:
        client.fetch_quran_snapshot()

    assert exc_info.value.message == (
        "Django corpus snapshot response did not match the expected schema"
    )
    assert exc_info.value.status_code == 200
    assert exc_info.value.errors[0]["loc"] == ("corpus_snapshot_hash",)


def test_django_corpus_client_rejects_unexpected_schema_version():
    payload = _snapshot_payload()
    payload["schema_version"] = "qgraph-corpus-snapshot-v2"

    client = DjangoCorpusClient(
        base_url="http://django.test",
        internal_token="secret-token",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=payload))
        ),
    )

    with pytest.raises(DjangoCorpusClientError) as exc_info:
        client.fetch_quran_snapshot()

    assert exc_info.value.message == ("Django corpus snapshot schema_version is not supported")
    assert exc_info.value.errors[0]["actual"] == "qgraph-corpus-snapshot-v2"


def test_django_corpus_client_maps_http_errors():
    client = DjangoCorpusClient(
        base_url="http://django.test",
        internal_token="secret-token",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda _request: httpx.Response(403, text="forbidden"))
        ),
    )

    with pytest.raises(DjangoCorpusClientError) as exc_info:
        client.fetch_quran_snapshot()

    assert exc_info.value.message == "Django corpus snapshot endpoint returned an error"
    assert exc_info.value.status_code == 403
    assert exc_info.value.errors == [{"body": "forbidden"}]
