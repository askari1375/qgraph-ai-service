from __future__ import annotations

from typing import Any, Sequence

import httpx
from pydantic import ValidationError

from src.api.schemas.corpus import QuranCorpusSnapshot
from src.config import Settings, get_settings

QURAN_CORPUS_SNAPSHOT_PATH = "/api/internal/ai/corpus-snapshots/quran"
INTERNAL_TOKEN_HEADER = "X-QGraph-Internal-Token"
# Django runs behind a TLS-terminating proxy with SSL redirect on, so a plain-HTTP
# internal call (http://web:8000) is answered with a 301 to https unless the request
# declares it arrived over a secure hop. We are that trusted internal caller, so we
# assert the forwarded-proto Django already trusts (SECURE_PROXY_SSL_HEADER), matching
# how the proxy marks external traffic.
FORWARDED_PROTO_HEADER = "X-Forwarded-Proto"
# Django producer contract (quran.services.corpus_snapshot.SCHEMA_VERSION).
EXPECTED_CORPUS_SCHEMA_VERSION = "qgraph-corpus-snapshot-v1"


class DjangoCorpusClientError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        errors: list[dict[str, Any]] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.errors = errors or []


class DjangoCorpusClient:
    def __init__(
        self,
        *,
        base_url: str,
        internal_token: str,
        timeout_seconds: float = 10.0,
        http_client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.internal_token = internal_token
        self._http_client = http_client or httpx.Client(timeout=timeout_seconds)

    def fetch_quran_snapshot(
        self,
        *,
        translation_languages: Sequence[str] | None = None,
        surah_numbers: Sequence[int] | None = None,
    ) -> QuranCorpusSnapshot:
        if not self.base_url:
            raise DjangoCorpusClientError("Django internal base URL is not configured")
        if not self.internal_token:
            raise DjangoCorpusClientError("Django internal token is not configured")

        params = _build_snapshot_params(
            translation_languages=translation_languages,
            surah_numbers=surah_numbers,
        )
        try:
            response = self._http_client.get(
                f"{self.base_url}{QURAN_CORPUS_SNAPSHOT_PATH}",
                params=params,
                headers={
                    INTERNAL_TOKEN_HEADER: self.internal_token,
                    FORWARDED_PROTO_HEADER: "https",
                },
            )
        except httpx.RequestError as exc:
            raise DjangoCorpusClientError(
                "Failed to pull Django corpus snapshot",
                errors=[{"message": str(exc)}],
            ) from exc

        if response.status_code >= 400:
            raise DjangoCorpusClientError(
                "Django corpus snapshot endpoint returned an error",
                status_code=response.status_code,
                errors=[{"body": response.text}],
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise DjangoCorpusClientError(
                "Django corpus snapshot response was not valid JSON",
                status_code=response.status_code,
                errors=[{"message": str(exc)}],
            ) from exc

        try:
            snapshot = QuranCorpusSnapshot.model_validate(payload)
        except ValidationError as exc:
            raise DjangoCorpusClientError(
                "Django corpus snapshot response did not match the expected schema",
                status_code=response.status_code,
                errors=exc.errors(
                    include_context=False,
                    include_input=False,
                    include_url=False,
                ),
            ) from exc

        if snapshot.schema_version != EXPECTED_CORPUS_SCHEMA_VERSION:
            raise DjangoCorpusClientError(
                "Django corpus snapshot schema_version is not supported",
                status_code=response.status_code,
                errors=[
                    {
                        "expected": EXPECTED_CORPUS_SCHEMA_VERSION,
                        "actual": snapshot.schema_version,
                    }
                ],
            )
        return snapshot


def _build_snapshot_params(
    *,
    translation_languages: Sequence[str] | None,
    surah_numbers: Sequence[int] | None,
) -> dict[str, str]:
    params: dict[str, str] = {}

    cleaned_languages = [
        language.strip()
        for language in translation_languages or []
        if isinstance(language, str) and language.strip()
    ]
    if cleaned_languages:
        params["translation_languages"] = ",".join(cleaned_languages)

    cleaned_surahs = [
        str(surah_number)
        for surah_number in surah_numbers or []
        if isinstance(surah_number, int)
        and not isinstance(surah_number, bool)
        and 1 <= surah_number <= 114
    ]
    if cleaned_surahs:
        params["surah_numbers"] = ",".join(cleaned_surahs)

    return params


def build_django_corpus_client(settings: Settings | None = None) -> DjangoCorpusClient:
    cfg = settings if settings is not None else get_settings()
    return DjangoCorpusClient(
        base_url=cfg.django_internal_base_url,
        internal_token=cfg.django_internal_token,
        timeout_seconds=cfg.django_internal_timeout_seconds,
    )
