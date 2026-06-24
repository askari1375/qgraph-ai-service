"""Build searchable documents from a Quran corpus snapshot.

One document = one piece of text, one language, one source (the narrow model). Each document carries a
``content_type`` (the scope/filter axis) and a ``canonical_content_id`` (the logical object it belongs
to — the collapse/dedup/graph-seed key). Build-level provenance (snapshot id/hash, schema/analysis
versions) lives in the index ``_meta`` only, not per document.

This is the home of the document builder; the serving path still uses the older
``src/services/search_documents.py`` until it is retired.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.api.schemas.corpus import CorpusAyah, CorpusAyahTranslation, QuranCorpusSnapshot
from src.search.contracts import (
    ContentType,
    build_ayah_canonical_id,
    build_quran_ayah_document_id,
    build_surah_canonical_id,
    build_surah_name_document_id,
    build_translation_document_id,
)

DOCUMENT_SCHEMA_VERSION = "qgraph_search_document.v2"

ARABIC_SOURCE_ID = "quran-arabic"
ARABIC_SOURCE_NAME = "Quran Arabic"
SURAH_NAME_SOURCE_ID = "quran-surah-names"
SURAH_NAME_SOURCE_NAME = "Quran Surah Names"

# Per-language primary text field; any other language falls back to content_general.
_LANGUAGE_CONTENT_FIELD = {"ar": "content_ar", "fa": "content_fa", "en": "content_en"}
_CONTENT_GENERAL_FIELD = "content_general"

# Which surah-name fields to index, and the language code each is stored under. The Arabic script name
# (arabic_name) supports queries like الفاتحة; the transliteration (e.g. "Al-Baqarah") supports
# Latin-script queries like "Baqara". english_name ("The Cow") is intentionally not indexed yet.
_SURAH_NAME_FIELDS = (("ar", "arabic_name"), ("en", "transliteration"))


class SearchDocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_type: ContentType
    surah_number: int = Field(ge=1, le=114)
    # Ayah-level docs carry these; surah-level docs (surah names) leave them unset.
    ayah_number: int | None = Field(default=None, ge=1)
    ayah_global_number: int | None = Field(default=None, ge=1)
    language_code: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)


class SearchIndexDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    canonical_content_id: str = Field(min_length=1)
    metadata: SearchDocumentMetadata


def build_search_documents(snapshot: QuranCorpusSnapshot) -> list[SearchIndexDocument]:
    documents: list[SearchIndexDocument] = []
    for ayah in snapshot.ayahs:
        documents.append(_build_arabic_document(ayah))
        for translation in ayah.translations:
            documents.append(_build_translation_document(ayah, translation))
    for surah in snapshot.surahs:
        documents.extend(_build_surah_name_documents(surah))
    return documents


def build_document_source(document: SearchIndexDocument) -> dict[str, Any]:
    """Map a document to its OpenSearch ``_source`` (the language-matched content field + metadata)."""
    field = _LANGUAGE_CONTENT_FIELD.get(document.metadata.language_code, _CONTENT_GENERAL_FIELD)
    return {
        "id": document.id,
        "canonical_content_id": document.canonical_content_id,
        field: document.content,
        "metadata": document.metadata.model_dump(mode="json"),
    }


def _build_arabic_document(ayah: CorpusAyah) -> SearchIndexDocument:
    return SearchIndexDocument(
        id=build_quran_ayah_document_id(ayah.surah_number, ayah.ayah_number),
        content=ayah.text_ar,
        canonical_content_id=build_ayah_canonical_id(ayah.surah_number, ayah.ayah_number),
        metadata=SearchDocumentMetadata(
            content_type=ContentType.QURAN_AYAH,
            surah_number=ayah.surah_number,
            ayah_number=ayah.ayah_number,
            ayah_global_number=ayah.ayah_global_number,
            language_code="ar",
            source_id=ARABIC_SOURCE_ID,
            source_name=ARABIC_SOURCE_NAME,
        ),
    )


def _build_translation_document(
    ayah: CorpusAyah, translation: CorpusAyahTranslation
) -> SearchIndexDocument:
    language_code = translation.language_code.casefold()
    return SearchIndexDocument(
        id=build_translation_document_id(
            ayah.surah_number, ayah.ayah_number, translation.source_id
        ),
        content=translation.text,
        canonical_content_id=build_ayah_canonical_id(ayah.surah_number, ayah.ayah_number),
        metadata=SearchDocumentMetadata(
            content_type=ContentType.TRANSLATION,
            surah_number=ayah.surah_number,
            ayah_number=ayah.ayah_number,
            ayah_global_number=ayah.ayah_global_number,
            language_code=language_code,
            source_id=translation.source_id,
            source_name=translation.source_name,
        ),
    )


def _build_surah_name_documents(surah: dict[str, Any]) -> list[SearchIndexDocument]:
    surah_number = _surah_number(surah)
    documents: list[SearchIndexDocument] = []
    for language_code, key in _SURAH_NAME_FIELDS:
        raw_value = surah.get(key)
        text = "" if raw_value is None else str(raw_value).strip()
        if not text:
            continue
        documents.append(
            SearchIndexDocument(
                id=build_surah_name_document_id(surah_number, language_code),
                content=text,
                canonical_content_id=build_surah_canonical_id(surah_number),
                metadata=SearchDocumentMetadata(
                    content_type=ContentType.SURAH_NAME,
                    surah_number=surah_number,
                    language_code=language_code,
                    source_id=SURAH_NAME_SOURCE_ID,
                    source_name=SURAH_NAME_SOURCE_NAME,
                ),
            )
        )
    return documents


def _surah_number(surah: dict[str, Any]) -> int:
    raw = surah.get("number")
    if isinstance(raw, bool) or not isinstance(raw, int) or not 1 <= raw <= 114:
        raise ValueError(f"surah snapshot entry has invalid 'number': {raw!r}")
    return raw
