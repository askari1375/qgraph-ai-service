from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.api.schemas.corpus import CorpusAyah, CorpusAyahTranslation, QuranCorpusSnapshot
from src.services.search_normalization import (
    NORMALIZATION_PROFILE_ID,
    NORMALIZATION_PROFILE_VERSION,
    normalize_text,
)

DOCUMENT_SCHEMA_VERSION = "qgraph_search_document.v1"
ARABIC_SOURCE_ID = "quran-arabic"
ARABIC_SOURCE_NAME = "Quran Arabic"

SearchDocumentKind = Literal["ayah_arabic", "translation"]


class SearchDocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus_snapshot_id: str = Field(min_length=1)
    corpus_snapshot_hash: str = Field(min_length=1)
    document_schema_version: str = Field(min_length=1)
    normalization_profile_id: str = Field(min_length=1)
    normalization_profile_version: str = Field(min_length=1)
    surah_number: int = Field(ge=1, le=114)
    ayah_number: int = Field(ge=1)
    ayah_global_number: int = Field(ge=1)
    language_code: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    document_kind: SearchDocumentKind


class SearchIndexDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    normalized_text: str = Field(min_length=1)
    metadata: SearchDocumentMetadata


def build_search_documents(snapshot: QuranCorpusSnapshot) -> list[SearchIndexDocument]:
    documents: list[SearchIndexDocument] = []
    for ayah in snapshot.ayahs:
        documents.append(_build_arabic_document(snapshot, ayah))
        for translation in ayah.translations:
            documents.append(_build_translation_document(snapshot, ayah, translation))
    return documents


def build_arabic_document_id(ayah: CorpusAyah) -> str:
    return f"ayah:{ayah.surah_number}:{ayah.ayah_number}:ar"


def build_translation_document_id(ayah: CorpusAyah, translation: CorpusAyahTranslation) -> str:
    return f"ayah:{ayah.surah_number}:{ayah.ayah_number}:translation:{translation.source_id}"


def _build_arabic_document(
    snapshot: QuranCorpusSnapshot,
    ayah: CorpusAyah,
) -> SearchIndexDocument:
    metadata = _build_metadata(
        snapshot=snapshot,
        ayah=ayah,
        language_code="ar",
        source_id=ARABIC_SOURCE_ID,
        source_name=ARABIC_SOURCE_NAME,
        document_kind="ayah_arabic",
    )
    return SearchIndexDocument(
        id=build_arabic_document_id(ayah),
        text=ayah.text_ar,
        normalized_text=normalize_text(ayah.text_ar, "ar"),
        metadata=metadata,
    )


def _build_translation_document(
    snapshot: QuranCorpusSnapshot,
    ayah: CorpusAyah,
    translation: CorpusAyahTranslation,
) -> SearchIndexDocument:
    language_code = translation.language_code.casefold()
    metadata = _build_metadata(
        snapshot=snapshot,
        ayah=ayah,
        language_code=language_code,
        source_id=translation.source_id,
        source_name=translation.source_name,
        document_kind="translation",
    )
    return SearchIndexDocument(
        id=build_translation_document_id(ayah, translation),
        text=translation.text,
        normalized_text=normalize_text(translation.text, language_code),
        metadata=metadata,
    )


def _build_metadata(
    *,
    snapshot: QuranCorpusSnapshot,
    ayah: CorpusAyah,
    language_code: str,
    source_id: str,
    source_name: str,
    document_kind: SearchDocumentKind,
) -> SearchDocumentMetadata:
    return SearchDocumentMetadata(
        corpus_snapshot_id=snapshot.corpus_snapshot_id,
        corpus_snapshot_hash=snapshot.corpus_snapshot_hash,
        document_schema_version=DOCUMENT_SCHEMA_VERSION,
        normalization_profile_id=NORMALIZATION_PROFILE_ID,
        normalization_profile_version=NORMALIZATION_PROFILE_VERSION,
        surah_number=ayah.surah_number,
        ayah_number=ayah.ayah_number,
        ayah_global_number=ayah.ayah_global_number,
        language_code=language_code,
        source_id=source_id,
        source_name=source_name,
        document_kind=document_kind,
    )
