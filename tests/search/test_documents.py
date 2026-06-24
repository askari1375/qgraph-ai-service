import pytest

from src.api.schemas.corpus import QuranCorpusSnapshot
from src.search.contracts import ContentType
from src.search.indexing.documents import (
    SURAH_NAME_SOURCE_ID,
    build_document_source,
    build_search_documents,
)


def _snapshot(*, surahs=None) -> QuranCorpusSnapshot:
    return QuranCorpusSnapshot.model_validate(
        {
            "schema_version": "qgraph-corpus-snapshot-v1",
            "corpus_snapshot_id": "snapshot-001",
            "corpus_snapshot_hash": "sha256:abc123",
            "produced_at": "2026-06-22T10:00:00Z",
            "filters": {},
            "counts": {},
            "translation_sources": [],
            "surahs": surahs if surahs is not None else [],
            "ayahs": [
                {
                    "surah_number": 2,
                    "ayah_number": 255,
                    "ayah_global_number": 262,
                    "text_ar": "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ",
                    "translations": [
                        {
                            "language_code": "en",
                            "source_id": "en-sahih",
                            "source_name": "Sahih International",
                            "text": "Allah - there is no deity except Him",
                        },
                        {
                            "language_code": "fa",
                            "source_id": "fa-fooladvand",
                            "source_name": "Fooladvand",
                            "text": "خدا کسی است که",
                        },
                    ],
                }
            ],
        }
    )


def test_ayah_and_translation_ids_unchanged():
    documents = build_search_documents(_snapshot())
    assert [doc.id for doc in documents] == [
        "ayah:2:255:ar",
        "ayah:2:255:translation:en-sahih",
        "ayah:2:255:translation:fa-fooladvand",
    ]


def test_canonical_content_id_groups_ayah_and_translations():
    documents = build_search_documents(_snapshot())
    assert {doc.canonical_content_id for doc in documents} == {"ayah:2:255"}


def test_content_type_assigned_per_document():
    documents = build_search_documents(_snapshot())
    assert [doc.metadata.content_type for doc in documents] == [
        ContentType.QURAN_AYAH,
        ContentType.TRANSLATION,
        ContentType.TRANSLATION,
    ]


def test_metadata_is_trimmed_and_has_no_normalized_text():
    document = build_search_documents(_snapshot())[1]
    assert document.metadata.model_dump() == {
        "content_type": "translation",
        "surah_number": 2,
        "ayah_number": 255,
        "ayah_global_number": 262,
        "language_code": "en",
        "source_id": "en-sahih",
        "source_name": "Sahih International",
    }
    # The dropped fields must not reappear.
    assert not hasattr(document, "normalized_text")
    assert "corpus_snapshot_id" not in document.metadata.model_dump()


def test_surah_name_documents_are_emitted_for_arabic_and_transliteration():
    surahs = [
        {
            "number": 2,
            "arabic_name": "البقرة",
            "transliteration": "Al-Baqarah",
            "english_name": "The Cow",
        }
    ]
    surah_docs = [
        doc
        for doc in build_search_documents(_snapshot(surahs=surahs))
        if doc.metadata.content_type == ContentType.SURAH_NAME
    ]
    by_id = {doc.id: doc for doc in surah_docs}
    assert set(by_id) == {"surah:2:name:ar", "surah:2:name:en"}

    ar = by_id["surah:2:name:ar"]
    assert ar.content == "البقرة"
    assert ar.metadata.language_code == "ar"
    assert ar.canonical_content_id == "surah:2"
    assert ar.metadata.ayah_number is None
    assert ar.metadata.source_id == SURAH_NAME_SOURCE_ID

    en = by_id["surah:2:name:en"]
    assert en.content == "Al-Baqarah"  # transliteration -> matches "Baqara"
    assert en.metadata.language_code == "en"


def test_surah_name_documents_skip_blank_fields():
    surahs = [{"number": 1, "arabic_name": "الفاتحة", "transliteration": "  "}]
    surah_docs = [
        doc
        for doc in build_search_documents(_snapshot(surahs=surahs))
        if doc.metadata.content_type == ContentType.SURAH_NAME
    ]
    assert [doc.id for doc in surah_docs] == ["surah:1:name:ar"]


def test_surah_with_invalid_number_raises():
    with pytest.raises(ValueError):
        build_search_documents(_snapshot(surahs=[{"arabic_name": "x"}]))


def test_build_document_source_routes_text_to_language_field():
    documents = build_search_documents(_snapshot())
    ar_source = build_document_source(documents[0])
    en_source = build_document_source(documents[1])
    fa_source = build_document_source(documents[2])

    assert ar_source["content_ar"] == "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ"
    assert "content_fa" not in ar_source and "content_en" not in ar_source
    assert en_source["content_en"] == "Allah - there is no deity except Him"
    assert fa_source["content_fa"] == "خدا کسی است که"
    # Shared keys and trimmed metadata travel with each source.
    assert ar_source["id"] == "ayah:2:255:ar"
    assert ar_source["canonical_content_id"] == "ayah:2:255"
    assert ar_source["metadata"]["content_type"] == "quran_ayah"
    assert "normalized_text" not in ar_source
