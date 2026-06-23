from src.api.schemas.corpus import QuranCorpusSnapshot
from src.services.search_documents import (
    DOCUMENT_SCHEMA_VERSION,
    build_search_documents,
)
from src.services.search_normalization import (
    NORMALIZATION_PROFILE_ID,
    NORMALIZATION_PROFILE_VERSION,
)


def _snapshot() -> QuranCorpusSnapshot:
    return QuranCorpusSnapshot.model_validate(
        {
            "schema_version": "qgraph-corpus-snapshot-v1",
            "corpus_snapshot_id": "snapshot-001",
            "corpus_snapshot_hash": "sha256:abc123",
            "produced_at": "2026-06-22T10:00:00Z",
            "filters": {},
            "counts": {"ayahs": 1, "translations": 2},
            "translation_sources": [],
            "surahs": [],
            "ayahs": [
                {
                    "surah_number": 1,
                    "ayah_number": 1,
                    "ayah_global_number": 1,
                    "text_ar": "إِنَّ ٱللّٰهَ",
                    "translations": [
                        {
                            "language_code": "en",
                            "source_id": "en-sahih",
                            "source_name": "Sahih International",
                            "text": "In the name of Allah",
                        },
                        {
                            "language_code": "fa",
                            "source_id": "fa-fooladvand",
                            "source_name": "Fooladvand",
                            "text": "به نام خداوند",
                        },
                    ],
                }
            ],
        }
    )


def test_document_builder_creates_stable_ids_for_ayah_and_translations():
    documents = build_search_documents(_snapshot())

    assert [document.id for document in documents] == [
        "ayah:1:1:ar",
        "ayah:1:1:translation:en-sahih",
        "ayah:1:1:translation:fa-fooladvand",
    ]


def test_document_builder_includes_corpus_and_normalization_metadata():
    document = build_search_documents(_snapshot())[1]

    assert document.metadata.model_dump() == {
        "corpus_snapshot_id": "snapshot-001",
        "corpus_snapshot_hash": "sha256:abc123",
        "document_schema_version": DOCUMENT_SCHEMA_VERSION,
        "normalization_profile_id": NORMALIZATION_PROFILE_ID,
        "normalization_profile_version": NORMALIZATION_PROFILE_VERSION,
        "surah_number": 1,
        "ayah_number": 1,
        "ayah_global_number": 1,
        "language_code": "en",
        "source_id": "en-sahih",
        "source_name": "Sahih International",
        "document_kind": "translation",
    }
    assert document.normalized_text == "in the name of allah"
