from src.api.schemas.corpus import QuranCorpusSnapshot
from src.search.indexing.documents import build_search_documents
from src.search.indexing.mapping import (
    ANALYSIS_PROFILE_VERSION,
    build_index_profile,
    build_index_settings,
)


def _snapshot() -> QuranCorpusSnapshot:
    return QuranCorpusSnapshot.model_validate(
        {
            "schema_version": "qgraph-corpus-snapshot-v1",
            "corpus_snapshot_id": "snapshot-001",
            "corpus_snapshot_hash": "sha256:abc123",
            "produced_at": "2026-06-22T10:00:00Z",
            "filters": {},
            "counts": {},
            "translation_sources": [],
            "surahs": [{"number": 1, "arabic_name": "الفاتحة", "transliteration": "Al-Fatihah"}],
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
                            "text": "In the name of Allah",
                        }
                    ],
                }
            ],
        }
    )


def _settings() -> dict:
    return build_index_settings({"qgraph": "meta"})


def test_primary_arabic_persian_use_custom_normalize_dont_stem_analyzers():
    properties = _settings()["mappings"]["properties"]
    assert properties["content_ar"]["analyzer"] == "arabic_normalized"
    assert properties["content_fa"]["analyzer"] == "persian_normalized"
    # The built-in stemmers are demoted to the .stemmed sub-fields, never the primary.
    assert properties["content_ar"]["fields"]["stemmed"]["analyzer"] == "arabic"
    assert properties["content_fa"]["fields"]["stemmed"]["analyzer"] == "persian"


def test_custom_analyzers_have_no_stemmer_or_stopword_filter():
    analyzers = _settings()["settings"]["analysis"]["analyzer"]
    for name in ("arabic_normalized", "persian_normalized"):
        filters = analyzers[name]["filter"]
        assert "arabic_normalization" in filters
        assert not any("stem" in f for f in filters)
        assert not any("stop" in f for f in filters)
    assert "persian_normalization" in analyzers["persian_normalized"]["filter"]
    assert analyzers["persian_normalized"]["char_filter"] == ["zwnj_to_space"]


def test_english_is_stemmed_primary_with_exact_subfield():
    content_en = _settings()["mappings"]["properties"]["content_en"]
    assert content_en["analyzer"] == "english"
    assert content_en["fields"]["exact"]["analyzer"] == "english_exact"


def test_no_normalized_text_field_and_strict_dynamic():
    mappings = _settings()["mappings"]
    assert "normalized_text" not in mappings["properties"]
    assert mappings["dynamic"] == "strict"


def test_metadata_carries_content_type_not_build_level_fields():
    metadata_props = _settings()["mappings"]["properties"]["metadata"]["properties"]
    assert metadata_props["content_type"] == {"type": "keyword"}
    assert "corpus_snapshot_id" not in metadata_props
    assert "document_schema_version" not in metadata_props


def test_meta_profile_is_embedded():
    meta = _settings()["mappings"]["_meta"]
    assert meta["qgraph_index_profile"] == {"qgraph": "meta"}


def test_index_profile_carries_analysis_version_and_build_summary():
    snapshot = _snapshot()
    documents = build_search_documents(snapshot)
    profile = build_index_profile(
        index_name="qgraph-ayah-lexical-20260624-001",
        snapshot=snapshot,
        documents=documents,
    )
    assert profile["analysis_profile_version"] == ANALYSIS_PROFILE_VERSION
    assert profile["document_schema_version"] == "qgraph_search_document.v2"
    assert profile["corpus_snapshot_id"] == "snapshot-001"
    assert profile["document_count"] == len(documents)
    assert set(profile["content_types"]) == {"quran_ayah", "translation", "surah_name"}
    assert set(profile["included_languages"]) == {"ar", "en"}
