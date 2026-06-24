import pytest
from pydantic import ValidationError

from src.search.contracts import (
    CONTENT_TYPES_NOW,
    DEFAULT_RESULT_CONTENT_TYPES,
    RETRIEVER_OPENSEARCH_LEXICAL,
    ContentType,
    QueryContext,
    Retriever,
    RetrievalCandidate,
    SearchFilters,
    build_ayah_canonical_id,
    build_quran_ayah_document_id,
    build_surah_canonical_id,
    build_surah_name_document_id,
    build_translation_document_id,
)


def test_document_id_helpers_match_d12_patterns():
    assert build_quran_ayah_document_id(2, 255) == "ayah:2:255:ar"
    assert build_translation_document_id(2, 255, "en-sahih") == "ayah:2:255:translation:en-sahih"
    assert build_surah_name_document_id(1, "ar") == "surah:1:name:ar"


def test_canonical_id_helpers_match_d12_patterns():
    # Arabic ayah and its translations collapse onto the same canonical id.
    assert build_ayah_canonical_id(2, 255) == "ayah:2:255"
    assert build_surah_canonical_id(1) == "surah:1"


def test_content_type_vocabulary():
    assert ContentType.QURAN_AYAH.value == "quran_ayah"
    assert CONTENT_TYPES_NOW == frozenset(
        {ContentType.QURAN_AYAH, ContentType.TRANSLATION, ContentType.SURAH_NAME}
    )
    # Surah-name docs are excluded from the default result scope.
    assert DEFAULT_RESULT_CONTENT_TYPES == (ContentType.QURAN_AYAH, ContentType.TRANSLATION)
    assert ContentType.SURAH_NAME not in DEFAULT_RESULT_CONTENT_TYPES
    # Future types exist in the vocabulary but are not in the "now" set.
    assert ContentType.TAFSIR_CHUNK not in CONTENT_TYPES_NOW


def test_retrieval_candidate_construction_and_strictness():
    candidate = RetrievalCandidate(
        document_id="ayah:2:255:ar",
        canonical_content_id="ayah:2:255",
        content_type=ContentType.QURAN_AYAH,
        retriever=RETRIEVER_OPENSEARCH_LEXICAL,
        score=12.5,
        rank=1,
    )
    assert candidate.text == ""
    assert candidate.matched_fields == []
    assert candidate.debug == {}
    with pytest.raises(ValidationError):
        RetrievalCandidate(
            document_id="ayah:2:255:ar",
            canonical_content_id="ayah:2:255",
            content_type=ContentType.QURAN_AYAH,
            retriever=RETRIEVER_OPENSEARCH_LEXICAL,
            score=1.0,
            rank=1,
            unexpected="nope",
        )


def test_search_filters_defaults_to_general_result_scope():
    filters = SearchFilters()
    assert filters.content_types == list(DEFAULT_RESULT_CONTENT_TYPES)
    assert filters.languages == []
    assert filters.ayah_global_min is None


def test_query_context_defaults():
    qc = QueryContext(raw_query="الرحمن")
    assert qc.collapse is True
    assert qc.query_embedding is None
    assert qc.top_k == 10
    assert qc.detected_language is None
    assert isinstance(qc.filters, SearchFilters)


def test_query_context_rejects_empty_query():
    with pytest.raises(ValidationError):
        QueryContext(raw_query="")


def test_lexical_retriever_satisfies_retriever_protocol():
    from src.search.retrievers.lexical_opensearch import LexicalRetriever

    retriever = LexicalRetriever()
    assert isinstance(retriever, Retriever)
    assert retriever.name == RETRIEVER_OPENSEARCH_LEXICAL
