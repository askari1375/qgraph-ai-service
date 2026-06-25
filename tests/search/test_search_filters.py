from src.search.contracts import (
    DEFAULT_RESULT_CONTENT_TYPES,
    ContentType,
    SearchFilters,
)


def test_defaults_to_general_result_scope_when_empty():
    filters = SearchFilters.from_request_filters({})
    assert filters.content_types == list(DEFAULT_RESULT_CONTENT_TYPES)
    assert filters.languages == []
    assert filters.source_ids == []
    assert filters.surah_numbers == []
    assert filters.ayah_global_min is None


def test_parses_and_coerces_known_keys():
    filters = SearchFilters.from_request_filters(
        {
            "surahs": [2, 2, 999, True, "x", 7],
            "languages": ["EN", "en", "Fa"],
            "source_ids": ["en-Sahih", "en-Sahih"],
            "ayah_global_min": 1,
            "ayah_global_max": 286,
        }
    )
    assert filters.surah_numbers == [2, 7]  # deduped, range-checked, bools/strings dropped
    assert filters.languages == ["en", "fa"]  # casefolded + deduped
    assert filters.source_ids == ["en-Sahih"]  # case preserved + deduped
    assert filters.ayah_global_min == 1
    assert filters.ayah_global_max == 286


def test_surah_ids_is_accepted_as_a_fallback_key():
    filters = SearchFilters.from_request_filters({"surah_ids": [1]})
    assert filters.surah_numbers == [1]


def test_explicit_surah_name_scope():
    filters = SearchFilters.from_request_filters({"content_types": ["surah_name"]})
    assert filters.content_types == [ContentType.SURAH_NAME]


def test_invalid_content_types_fall_back_to_default():
    filters = SearchFilters.from_request_filters({"content_types": ["bogus", 5]})
    assert filters.content_types == list(DEFAULT_RESULT_CONTENT_TYPES)


def test_to_opensearch_filter_default_scope_excludes_surah_name():
    clauses = SearchFilters.from_request_filters({}).to_opensearch_filter()
    content_type_clause = next(
        c for c in clauses if "terms" in c and "metadata.content_type" in c["terms"]
    )
    assert content_type_clause["terms"]["metadata.content_type"] == ["quran_ayah", "translation"]


def test_to_opensearch_filter_compiles_all_clauses():
    filters = SearchFilters.from_request_filters(
        {
            "surahs": [2],
            "languages": ["en"],
            "source_ids": ["en-sahih"],
            "ayah_global_min": 10,
            "ayah_global_max": 20,
        }
    )
    clauses = filters.to_opensearch_filter()
    assert {"terms": {"metadata.language_code": ["en"]}} in clauses
    assert {"terms": {"metadata.source_id": ["en-sahih"]}} in clauses
    assert {"terms": {"metadata.surah_number": [2]}} in clauses
    assert {"range": {"metadata.ayah_global_number": {"gte": 10, "lte": 20}}} in clauses


def test_to_opensearch_filter_omits_absent_range_bounds():
    clauses = SearchFilters.from_request_filters({"ayah_global_min": 5}).to_opensearch_filter()
    assert {"range": {"metadata.ayah_global_number": {"gte": 5}}} in clauses
