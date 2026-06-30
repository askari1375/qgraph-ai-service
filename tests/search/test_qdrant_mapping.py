"""Project↔Qdrant mapping: deterministic point ids, payload shape, filter parity."""

from qdrant_client import models

from src.search.contracts import ContentType, SearchFilters
from src.search.indexing.documents import SearchDocumentMetadata, SearchIndexDocument
from src.search.vector.mapping import (
    PAYLOAD_INDEX_FIELDS,
    build_point_id,
    build_point_payload,
    compile_qdrant_filter,
)


def test_point_id_is_deterministic_and_stable():
    # Pinned values guard against an accidental namespace change (which would orphan every point).
    assert build_point_id("ayah:2:255:ar") == "9ce77f82-231d-5def-ada7-6d28f12c6446"
    assert build_point_id("ayah:1:1:translation:en-sahih") == "fcfd65e9-edcb-59ae-8341-640da39cf02e"
    assert build_point_id("ayah:2:255:ar") == build_point_id("ayah:2:255:ar")
    assert build_point_id("ayah:2:255:ar") != build_point_id("ayah:2:256:ar")


def _document() -> SearchIndexDocument:
    return SearchIndexDocument(
        id="ayah:2:255:ar",
        content="اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ",
        canonical_content_id="ayah:2:255",
        metadata=SearchDocumentMetadata(
            content_type=ContentType.QURAN_AYAH,
            surah_number=2,
            ayah_number=255,
            ayah_global_number=262,
            language_code="ar",
            source_id="quran-uthmani",
            source_name="Uthmani",
        ),
    )


def test_build_point_payload_is_flat_and_complete():
    payload = build_point_payload(_document())
    assert payload == {
        "document_id": "ayah:2:255:ar",
        "canonical_content_id": "ayah:2:255",
        "content_type": "quran_ayah",
        "text": "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ",
        "surah_number": 2,
        "ayah_number": 255,
        "ayah_global_number": 262,
        "language_code": "ar",
        "source_id": "quran-uthmani",
        "source_name": "Uthmani",
    }
    # Every payload-index field is a real payload key.
    assert set(PAYLOAD_INDEX_FIELDS) <= set(payload)


def test_compile_filter_none_when_no_conditions():
    # content_types=[] clears the only always-present condition, so nothing compiles.
    assert compile_qdrant_filter(SearchFilters(content_types=[])) is None


def test_compile_filter_mirrors_opensearch_conditions():
    filters = SearchFilters(
        content_types=[ContentType.TRANSLATION],
        languages=["fa", "en"],
        source_ids=["src-a"],
        surah_numbers=[2, 3],
        ayah_global_min=10,
        ayah_global_max=20,
    )
    compiled = compile_qdrant_filter(filters)
    assert isinstance(compiled, models.Filter)
    by_key = {cond.key: cond for cond in compiled.must}
    assert set(by_key) == {
        "content_type",
        "language_code",
        "source_id",
        "surah_number",
        "ayah_global_number",
    }
    assert by_key["content_type"].match.any == ["translation"]
    assert by_key["language_code"].match.any == ["fa", "en"]
    assert by_key["surah_number"].match.any == [2, 3]
    assert by_key["ayah_global_number"].range.gte == 10
    assert by_key["ayah_global_number"].range.lte == 20


def test_compile_filter_open_ended_range():
    compiled = compile_qdrant_filter(SearchFilters(content_types=[], ayah_global_min=5))
    assert compiled is not None
    condition = compiled.must[0]
    assert condition.key == "ayah_global_number"
    assert condition.range.gte == 5
    assert condition.range.lte is None
