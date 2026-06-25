from src.api.schemas.search import SearchExecuteRequest
from src.search.contracts import ContentType, RetrievalCandidate
from src.search.response_builder import build_execute_response

_PROVENANCE = {"corpus_snapshot_id": "snapshot-001", "index_id": "qgraph-ayah-lexical-20260625-002"}


def _ayah(rank: int, score: float, **overrides) -> RetrievalCandidate:
    values = {
        "document_id": f"ayah:1:{rank}:ar",
        "canonical_content_id": f"ayah:1:{rank}",
        "content_type": ContentType.QURAN_AYAH,
        "retriever": "opensearch_lexical",
        "score": score,
        "rank": rank,
        "text": "بسم الله الرحمن الرحيم",
        "highlighted_text": "بسم الله <mark>الرحمن</mark> الرحيم",
        "metadata": {
            "surah_number": 1,
            "ayah_number": rank,
            "language_code": "ar",
            "source_name": "Quran Arabic",
        },
    }
    values.update(overrides)
    return RetrievalCandidate(**values)


def _translation(rank: int, score: float, language_code: str, **overrides) -> RetrievalCandidate:
    metadata = {
        "surah_number": 1,
        "ayah_number": rank,
        "language_code": language_code,
        "source_id": f"{language_code}-source",
        "source_name": f"{language_code} source",
    }
    metadata.update(overrides.pop("metadata", {}))
    values = {
        "document_id": f"ayah:1:{rank}:translation:{language_code}-source",
        "canonical_content_id": f"ayah:1:{rank}",
        "content_type": ContentType.TRANSLATION,
        "retriever": "opensearch_lexical",
        "score": score,
        "rank": rank,
        "text": "In the name of Allah",
        "highlighted_text": "In the name of <mark>Allah</mark>",
        "metadata": metadata,
    }
    values.update(overrides)
    return RetrievalCandidate(**values)


def _build(ayah_candidates=None, *, translation_candidates=None, surah_distribution=None):
    return build_execute_response(
        SearchExecuteRequest(query="الرحمن"),
        ayah_candidates=ayah_candidates or [],
        translation_candidates=translation_candidates or [],
        surah_distribution=surah_distribution or [],
        provenance=_PROVENANCE,
        render_schema_version="v1",
        confidence_scale_k=10.0,
    )


def test_renders_typed_arabic_ayah_results_block():
    response = _build([_ayah(1, 9.0)])
    block = response.blocks[0]
    assert block.block_type == "ayah_results"
    assert block.title == "Quran"
    assert block.payload["language_code"] == "ar"
    assert response.render_schema_version == "v1"
    assert response.metadata["backend"] == "open_search"
    assert response.metadata["corpus_snapshot_id"] == "snapshot-001"

    item = block.items[0]
    assert item.rank == 1
    assert item.result_type == "ayah"
    assert item.score == 1.0  # min-max normalized
    assert item.title == "Surah 1, Ayah 1"
    assert item.highlighted_text == "بسم الله <mark>الرحمن</mark> الرحيم"  # mark kept
    assert item.match_metadata["canonical_content_id"] == "ayah:1:1"
    assert item.match_metadata["content_type"] == "quran_ayah"
    assert item.match_metadata["text"] == "بسم الله الرحمن الرحيم"
    assert item.provenance["lexical_score"] == 9.0


def test_item_ranks_are_unique_and_scores_normalized():
    block = _build([_ayah(1, 9.0), _ayah(2, 3.0)]).blocks[0]
    assert [i.rank for i in block.items] == [1, 2]
    assert block.items[0].score == 1.0
    assert block.items[1].score == 3.0 / 9.0


def test_blocks_are_ordered_chart_then_arabic_then_per_language_translations():
    response = _build(
        [_ayah(1, 9.0)],
        translation_candidates=[
            _translation(1, 8.0, "en"),
            _translation(2, 7.0, "fa"),
            _translation(3, 6.0, "en"),
        ],
        surah_distribution=[{"surah": 1, "value": 3}],
    )
    assert [b.order for b in response.blocks] == [0, 1, 2, 3]
    assert [b.block_type for b in response.blocks] == [
        "surah_distribution",
        "ayah_results",
        "ayah_results",
        "ayah_results",
    ]
    assert [b.title for b in response.blocks] == [
        "Where this appears",
        "Quran",
        "English translations",
        "Persian translations",
    ]
    english = response.blocks[2]
    assert english.payload["language_code"] == "en"
    # The two English translations are grouped together and re-ranked within the block.
    assert [i.rank for i in english.items] == [1, 2]
    assert all(i.match_metadata["language_code"] == "en" for i in english.items)
    persian = response.blocks[3]
    assert persian.payload["language_code"] == "fa"
    assert [i.match_metadata["language_code"] for i in persian.items] == ["fa"]


def test_translations_excluded_when_none_retrieved():
    response = _build([_ayah(1, 9.0)], translation_candidates=[])
    assert [b.block_type for b in response.blocks] == ["ayah_results"]
    assert response.blocks[0].title == "Quran"


def test_surah_name_item_uses_surah_reference():
    candidate = _ayah(
        1,
        5.0,
        content_type=ContentType.SURAH_NAME,
        canonical_content_id="surah:2",
        text="البقرة",
        metadata={"surah_number": 2, "source_name": "Quran Surah Names"},
    )
    item = _build([candidate]).blocks[0].items[0]
    assert item.result_type == "surah"
    assert item.title == "Surah 2"


def test_no_distribution_block_when_empty():
    response = _build([_ayah(1, 9.0)], surah_distribution=[])
    assert [b.block_type for b in response.blocks] == ["ayah_results"]


def test_distribution_block_is_first_when_present():
    response = _build([_ayah(1, 9.0)], surah_distribution=[{"surah": 1, "value": 3}])
    chart = response.blocks[0]
    assert chart.block_type == "surah_distribution"
    assert chart.order == 0
    assert chart.payload["values"] == [{"surah": 1, "value": 3}]
    assert chart.payload["max_value"] == 3


def test_empty_candidates_warns_on_arabic_block():
    response = _build([])
    block = response.blocks[0]
    assert block.block_type == "ayah_results"
    assert block.items == []
    assert block.warning_text == "No lexical matches were returned."
    assert response.overall_confidence == 0.0


def test_confidence_reflects_top_absolute_score_across_groups():
    weak = _build([_ayah(1, 0.5)]).overall_confidence
    strong = _build([_ayah(1, 0.5)], translation_candidates=[_translation(1, 40.0, "en")])
    assert 0.0 < weak < strong.overall_confidence < 1.0
