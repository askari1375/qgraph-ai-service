from src.api.schemas.search import SearchExecuteRequest
from src.search.contracts import ContentType, RetrievalCandidate
from src.search.response_builder import build_execute_response

_PROVENANCE = {"corpus_snapshot_id": "snapshot-001", "index_id": "qgraph-ayah-lexical-20260625-002"}


def _candidate(rank: int, score: float, **overrides) -> RetrievalCandidate:
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


def _build(candidates, *, surah_distribution=None):
    return build_execute_response(
        SearchExecuteRequest(query="الرحمن"),
        candidates,
        surah_distribution=surah_distribution or [],
        provenance=_PROVENANCE,
        render_schema_version="v1",
        confidence_scale_k=10.0,
    )


def test_renders_typed_ayah_results_block():
    response = _build([_candidate(1, 9.0)])
    block = response.blocks[0]
    assert block.block_type == "ayah_results"
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
    block = _build([_candidate(1, 9.0), _candidate(2, 3.0)]).blocks[0]
    assert [i.rank for i in block.items] == [1, 2]
    assert block.items[0].score == 1.0
    assert block.items[1].score == 3.0 / 9.0


def test_surah_name_item_uses_surah_reference():
    candidate = _candidate(
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


def test_distribution_block_appended_when_present():
    response = _build([_candidate(1, 9.0)], surah_distribution=[{"surah": 1, "value": 3}])
    assert [b.block_type for b in response.blocks] == ["ayah_results", "surah_distribution"]
    chart = response.blocks[1]
    assert chart.order == 1
    assert chart.payload["values"] == [{"surah": 1, "value": 3}]
    assert chart.payload["max_value"] == 3


def test_no_distribution_block_when_empty():
    response = _build([_candidate(1, 9.0)], surah_distribution=[])
    assert [b.block_type for b in response.blocks] == ["ayah_results"]


def test_empty_candidates_warns():
    response = _build([])
    block = response.blocks[0]
    assert block.block_type == "ayah_results"
    assert block.items == []
    assert block.warning_text == "No lexical matches were returned."
    assert response.overall_confidence == 0.0


def test_confidence_reflects_top_absolute_score():
    weak = _build([_candidate(1, 0.5)]).overall_confidence
    strong = _build([_candidate(1, 40.0)]).overall_confidence
    assert 0.0 < weak < strong < 1.0
