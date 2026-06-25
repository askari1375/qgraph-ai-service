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
        "metadata": {"surah_number": 1, "ayah_number": rank, "source_name": "Quran Arabic"},
    }
    values.update(overrides)
    return RetrievalCandidate(**values)


def _build(candidates):
    return build_execute_response(
        SearchExecuteRequest(query="الرحمن"),
        candidates,
        provenance=_PROVENANCE,
        render_schema_version="v1",
        confidence_scale_k=10.0,
    )


def test_renders_a_single_markdown_block():
    response = _build([_candidate(1, 9.0)])
    assert len(response.blocks) == 1
    block = response.blocks[0]
    assert block.block_type == "markdown"
    assert block.items == []
    assert response.render_schema_version == "v1"
    assert response.metadata["backend"] == "open_search"
    assert response.metadata["corpus_snapshot_id"] == "snapshot-001"
    assert block.provenance["backend"] == "open_search"


def test_markdown_content_lists_candidates_without_mark_tags():
    response = _build([_candidate(1, 9.0), _candidate(2, 4.0)])
    content = response.blocks[0].payload["content"]
    assert "| # | Reference | Match | Source |" in content
    assert "Surah 1, Ayah 1" in content
    assert "Surah 1, Ayah 2" in content
    assert "Quran Arabic" in content
    assert "<mark>" not in content and "</mark>" not in content


def test_surah_name_candidate_renders_surah_reference():
    candidate = _candidate(
        1,
        5.0,
        content_type=ContentType.SURAH_NAME,
        canonical_content_id="surah:2",
        text="البقرة",
        highlighted_text="البقرة",
        metadata={"surah_number": 2, "source_name": "Quran Surah Names"},
    )
    content = _build([candidate]).blocks[0].payload["content"]
    assert "Surah 2" in content
    assert "Surah 2, Ayah" not in content


def test_empty_candidates_warns_and_notes_no_matches():
    response = _build([])
    block = response.blocks[0]
    assert block.block_type == "markdown"
    assert block.warning_text == "No lexical matches were returned."
    assert "No lexical matches" in block.payload["content"]
    assert response.overall_confidence == 0.0


def test_confidence_reflects_top_absolute_score():
    weak = _build([_candidate(1, 0.5)]).overall_confidence
    strong = _build([_candidate(1, 40.0)]).overall_confidence
    assert 0.0 < weak < strong < 1.0


def test_markdown_cell_escapes_pipes():
    candidate = _candidate(1, 9.0, text="a | b", highlighted_text="a | b")
    content = _build([candidate]).blocks[0].payload["content"]
    assert "a \\| b" in content
