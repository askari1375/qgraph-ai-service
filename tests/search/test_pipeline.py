from src.search.contracts import ContentType, QueryContext, RetrievalCandidate
from src.search.pipeline import RetrievalPipeline


class _FakeRetriever:
    def __init__(self, name: str, candidates: list[RetrievalCandidate]):
        self.name = name
        self._candidates = candidates
        self.calls = 0

    def retrieve(self, query_context: QueryContext) -> list[RetrievalCandidate]:
        self.calls += 1
        return self._candidates


def _candidate(
    document_id: str, rank: int, retriever: str = "opensearch_lexical"
) -> RetrievalCandidate:
    return RetrievalCandidate(
        document_id=document_id,
        canonical_content_id=document_id,
        content_type=ContentType.QURAN_AYAH,
        retriever=retriever,
        score=1.0,
        rank=rank,
    )


def test_single_retriever_is_pass_through():
    candidates = [_candidate("ayah:1:1:ar", 1)]
    retriever = _FakeRetriever("opensearch_lexical", candidates)
    pipeline = RetrievalPipeline([retriever])

    result = pipeline.run(QueryContext(raw_query="الرحمن"))

    assert result == candidates
    assert retriever.calls == 1


def test_no_retrievers_returns_empty():
    assert RetrievalPipeline([]).run(QueryContext(raw_query="x")) == []


def test_multiple_retrievers_are_fused_by_rrf():
    # The same document found by both retrievers is merged and outranks a single-backend document.
    shared, solo = "ayah:1:1:ar", "ayah:2:255:ar"
    lexical = _FakeRetriever("opensearch_lexical", [_candidate(shared, 1), _candidate(solo, 2)])
    semantic = _FakeRetriever("qdrant_dense", [_candidate(shared, 1, retriever="qdrant_dense")])
    result = RetrievalPipeline([lexical, semantic]).run(QueryContext(raw_query="x", collapse=False))

    assert [c.document_id for c in result] == [shared, solo]
    assert [c.rank for c in result] == [1, 2]
    # The shared doc carries both backends' raw ranks plus the fused score.
    shared_debug = result[0].debug
    assert set(shared_debug["per_retriever"]) == {"opensearch_lexical", "qdrant_dense"}
    assert shared_debug["fusion"]["fused_score"] > result[1].debug["fusion"]["fused_score"]


def test_fusion_collapses_by_canonical_id_when_requested():
    a = RetrievalCandidate(
        document_id="ayah:1:1:ar",
        canonical_content_id="ayah:1:1",
        content_type=ContentType.QURAN_AYAH,
        retriever="opensearch_lexical",
        score=1.0,
        rank=1,
    )
    b = RetrievalCandidate(
        document_id="ayah:1:1:translation:en",
        canonical_content_id="ayah:1:1",
        content_type=ContentType.TRANSLATION,
        retriever="qdrant_dense",
        score=1.0,
        rank=1,
    )
    collapsed = RetrievalPipeline([_FakeRetriever("l", [a]), _FakeRetriever("s", [b])]).run(
        QueryContext(raw_query="x", collapse=True)
    )
    assert len(collapsed) == 1
