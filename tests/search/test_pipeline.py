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


def _candidate(document_id: str, rank: int) -> RetrievalCandidate:
    return RetrievalCandidate(
        document_id=document_id,
        canonical_content_id=document_id,
        content_type=ContentType.QURAN_AYAH,
        retriever="opensearch_lexical",
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


def test_multiple_retrievers_are_concatenated_for_now():
    a = _FakeRetriever("a", [_candidate("ayah:1:1:ar", 1)])
    b = _FakeRetriever("b", [_candidate("ayah:2:255:ar", 1)])
    result = RetrievalPipeline([a, b]).run(QueryContext(raw_query="x"))
    assert [c.document_id for c in result] == ["ayah:1:1:ar", "ayah:2:255:ar"]
