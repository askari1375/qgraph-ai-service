"""The named, versioned semantic-corpus policy: which documents get embedded.

The lexical OpenSearch index keeps **every** translation (cheap, supports any source filter). The
semantic Qdrant collection is deliberately a small curated subset — one stable representation per
supported language — so it does not become a 30×-larger duplicate of the lexical index, does not blow
the 4 GB production host, and does not let many renderings of one ayah dominate semantic results.

V1 policy (`ar_arberry_moezzi_v1`): Arabic ayahs + the canonical English (Arberry) + canonical Persian
(Moezzi) translations only — 6,236 × 3 = 18,708 points, surah-names excluded. This is an explicit
product/editorial decision: the source choice is *not* inferred from an automated quality ranking, and
changing either translation source means a newly built and evaluated collection, never an in-place edit.
"""

from __future__ import annotations

from src.search.indexing.documents import SearchIndexDocument
from src.search.vector.qdrant_store import QdrantError

SEMANTIC_CORPUS_POLICY_ID = "qgraph_quran_semantic_corpus"
SEMANTIC_CORPUS_POLICY_VERSION = "ar_arberry_moezzi_v1"

ARABIC_SOURCE_ID = "quran-arabic"
#: The one canonical translation per language embedded into the semantic collection.
CANONICAL_TRANSLATION_SOURCE_IDS: tuple[str, ...] = ("en.arberry", "fa.moezzi")
#: Exactly the source IDs this policy embeds (Arabic + the canonical translations).
SEMANTIC_SOURCE_IDS: tuple[str, ...] = (ARABIC_SOURCE_ID, *CANONICAL_TRANSLATION_SOURCE_IDS)

_CANONICAL_TRANSLATION_SET = frozenset(CANONICAL_TRANSLATION_SOURCE_IDS)
_SEMANTIC_SOURCE_SET = frozenset(SEMANTIC_SOURCE_IDS)


def is_canonical_translation(source_id: str) -> bool:
    """Whether a document's source is one of the policy's canonical translations."""
    return source_id in _CANONICAL_TRANSLATION_SET


def default_scope_descriptor() -> dict[str, object]:
    """The reproducible record of what the collection was built to serve (stamped into the profile)."""
    return {
        "arabic_ayahs": True,
        "canonical_translations": list(CANONICAL_TRANSLATION_SOURCE_IDS),
        "surah_names": False,
        "enforced_by": "collection_curation",
    }


def select_semantic_documents(
    documents: list[SearchIndexDocument],
) -> list[SearchIndexDocument]:
    """Keep only the policy's source documents; fail loudly if the curated set is incomplete.

    The full canonical Django snapshot still defines the corpus identity; this deterministically carves
    out the embedded subset. Surah-names and non-canonical translations fall away because their source
    IDs are not in the policy. Build refuses to proceed (no paid embedding) if a selected source is
    missing, under-covers the ayahs, or has duplicate document IDs — so a silently partial collection
    can never be built.
    """
    selected = [doc for doc in documents if doc.metadata.source_id in _SEMANTIC_SOURCE_SET]

    counts: dict[str, int] = {}
    seen_ids: set[str] = set()
    for doc in selected:
        if doc.id in seen_ids:
            raise QdrantError(
                f"duplicate document id {doc.id} in the semantic corpus",
                reason="semantic_corpus_invalid",
                detail={"document_id": doc.id},
            )
        seen_ids.add(doc.id)
        counts[doc.metadata.source_id] = counts.get(doc.metadata.source_id, 0) + 1

    missing = [source_id for source_id in SEMANTIC_SOURCE_IDS if source_id not in counts]
    if missing:
        raise QdrantError(
            "semantic corpus is missing required sources",
            reason="semantic_corpus_incomplete",
            detail={"missing_sources": missing, "present": sorted(counts)},
        )

    # Each curated translation must cover exactly the Arabic ayah set — no partial source.
    ayah_count = counts[ARABIC_SOURCE_ID]
    uneven = {
        source_id: count
        for source_id, count in counts.items()
        if source_id in _CANONICAL_TRANSLATION_SET and count != ayah_count
    }
    if uneven:
        raise QdrantError(
            "a curated translation does not cover every Arabic ayah",
            reason="semantic_corpus_incomplete",
            detail={"arabic_ayahs": ayah_count, "uneven_sources": uneven},
        )

    return selected
