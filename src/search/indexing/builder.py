"""Orchestrate an index build and its alias-swap activation.

The builder turns a corpus snapshot into a *new physical index version* and activates it by repointing
the serving alias — the alias swap **is** the activation, so there is no snapshot id/hash to copy.

Build chain:
    pull snapshot (corpus_client) -> build_search_documents -> build_index_settings
      -> create "<prefix>-<date>-<NNN>" -> bulk index -> validate against the golden set
      -> (optionally) swap the alias to the new version

``build`` never auto-activates unless asked; it prints the report and the activate command, so a build
that fails golden-set validation never becomes the served index.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from src.config import Settings, get_settings
from src.search import opensearch_client as osc
from src.search.contracts import QueryContext, SearchFilters
from src.search.indexing.documents import build_document_source, build_search_documents
from src.search.indexing.eval_set import GOLDEN_QUERIES
from src.search.indexing.mapping import build_index_profile, build_index_settings
from src.search.opensearch_client import OpenSearchAdapter, OpenSearchError
from src.search.retrievers.lexical_opensearch import (
    build_search_body,
    compatibility_mismatches,
    ensure_compatible_index,
)
from src.services.corpus_client import build_django_corpus_client


def build_index(
    *,
    settings: Settings | None = None,
    adapter: OpenSearchAdapter | None = None,
    activate: bool = False,
    dry_run: bool = False,
    languages: Sequence[str] | None = None,
    surahs: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build a new physical index version from the current corpus snapshot."""
    settings = settings or get_settings()
    snapshot = build_django_corpus_client(settings).fetch_quran_snapshot(
        translation_languages=languages, surah_numbers=surahs
    )
    documents = build_search_documents(snapshot)
    if not documents:
        raise OpenSearchError("Corpus snapshot produced no documents", reason="empty_corpus")

    adapter = adapter or _adapter(settings)
    physical_index = _next_physical_index_name(adapter, settings.opensearch_index_prefix)
    profile_meta = build_index_profile(
        index_name=physical_index, snapshot=snapshot, documents=documents
    )
    report: dict[str, Any] = {
        "index": physical_index,
        "alias": settings.opensearch_alias,
        "document_count": len(documents),
        "corpus_snapshot_id": snapshot.corpus_snapshot_id,
    }
    if dry_run:
        report["dry_run"] = True
        return report

    osc.create_index(adapter, physical_index, build_index_settings(profile_meta))
    osc.bulk_index(
        adapter, physical_index, ((doc.id, build_document_source(doc)) for doc in documents)
    )
    osc.refresh(adapter, physical_index)

    validation = _validate_golden_set(adapter, physical_index)
    report["validation"] = validation
    report["ok"] = not validation["hard_failures"]
    report["activated"] = False
    if report["ok"] and activate:
        _swap(adapter, settings.opensearch_alias, physical_index)
        report["activated"] = True
    return report


def activate_index(
    physical_index: str,
    *,
    settings: Settings | None = None,
    adapter: OpenSearchAdapter | None = None,
    delete_old: bool = False,
) -> dict[str, Any]:
    """Point the serving alias at an already-built index version (atomic swap)."""
    settings = settings or get_settings()
    adapter = adapter or _adapter(settings)
    alias = settings.opensearch_alias
    ensure_compatible_index(adapter, physical_index)
    previous = [index for index in osc.get_alias_targets(adapter, alias) if index != physical_index]
    osc.swap_alias(adapter, alias, physical_index, remove_indices=previous)
    if delete_old:
        for index in previous:
            osc.delete_index(adapter, index)
    return {
        "alias": alias,
        "active_index": physical_index,
        "previous_indices": previous,
        "deleted_old": delete_old,
        "profile": osc.read_index_profile(adapter, alias),
    }


def index_status(
    *,
    settings: Settings | None = None,
    adapter: OpenSearchAdapter | None = None,
) -> dict[str, Any]:
    """Report the alias target, its build profile, and the code-compatibility result."""
    settings = settings or get_settings()
    adapter = adapter or _adapter(settings)
    alias = settings.opensearch_alias
    targets = osc.get_alias_targets(adapter, alias)
    status: dict[str, Any] = {"alias": alias, "active_indices": targets}
    if not targets:
        status["compatible"] = None
        return status
    profile = osc.read_index_profile(adapter, alias)
    mismatches = compatibility_mismatches(profile)
    status["profile"] = profile
    status["compatible"] = not mismatches
    if mismatches:
        status["mismatches"] = mismatches
    return status


def _adapter(settings: Settings) -> OpenSearchAdapter:
    return osc.build_opensearch_adapter(
        url=settings.opensearch_url,
        timeout_seconds=settings.opensearch_timeout_seconds,
        username=settings.opensearch_username,
        password=settings.opensearch_password,
        verify=settings.opensearch_ca_cert_path or settings.opensearch_verify_certs,
    )


def _next_physical_index_name(adapter: OpenSearchAdapter, prefix: str) -> str:
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    base = f"{prefix}-{date}"
    suffixes = []
    for name in osc.list_index_names(adapter, f"{base}-*"):
        tail = name[len(base) + 1 :]
        if tail.isdigit():
            suffixes.append(int(tail))
    sequence = (max(suffixes) + 1) if suffixes else 1
    return f"{base}-{sequence:03d}"


def _swap(adapter: OpenSearchAdapter, alias: str, physical_index: str) -> None:
    previous = [index for index in osc.get_alias_targets(adapter, alias) if index != physical_index]
    osc.swap_alias(adapter, alias, physical_index, remove_indices=previous)


def _validate_golden_set(adapter: OpenSearchAdapter, index: str) -> dict[str, Any]:
    """Judge a freshly built index against the golden set before it can be activated.

    Structural expectations are enforced for **every** case (hard): each query must return at least one
    hit, and at least one hit must carry the expected ``content_type`` and ``language_code`` — this is
    what catches analyzer/scope/language regressions even when a case has no pinned ids. The
    ``must_include_canonical_ids`` are hard only for ``CONFIRMED`` cases (a missing id fails the build)
    and soft for ``PENDING`` ones (reported, never failing).
    """
    cases: list[dict[str, Any]] = []
    hard_failures: list[str] = []
    soft_misses: list[str] = []
    for case in GOLDEN_QUERIES:
        query_context = QueryContext(
            raw_query=case.query,
            top_k=case.top_k,
            filters=SearchFilters(content_types=list(case.scope)),
        )
        payload = osc.search(adapter, index, build_search_body(query_context))
        hits = _hit_summaries(payload)
        found_ids = [hit["canonical_content_id"] for hit in hits if hit["canonical_content_id"]]
        missing = [cid for cid in case.must_include_canonical_ids if cid not in found_ids]
        problems = _structural_problems(hits, case)
        cases.append(
            {
                "id": case.id,
                "query": case.query,
                "status": case.status.value,
                "hit_count": len(hits),
                "missing": missing,
                "problems": problems,
            }
        )
        if problems:
            hard_failures.append(case.id)
        elif missing:
            (hard_failures if case.is_hard else soft_misses).append(case.id)
    return {"cases": cases, "hard_failures": hard_failures, "soft_misses": soft_misses}


def _structural_problems(hits: list[dict[str, Any]], case: Any) -> list[str]:
    if not hits:
        return ["no_hits"]
    expected_types = {content_type.value for content_type in case.expected_content_types}
    has_expected = any(
        hit["content_type"] in expected_types and hit["language_code"] == case.expected_language
        for hit in hits
    )
    return [] if has_expected else ["missing_expected_type_or_language"]


def _hit_summaries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for hit in payload.get("hits", {}).get("hits", []):
        if not isinstance(hit, dict):
            continue
        source = hit.get("_source") if isinstance(hit.get("_source"), dict) else {}
        metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        canonical = source.get("canonical_content_id")
        summaries.append(
            {
                "canonical_content_id": str(canonical) if canonical else "",
                "content_type": metadata.get("content_type"),
                "language_code": metadata.get("language_code"),
            }
        )
    return summaries
