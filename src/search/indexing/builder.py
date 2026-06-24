"""Orchestrate an index build and its alias-swap activation.

The builder is the offline workflow that turns a corpus snapshot into a *new physical index version*
and then activates it by repointing an alias — the alias swap **is** the activation, which removes
any hand-copied snapshot-id/hash configuration step.

Build chain:
    pull snapshot (corpus_client) -> build_search_documents -> build_index_settings
      -> create ``{name}-v{n}`` -> bulk index -> validate against the golden eval set
      -> swap the alias to the new version (only if validation passes)

This sits on top of the existing ``OpenSearchLexicalBackend`` bulk machinery.
"""

from __future__ import annotations

from typing import Any


def build_index(*, activate: bool = False, dry_run: bool = False) -> dict[str, Any]:
    """Build a new physical index version from the current corpus snapshot.

    Not implemented yet: create + bulk-load a fresh ``{name}-v{n}`` index, validate it against the
    eval set, and (when ``activate``) swap the alias to it. ``dry_run`` reports the plan without
    writing. Returns a build report (version, document count, validation summary).
    """
    raise NotImplementedError("indexing.builder.build_index is not implemented yet")


def activate_index(version: str) -> dict[str, Any]:
    """Point the serving alias at an already-built index version (atomic swap).

    Not implemented yet: validate the target version exists and is healthy, then move the alias.
    This is the whole of "activation" — no configuration to edit, no app restart.
    """
    raise NotImplementedError("indexing.builder.activate_index is not implemented yet")


def index_status() -> dict[str, Any]:
    """Report which index version the alias currently serves and the available versions.

    Not implemented yet: read the alias target and each version's ``_meta`` profile so provenance
    comes from the index, not from configuration.
    """
    raise NotImplementedError("indexing.builder.index_status is not implemented yet")
