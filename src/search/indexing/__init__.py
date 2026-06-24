"""Offline index build & activation pipeline.

A real, tested entrypoint for building and activating the OpenSearch index. The chain is:

    corpus snapshot (from Django)
      -> documents.build_search_documents      # + surah-name docs, + canonical_content_id
      -> mapping.build_index_settings           # custom normalize-don't-stem analyzers
      -> builder: create versioned index -> bulk -> validate vs eval set -> swap alias

Activation is an **alias swap**, which replaces hand-copied snapshot-id/hash configuration. ``cli``
exposes ``build|activate|status|--dry-run``.

The modules here are currently skeletons; their bodies are filled in as the indexing work lands.
"""
