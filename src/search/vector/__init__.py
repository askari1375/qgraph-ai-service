"""Qdrant vector-store boundary and the semantic-index profile.

The semantic counterpart to the lexical OpenSearch modules: a narrow Qdrant adapter, the immutable
per-collection profile + sidecar store + compatibility checks, and the project↔Qdrant mapping
primitives (deterministic point IDs, payloads, filter compilation). Nothing here is wired into the
request path yet.
"""
