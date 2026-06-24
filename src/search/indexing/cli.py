"""Command-line entrypoint for index build & activation.

The AI service owns index administration via this CLI:
``python -m src.search.indexing.cli build|activate|status [--dry-run]``. A thin AI-service admin
endpoint may wrap this later; if Django ever exposes it in its admin, that admin is only a UI calling
the AI-service endpoint — Django never understands OpenSearch internals.

The commands are implemented on top of :mod:`src.search.indexing.builder`.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    """Parse ``build|activate|status [--dry-run]`` and dispatch to the builder.

    Not implemented yet: wire to :mod:`src.search.indexing.builder` and return a process exit code
    (0 on success, non-zero on a failed build/validation/activation).
    """
    raise NotImplementedError("indexing.cli.main is not implemented yet")


if __name__ == "__main__":  # pragma: no cover - exercised once implemented
    raise SystemExit(main())
