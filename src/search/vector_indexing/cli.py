"""Command-line entrypoint for semantic collection build & activation.

The AI service owns collection administration:
``python -m src.search.vector_indexing.cli build|activate|status [options]`` — the semantic counterpart
of ``src.search.indexing.cli``. ``build`` and ``status`` talk to Qdrant; a real (non-dry-run) build also
needs a configured embedding provider and fails with a clear message until one is wired.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from src.search.embeddings.contracts import EmbeddingError
from src.search.vector.qdrant_store import QdrantError
from src.search.vector_indexing import builder
from src.services.corpus_client import DjangoCorpusClientError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.search.vector_indexing.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build", help="Build a new semantic collection (does not activate)"
    )
    build_parser.add_argument(
        "--activate", action="store_true", help="Activate if validation passes"
    )
    build_parser.add_argument(
        "--dry-run", action="store_true", help="Report the plan without writing or embedding"
    )
    build_parser.add_argument(
        "--languages", help="Comma-separated translation languages (default: all)"
    )
    build_parser.add_argument("--surahs", help="Comma-separated surah numbers (default: all)")

    activate_parser = subparsers.add_parser(
        "activate", help="Point the alias at a built collection"
    )
    activate_parser.add_argument("collection", help="Physical collection name to activate")
    activate_parser.add_argument(
        "--delete-old", action="store_true", help="Delete previous collections and their profiles"
    )

    subparsers.add_parser("status", help="Show the active collection and compatibility")

    args = parser.parse_args(argv)

    try:
        return _dispatch(args)
    except DjangoCorpusClientError as exc:
        _print_corpus_error(exc)
        return 1
    except (EmbeddingError, QdrantError) as exc:
        _print_backend_error(exc)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "build":
        report = builder.build_semantic_collection(
            activate=args.activate,
            dry_run=args.dry_run,
            languages=_csv(args.languages),
            surahs=_csv_int(args.surahs),
        )
        _print(report)
        if report.get("dry_run"):
            return 0
        if not report.get("ok", True):
            print("\nBuild FAILED validation; not activating.")
            return 1
        if not report.get("activated"):
            print(
                "\nTo activate: python -m src.search.vector_indexing.cli activate "
                f"{report['collection']}"
            )
        return 0

    if args.command == "activate":
        _print(builder.activate_semantic_collection(args.collection, delete_old=args.delete_old))
        return 0

    _print(builder.semantic_status())
    return 0


def _print_corpus_error(exc: DjangoCorpusClientError) -> None:
    """Surface the Django corpus-snapshot failure with its HTTP status and body."""
    print(f"ERROR: {exc.message}", file=sys.stderr)
    if exc.status_code is not None:
        print(f"  HTTP status: {exc.status_code}", file=sys.stderr)
    for detail in exc.errors:
        print(f"  {json.dumps(detail, ensure_ascii=False)}", file=sys.stderr)


def _print_backend_error(exc: EmbeddingError | QdrantError) -> None:
    """Surface an embedding/Qdrant failure with its stable reason and any detail."""
    print(f"ERROR: {exc}", file=sys.stderr)
    print(f"  reason: {exc.reason}", file=sys.stderr)
    if exc.detail:
        print(f"  {json.dumps(exc.detail, ensure_ascii=False)}", file=sys.stderr)


def _csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_int(value: str | None) -> list[int] | None:
    parsed = _csv(value)
    if parsed is None:
        return None
    return [int(item) for item in parsed]


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
