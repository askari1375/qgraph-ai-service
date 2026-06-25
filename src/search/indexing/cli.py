"""Command-line entrypoint for index build & activation.

The AI service owns index administration:
``python -m src.search.indexing.cli build|activate|status [options]``. A thin AI-service admin
endpoint may wrap this later; if Django ever exposes it, that admin is only a UI calling the
AI-service endpoint — Django never understands OpenSearch internals.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from src.search.indexing import builder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.search.indexing.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build", help="Build a new index version (does not activate)"
    )
    build_parser.add_argument(
        "--activate", action="store_true", help="Activate if validation passes"
    )
    build_parser.add_argument(
        "--dry-run", action="store_true", help="Report the plan without writing"
    )
    build_parser.add_argument(
        "--languages", help="Comma-separated translation languages (default: all)"
    )
    build_parser.add_argument("--surahs", help="Comma-separated surah numbers (default: all)")

    activate_parser = subparsers.add_parser("activate", help="Point the alias at a built index")
    activate_parser.add_argument("index", help="Physical index name to activate")
    activate_parser.add_argument(
        "--delete-old", action="store_true", help="Delete previous indices"
    )

    subparsers.add_parser("status", help="Show the active index and compatibility")

    args = parser.parse_args(argv)

    if args.command == "build":
        report = builder.build_index(
            activate=args.activate,
            dry_run=args.dry_run,
            languages=_csv(args.languages),
            surahs=_csv_int(args.surahs),
        )
        _print(report)
        if report.get("dry_run"):
            return 0
        if not report.get("ok", True):
            print("\nBuild FAILED golden-set validation; not activating.")
            return 1
        if not report.get("activated"):
            print(f"\nTo activate: python -m src.search.indexing.cli activate {report['index']}")
        return 0

    if args.command == "activate":
        _print(builder.activate_index(args.index, delete_old=args.delete_old))
        return 0

    _print(builder.index_status())
    return 0


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
