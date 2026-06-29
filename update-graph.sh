#!/usr/bin/env bash
# Update THIS repo's graphify knowledge graph.
#
#   graphify update .   re-extracts code (AST) + markdown structure and re-merges the
#                       cached LLM semantics. No LLM call, no API cost. This is the same
#                       mechanism the post-commit hook runs automatically on each commit;
#                       run this script for an on-demand refresh between commits.
#
# Deep doc/markdown SEMANTIC extraction (new concepts, rationale, cross-doc links) needs
# an LLM — for that, run `/graphify . --update` inside a Claude Code session. The cached
# semantics from the last LLM pass are preserved by `graphify update` until you do.
#
# If this repo lives inside the QGraph workspace, the merged cross-repo graph is refreshed
# afterwards so root-level queries stay current without leaving this repo.
set -euo pipefail
cd "$(dirname "$0")"

graphify update .

ROOT="$(cd .. && pwd)"
if [ -x "$ROOT/refresh-graph.sh" ] && [ -f "$ROOT/qgraph-backend/graphify-out/graph.json" ]; then
  echo "Refreshing merged workspace graph..."
  "$ROOT/refresh-graph.sh" || true
fi
