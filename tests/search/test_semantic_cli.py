"""Semantic CLI wiring: dispatch, flag forwarding, exit codes, and error surfacing."""

from typing import Any

from src.search.embeddings.contracts import EmbeddingError
from src.search.vector.qdrant_store import QdrantError
from src.search.vector_indexing import cli
from src.services.corpus_client import DjangoCorpusClientError


def test_build_surfaces_corpus_error_status_and_body(monkeypatch, capsys):
    def fail(**_kw):
        raise DjangoCorpusClientError(
            "Django corpus snapshot endpoint returned an error",
            status_code=400,
            errors=[{"body": "Invalid HTTP_HOST header: 'web:8000'."}],
        )

    monkeypatch.setattr(cli.builder, "build_semantic_collection", fail)
    assert cli.main(["build"]) == 1

    err = capsys.readouterr().err
    assert "HTTP status: 400" in err
    assert "web:8000" in err


def test_build_surfaces_provider_not_configured(monkeypatch, capsys):
    def fail(**_kw):
        raise EmbeddingError(
            "no production embedding provider is configured",
            reason="embedding_provider_not_configured",
        )

    monkeypatch.setattr(cli.builder, "build_semantic_collection", fail)
    assert cli.main(["build"]) == 1
    assert "embedding_provider_not_configured" in capsys.readouterr().err


def test_build_command_prints_activate_hint_and_returns_zero(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.builder,
        "build_semantic_collection",
        lambda **_kw: {"collection": "col-1", "ok": True, "activated": False},
    )
    assert cli.main(["build"]) == 0
    assert "To activate" in capsys.readouterr().out


def test_build_failure_returns_one(monkeypatch):
    monkeypatch.setattr(
        cli.builder,
        "build_semantic_collection",
        lambda **_kw: {"collection": "col-1", "ok": False, "activated": False, "validation": {}},
    )
    assert cli.main(["build"]) == 1


def test_build_forwards_flags_and_filters(monkeypatch):
    captured: dict[str, Any] = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return {"collection": "col-1", "ok": True, "activated": True}

    monkeypatch.setattr(cli.builder, "build_semantic_collection", fake)
    cli.main(["build", "--activate", "--languages", "en, fa", "--surahs", "1,2"])

    assert captured["activate"] is True
    assert captured["languages"] == ["en", "fa"]
    assert captured["surahs"] == [1, 2]


def test_dry_run_returns_zero(monkeypatch):
    monkeypatch.setattr(
        cli.builder,
        "build_semantic_collection",
        lambda **_kw: {"collection": "col-1", "dry_run": True},
    )
    assert cli.main(["build", "--dry-run"]) == 0


def test_activate_command_forwards_args(monkeypatch):
    captured: dict[str, Any] = {}

    def fake(collection, **kwargs):
        captured["collection"] = collection
        captured.update(kwargs)
        return {"active_collection": collection}

    monkeypatch.setattr(cli.builder, "activate_semantic_collection", fake)
    assert cli.main(["activate", "my-collection", "--delete-old"]) == 0
    assert captured["collection"] == "my-collection"
    assert captured["delete_old"] is True


def test_activate_surfaces_qdrant_error(monkeypatch, capsys):
    def fail(collection, **_kw):
        raise QdrantError("alias is missing", reason="semantic_alias_invalid")

    monkeypatch.setattr(cli.builder, "activate_semantic_collection", fail)
    assert cli.main(["activate", "col-1"]) == 1
    assert "semantic_alias_invalid" in capsys.readouterr().err


def test_status_command(monkeypatch):
    monkeypatch.setattr(
        cli.builder, "semantic_status", lambda: {"alias": "a", "active_collection": None}
    )
    assert cli.main(["status"]) == 0
