from typing import Any

from src.search.indexing import cli


def test_build_command_prints_activate_hint_and_returns_zero(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.builder, "build_index", lambda **_kw: {"index": "idx-1", "ok": True, "activated": False}
    )
    assert cli.main(["build"]) == 0
    assert "To activate" in capsys.readouterr().out


def test_build_failure_returns_one(monkeypatch):
    monkeypatch.setattr(
        cli.builder,
        "build_index",
        lambda **_kw: {"index": "idx-1", "ok": False, "activated": False, "validation": {}},
    )
    assert cli.main(["build"]) == 1


def test_build_forwards_flags_and_filters(monkeypatch):
    captured: dict[str, Any] = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return {"index": "idx-1", "ok": True, "activated": True}

    monkeypatch.setattr(cli.builder, "build_index", fake)
    cli.main(["build", "--activate", "--languages", "en, fa", "--surahs", "1,2"])

    assert captured["activate"] is True
    assert captured["languages"] == ["en", "fa"]
    assert captured["surahs"] == [1, 2]


def test_dry_run_returns_zero(monkeypatch):
    monkeypatch.setattr(
        cli.builder, "build_index", lambda **_kw: {"index": "idx-1", "dry_run": True}
    )
    assert cli.main(["build", "--dry-run"]) == 0


def test_activate_command_forwards_args(monkeypatch):
    captured: dict[str, Any] = {}

    def fake(index, **kwargs):
        captured["index"] = index
        captured.update(kwargs)
        return {"active_index": index}

    monkeypatch.setattr(cli.builder, "activate_index", fake)
    assert cli.main(["activate", "my-index", "--delete-old"]) == 0
    assert captured["index"] == "my-index"
    assert captured["delete_old"] is True


def test_status_command(monkeypatch):
    monkeypatch.setattr(cli.builder, "index_status", lambda: {"alias": "a", "active_indices": []})
    assert cli.main(["status"]) == 0
