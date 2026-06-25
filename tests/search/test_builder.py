import json
from typing import Any

from src.api.schemas.corpus import QuranCorpusSnapshot
from src.config import Settings
from src.search.indexing import builder


class _Resp:
    def __init__(self, status_code: int = 200, payload: Any = None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = ""

    def json(self) -> Any:
        return self._payload


class _Cluster:
    """A tiny in-memory OpenSearch stand-in supporting the builder's operations."""

    def __init__(self):
        self.indices: dict[str, dict[str, Any]] = {}
        self.aliases: dict[str, list[str]] = {}

    def put(self, path: str, *, json_payload) -> _Resp:
        name = path.lstrip("/")
        meta = json_payload["mappings"]["_meta"]["qgraph_index_profile"]
        self.indices[name] = {"meta": meta, "docs": {}}
        return _Resp(200, {"acknowledged": True})

    def post(self, path: str, *, json_payload=None, content=None, headers=None) -> _Resp:
        if path == "/_bulk":
            self._ingest_bulk(content)
            return _Resp(200, {"errors": False})
        if path == "/_aliases":
            self._apply_aliases(json_payload["actions"])
            return _Resp(200, {"acknowledged": True})
        if path.endswith("/_refresh"):
            return _Resp(200, {})
        if path.endswith("/_search"):
            return _Resp(200, self._search(path[1 : -len("/_search")]))
        return _Resp(404)

    def get(self, path: str) -> _Resp:
        if path.startswith("/_cat/indices/"):
            prefix = path[len("/_cat/indices/") :].split("?")[0].rstrip("*")
            return _Resp(200, [{"index": n} for n in self.indices if n.startswith(prefix)])
        if path.startswith("/_alias/"):
            targets = self.aliases.get(path[len("/_alias/") :], [])
            if not targets:
                return _Resp(404)
            return _Resp(200, {idx: {"aliases": {}} for idx in targets})
        name = path.lstrip("/")
        if name in self.aliases:
            return _Resp(200, {idx: self._mappings(idx) for idx in self.aliases[name]})
        if name in self.indices:
            return _Resp(200, {name: self._mappings(name)})
        return _Resp(404)

    def delete(self, path: str) -> _Resp:
        self.indices.pop(path.lstrip("/"), None)
        return _Resp(200, {})

    def _mappings(self, index: str) -> dict[str, Any]:
        return {"mappings": {"_meta": {"qgraph_index_profile": self.indices[index]["meta"]}}}

    def _ingest_bulk(self, content: str) -> None:
        lines = [line for line in content.split("\n") if line]
        for i in range(0, len(lines), 2):
            action = json.loads(lines[i])["index"]
            self.indices[action["_index"]]["docs"][action["_id"]] = json.loads(lines[i + 1])

    def _apply_aliases(self, actions: list[dict[str, Any]]) -> None:
        for action in actions:
            if "remove" in action:
                alias, index = action["remove"]["alias"], action["remove"]["index"]
                if index in self.aliases.get(alias, []):
                    self.aliases[alias].remove(index)
            if "add" in action:
                alias, index = action["add"]["alias"], action["add"]["index"]
                self.aliases.setdefault(alias, []).append(index)

    def _search(self, index: str) -> dict[str, Any]:
        docs = self.indices.get(index, {}).get("docs", {})
        return {
            "hits": {"hits": [{"_id": d, "_score": 1.0, "_source": s} for d, s in docs.items()]}
        }


def _settings() -> Settings:
    return Settings(
        opensearch_url="http://opensearch:9200",
        opensearch_alias="qgraph-ayah-lexical-active",
        opensearch_index_prefix="qgraph-ayah-lexical",
    )


def _ayah(surah: int, ayah: int, global_number: int, text_ar: str) -> dict:
    return {
        "surah_number": surah,
        "ayah_number": ayah,
        "ayah_global_number": global_number,
        "text_ar": text_ar,
        "translations": [
            {
                "language_code": "en",
                "source_id": "en-sahih",
                "source_name": "Sahih International",
                "text": "In the name of Allah, the Entirely Merciful",
            }
        ],
    }


def _make_snapshot(*, surahs: list[dict], ayahs: list[dict]) -> QuranCorpusSnapshot:
    return QuranCorpusSnapshot.model_validate(
        {
            "schema_version": "qgraph-corpus-snapshot-v1",
            "corpus_snapshot_id": "snapshot-001",
            "corpus_snapshot_hash": "sha256:abc123",
            "produced_at": "2026-06-25T10:00:00Z",
            "filters": {},
            "counts": {},
            "translation_sources": [],
            "surahs": surahs,
            "ayahs": ayahs,
        }
    )


def _passing_snapshot() -> QuranCorpusSnapshot:
    # Carries every confirmed golden canonical: ayah:1:1, ayah:1:3, surah:1, surah:2.
    return _make_snapshot(
        surahs=[
            {"number": 1, "arabic_name": "الفاتحة", "transliteration": "Al-Fatihah"},
            {"number": 2, "arabic_name": "البقرة", "transliteration": "Al-Baqarah"},
        ],
        ayahs=[
            _ayah(1, 1, 1, "بسم الله الرحمن الرحيم"),
            _ayah(1, 3, 3, "الرحمن الرحيم"),
        ],
    )


def _failing_snapshot() -> QuranCorpusSnapshot:
    # Lacks the confirmed canonicals, so golden-set validation must fail.
    return _make_snapshot(
        surahs=[{"number": 3, "arabic_name": "آل عمران", "transliteration": "Aal-e-Imran"}],
        ayahs=[_ayah(3, 1, 1, "الم")],
    )


def _patch_corpus(monkeypatch, snapshot: QuranCorpusSnapshot) -> None:
    class _FakeCorpusClient:
        def fetch_quran_snapshot(self, **_kwargs) -> QuranCorpusSnapshot:
            return snapshot

    monkeypatch.setattr(builder, "build_django_corpus_client", lambda settings: _FakeCorpusClient())


def test_build_creates_index_and_passes_validation(monkeypatch):
    _patch_corpus(monkeypatch, _passing_snapshot())
    cluster = _Cluster()

    report = builder.build_index(settings=_settings(), adapter=cluster)

    assert report["ok"] is True
    assert report["activated"] is False
    assert report["index"].startswith("qgraph-ayah-lexical-")
    assert report["document_count"] == 8  # 2 ayat x (arabic+translation) + 2 surahs x 2 names
    assert report["validation"]["hard_failures"] == []
    assert report["index"] in cluster.indices
    # Build alone never activates.
    assert cluster.aliases.get("qgraph-ayah-lexical-active", []) == []


def test_build_with_activate_swaps_the_alias(monkeypatch):
    _patch_corpus(monkeypatch, _passing_snapshot())
    cluster = _Cluster()

    report = builder.build_index(settings=_settings(), adapter=cluster, activate=True)

    assert report["ok"] is True
    assert report["activated"] is True
    assert cluster.aliases["qgraph-ayah-lexical-active"] == [report["index"]]


def test_build_fails_validation_when_confirmed_hits_missing(monkeypatch):
    # Missing the confirmed canonicals (ayah:1:1, ayah:1:3, surah:1, surah:2).
    _patch_corpus(monkeypatch, _failing_snapshot())
    cluster = _Cluster()

    report = builder.build_index(settings=_settings(), adapter=cluster, activate=True)

    assert report["ok"] is False
    assert report["activated"] is False
    assert "ar-basmala-phrase" in report["validation"]["hard_failures"]


def test_dry_run_writes_nothing(monkeypatch):
    _patch_corpus(monkeypatch, _passing_snapshot())
    cluster = _Cluster()

    report = builder.build_index(settings=_settings(), adapter=cluster, dry_run=True)

    assert report["dry_run"] is True
    assert cluster.indices == {}


def test_activate_then_status_reports_active_index(monkeypatch):
    _patch_corpus(monkeypatch, _passing_snapshot())
    settings = _settings()
    cluster = _Cluster()
    report = builder.build_index(settings=settings, adapter=cluster)

    activation = builder.activate_index(report["index"], settings=settings, adapter=cluster)
    assert activation["active_index"] == report["index"]

    status = builder.index_status(settings=settings, adapter=cluster)
    assert status["active_indices"] == [report["index"]]
    assert status["compatible"] is True


def test_activate_with_delete_old_removes_previous(monkeypatch):
    _patch_corpus(monkeypatch, _passing_snapshot())
    settings = _settings()
    cluster = _Cluster()

    first = builder.build_index(settings=settings, adapter=cluster)
    builder.activate_index(first["index"], settings=settings, adapter=cluster)
    # A second build needs a distinct name; the in-memory cluster keys on the generated name.
    cluster.indices.setdefault(first["index"] + "-x", cluster.indices[first["index"]])

    second_index = first["index"] + "-x"
    builder.activate_index(second_index, settings=settings, adapter=cluster, delete_old=True)

    assert cluster.aliases["qgraph-ayah-lexical-active"] == [second_index]
    assert first["index"] not in cluster.indices
