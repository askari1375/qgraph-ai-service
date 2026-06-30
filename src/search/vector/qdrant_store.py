"""Narrow Qdrant vector-store adapter.

qdrant-client is itself the transport, so — unlike the httpx-based OpenSearch client — the adapter
exposes high-level operations directly on a class. It returns plain project types; Qdrant SDK objects
(``models.*``, ``ScoredPoint``) never leave this module. The shape still mirrors
``opensearch_client.py``: a Protocol, a concrete implementation, a typed error, and a factory.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

DEFAULT_QUERY_LIMIT = 50


class QdrantError(Exception):
    def __init__(
        self,
        message: str,
        *,
        reason: str,
        status_code: int | None = None,
        detail: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.reason = reason
        self.status_code = status_code
        self.detail = detail or {}


class VectorPoint(BaseModel):
    """One point to upsert: a stable id, its dense vector, and the filterable payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    point_id: str = Field(min_length=1)
    vector: list[float]
    payload: dict[str, Any] = Field(default_factory=dict)


class VectorHit(BaseModel):
    """A single search hit, in plain project shape (no Qdrant ``ScoredPoint`` leaks out)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    point_id: str
    score: float
    payload: dict[str, Any] = Field(default_factory=dict)


class CollectionConfig(BaseModel):
    """The live collection's vector config, read back for the drift check."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vector_name: str
    dimensions: int
    distance: str


_DISTANCE_BY_NAME: dict[str, models.Distance] = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
    "manhattan": models.Distance.MANHATTAN,
}
_DISTANCE_TO_NAME: dict[models.Distance, str] = {
    value: name for name, value in _DISTANCE_BY_NAME.items()
}
_PAYLOAD_SCHEMA_TYPES: dict[str, models.PayloadSchemaType] = {
    "keyword": models.PayloadSchemaType.KEYWORD,
    "integer": models.PayloadSchemaType.INTEGER,
}


@runtime_checkable
class QdrantStore(Protocol):
    """The narrow set of vector-store operations this feature needs."""

    def create_collection(
        self, name: str, *, vector_name: str, dimensions: int, distance: str
    ) -> None: ...

    def create_payload_indexes(self, name: str, fields: Mapping[str, str]) -> None: ...

    def upsert_points(
        self, name: str, *, vector_name: str, points: Sequence[VectorPoint]
    ) -> None: ...

    def count_points(self, name: str) -> int: ...

    def collection_exists(self, name: str) -> bool: ...

    def collection_config(self, name: str) -> CollectionConfig: ...

    def query(
        self,
        name: str,
        *,
        vector: Sequence[float],
        vector_name: str,
        query_filter: models.Filter | None = None,
        limit: int = DEFAULT_QUERY_LIMIT,
        with_payload: bool = True,
    ) -> list[VectorHit]: ...

    def resolve_alias(self, alias: str) -> str: ...

    def swap_alias(self, alias: str, new_collection: str) -> None: ...

    def delete_collection(self, name: str) -> None: ...

    def close(self) -> None: ...


@contextmanager
def _translate_errors(reason: str, detail: dict[str, Any] | None = None) -> Iterator[None]:
    """Wrap qdrant-client transport/response failures as a typed ``QdrantError``."""
    try:
        yield
    except QdrantError:
        raise
    except (UnexpectedResponse, ResponseHandlingException) as exc:
        raise QdrantError(str(exc), reason=reason, detail=detail or {}) from exc


class QdrantClientStore:
    """Concrete :class:`QdrantStore` wrapping a ``qdrant_client.QdrantClient``."""

    def __init__(self, client: QdrantClient):
        self._client = client

    def create_collection(
        self, name: str, *, vector_name: str, dimensions: int, distance: str
    ) -> None:
        distance_enum = _DISTANCE_BY_NAME.get(distance.casefold())
        if distance_enum is None:
            raise QdrantError(
                f"unsupported distance metric {distance!r}",
                reason="qdrant_request_failed",
                detail={"distance": distance},
            )
        with _translate_errors("qdrant_request_failed", {"collection": name}):
            self._client.create_collection(
                collection_name=name,
                vectors_config={
                    vector_name: models.VectorParams(size=dimensions, distance=distance_enum)
                },
            )

    def create_payload_indexes(self, name: str, fields: Mapping[str, str]) -> None:
        with _translate_errors("qdrant_request_failed", {"collection": name}):
            for field_name, kind in fields.items():
                schema = _PAYLOAD_SCHEMA_TYPES.get(kind)
                if schema is None:
                    raise QdrantError(
                        f"unsupported payload index kind {kind!r}",
                        reason="qdrant_request_failed",
                        detail={"field": field_name, "kind": kind},
                    )
                self._client.create_payload_index(
                    collection_name=name, field_name=field_name, field_schema=schema
                )

    def upsert_points(self, name: str, *, vector_name: str, points: Sequence[VectorPoint]) -> None:
        structs = [
            models.PointStruct(
                id=point.point_id,
                vector={vector_name: list(point.vector)},
                payload=dict(point.payload),
            )
            for point in points
        ]
        if not structs:
            return
        with _translate_errors("qdrant_request_failed", {"collection": name}):
            self._client.upsert(collection_name=name, points=structs, wait=True)

    def count_points(self, name: str) -> int:
        with _translate_errors("qdrant_request_failed", {"collection": name}):
            return self._client.count(collection_name=name, exact=True).count

    def collection_exists(self, name: str) -> bool:
        with _translate_errors("qdrant_request_failed", {"collection": name}):
            return self._client.collection_exists(collection_name=name)

    def collection_config(self, name: str) -> CollectionConfig:
        with _translate_errors("qdrant_request_failed", {"collection": name}):
            info = self._client.get_collection(collection_name=name)
        vectors = info.config.params.vectors
        if not isinstance(vectors, dict) or len(vectors) != 1:
            raise QdrantError(
                f"collection {name} must have exactly one named vector",
                reason="semantic_collection_config_invalid",
                detail={"collection": name},
            )
        resolved_name, params = next(iter(vectors.items()))
        return CollectionConfig(
            vector_name=resolved_name,
            dimensions=params.size,
            distance=_DISTANCE_TO_NAME.get(params.distance, str(params.distance)),
        )

    def query(
        self,
        name: str,
        *,
        vector: Sequence[float],
        vector_name: str,
        query_filter: models.Filter | None = None,
        limit: int = DEFAULT_QUERY_LIMIT,
        with_payload: bool = True,
    ) -> list[VectorHit]:
        with _translate_errors("qdrant_request_failed", {"collection": name}):
            response = self._client.query_points(
                collection_name=name,
                query=list(vector),
                using=vector_name,
                query_filter=query_filter,
                limit=limit,
                with_payload=with_payload,
            )
        return [
            VectorHit(point_id=str(point.id), score=float(point.score), payload=point.payload or {})
            for point in response.points
        ]

    def resolve_alias(self, alias: str) -> str:
        with _translate_errors("qdrant_request_failed", {"alias": alias}):
            descriptions = self._client.get_aliases().aliases
        targets = [d.collection_name for d in descriptions if d.alias_name == alias]
        if len(targets) != 1:
            raise QdrantError(
                f"alias {alias} must resolve to exactly one collection",
                reason="semantic_alias_invalid",
                detail={"alias": alias, "targets": targets},
            )
        return targets[0]

    def swap_alias(self, alias: str, new_collection: str) -> None:
        # Delete-then-create in one call so activation/rollback is an atomic alias repoint. Deleting a
        # not-yet-existing alias (first activation) is a no-op.
        operations = [
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)),
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name=new_collection, alias_name=alias)
            ),
        ]
        with _translate_errors(
            "qdrant_request_failed", {"alias": alias, "collection": new_collection}
        ):
            self._client.update_collection_aliases(change_aliases_operations=operations)

    def delete_collection(self, name: str) -> None:
        with _translate_errors("qdrant_request_failed", {"collection": name}):
            self._client.delete_collection(collection_name=name)

    def close(self) -> None:
        self._client.close()


def build_qdrant_store(
    *,
    url: str,
    api_key: str = "",
    timeout_seconds: float = 10.0,
    prefer_grpc: bool = False,
) -> QdrantClientStore:
    if not url:
        raise QdrantError("Qdrant URL is not configured", reason="qdrant_not_configured")
    client = QdrantClient(
        url=url,
        api_key=api_key or None,
        timeout=int(timeout_seconds),
        prefer_grpc=prefer_grpc,
    )
    return QdrantClientStore(client)
