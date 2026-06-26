from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

JsonObject = dict[str, Any]
SearchMode = Literal["sync", "async"]


class RequesterContext(BaseModel):
    """Minimal requester facts the planner uses to choose a response policy.

    Django builds this from the request's authentication state. It is kept
    deliberately small — only the real, current signal (authenticated vs. guest)
    — so that richer user-tier data is added here only when a near-term policy
    actually needs it.
    """

    is_authenticated: bool = False
    is_guest: bool = True

    @classmethod
    def from_context(cls, context: JsonObject | None) -> "RequesterContext":
        requester = context.get("requester") if isinstance(context, dict) else None
        if not isinstance(requester, dict):
            return cls()
        is_authenticated = bool(requester.get("is_authenticated", False))
        is_guest = bool(requester.get("is_guest", not is_authenticated))
        return cls(is_authenticated=is_authenticated, is_guest=is_guest)


class SearchPlanRequest(BaseModel):
    query: str = Field(min_length=1)
    filters: JsonObject = Field(default_factory=dict)
    output_preferences: JsonObject = Field(default_factory=dict)
    context: JsonObject = Field(default_factory=dict)


class SearchPlanResponse(BaseModel):
    mode: SearchMode
    policy_label: str
    policy_snapshot: JsonObject = Field(default_factory=dict)
    routing_metadata: JsonObject = Field(default_factory=dict)
    backend_name: str
    backend_version: str


class SearchExecuteRequest(BaseModel):
    query: str = Field(min_length=1)
    filters: JsonObject = Field(default_factory=dict)
    output_preferences: JsonObject = Field(default_factory=dict)
    context: JsonObject = Field(default_factory=dict)


class SearchResultItem(BaseModel):
    rank: int = Field(ge=1)
    result_type: str
    score: float = Field(ge=0.0, le=1.0)
    title: str
    snippet_text: str
    highlighted_text: str
    match_metadata: JsonObject = Field(default_factory=dict)
    explanation: str = ""
    provenance: JsonObject = Field(default_factory=dict)


class SearchResponseBlock(BaseModel):
    order: int = Field(ge=0)
    block_type: str
    title: str
    payload: JsonObject = Field(default_factory=dict)
    explanation: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: JsonObject = Field(default_factory=dict)
    warning_text: str = ""
    items: list[SearchResultItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_item_ranks(self) -> "SearchResponseBlock":
        ranks = [item.rank for item in self.items]
        if len(ranks) != len(set(ranks)):
            raise ValueError("items.rank must be unique within each block")
        return self


class SearchExecuteResponse(BaseModel):
    title: str
    overall_confidence: float = Field(ge=0.0, le=1.0)
    render_schema_version: str
    metadata: JsonObject = Field(default_factory=dict)
    blocks: list[SearchResponseBlock] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_block_orders(self) -> "SearchExecuteResponse":
        orders = [block.order for block in self.blocks]
        if len(orders) != len(set(orders)):
            raise ValueError("blocks.order must be unique")
        return self


class SearchJobClientRef(BaseModel):
    query_id: int | None = None
    execution_id: int | None = None


class SearchJobCreateRequest(BaseModel):
    query: str = Field(min_length=1)
    filters: JsonObject = Field(default_factory=dict)
    output_preferences: JsonObject = Field(default_factory=dict)
    context: JsonObject = Field(default_factory=dict)
    idempotency_key: str = Field(min_length=1, max_length=128)
    client_ref: SearchJobClientRef = Field(default_factory=SearchJobClientRef)
