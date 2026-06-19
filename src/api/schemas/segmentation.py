from datetime import datetime
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JsonObject = dict[str, Any]
HEX_COLOR_PATTERN = re.compile(r"^#[0-9a-fA-F]{6}$")


class AyahTranslationInput(BaseModel):
    lang: str
    text: str


class AyahInput(BaseModel):
    id: int | None = None
    number_in_surah: int = Field(ge=1)
    text_ar: str | None = None
    translations: list[AyahTranslationInput] = Field(default_factory=list)


class SegmentationOptions(BaseModel):
    model_config = ConfigDict(extra="allow")

    granularity: str = "medium"
    max_segments: int = Field(default=20, ge=1)
    include_tags: bool = True
    include_summaries: bool = True


class SegmentationContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    workspace_slug: str | None = None
    requested_by_user_id: int | None = None


class SegmentationGenerateRequest(BaseModel):
    surah_id: int = Field(ge=1)
    ayahs: list[AyahInput] = Field(default_factory=list)
    options: SegmentationOptions = Field(default_factory=SegmentationOptions)
    context: SegmentationContext = Field(default_factory=SegmentationContext)


class SegmentTag(BaseModel):
    name: str
    color: str = "#22c55e"
    description: str = ""


class GeneratedSegment(BaseModel):
    start_ayah: int = Field(ge=1)
    end_ayah: int = Field(ge=1)
    title: str = ""
    summary: str = ""
    tags: list[SegmentTag] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_range(self) -> "GeneratedSegment":
        if self.start_ayah > self.end_ayah:
            raise ValueError("start_ayah must be less than or equal to end_ayah")
        return self


class SegmentationGenerateResponse(BaseModel):
    external_id: str
    model_name: str
    model_version: str
    params: JsonObject = Field(default_factory=dict)
    produced_at: datetime
    segments: list[GeneratedSegment] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_sorted_non_overlapping_segments(self) -> "SegmentationGenerateResponse":
        previous_end = 0
        for segment in self.segments:
            if segment.start_ayah <= previous_end:
                raise ValueError("segments must be sorted and non-overlapping")
            previous_end = segment.end_ayah
        return self


class SegmentationArtifactManifestSurah(BaseModel):
    model_config = ConfigDict(extra="forbid")

    surah_number: int = Field(ge=1, le=114)
    segment_count: int = Field(ge=0)


class SegmentationArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    schema_version: str
    title: str
    description: str
    model_name: str
    model_version: str
    params: JsonObject
    produced_at: datetime
    surahs: list[SegmentationArtifactManifestSurah]

    @model_validator(mode="after")
    def validate_unique_surahs(self) -> "SegmentationArtifactManifest":
        surah_numbers = [surah.surah_number for surah in self.surahs]
        if len(surah_numbers) != len(set(surah_numbers)):
            raise ValueError("manifest surahs must not contain duplicate surah_number values")
        return self


class SegmentationArtifactTag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    color: str
    description: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("tag name must be non-empty")
        return value

    @field_validator("color")
    @classmethod
    def validate_color(cls, value: str) -> str:
        if not HEX_COLOR_PATTERN.fullmatch(value):
            raise ValueError("tag color must be a valid hex color")
        return value


class SegmentationArtifactSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_ayah_number: int = Field(ge=1)
    end_ayah_number: int = Field(ge=1)
    title: str
    summary: str
    tags: list[SegmentationArtifactTag]

    @model_validator(mode="after")
    def validate_range(self) -> "SegmentationArtifactSegment":
        if self.start_ayah_number > self.end_ayah_number:
            raise ValueError("start_ayah_number must be less than or equal to end_ayah_number")
        return self


class SegmentationArtifactSurahPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    external_id: str
    surah_number: int = Field(ge=1, le=114)
    model_name: str
    model_version: str
    params: JsonObject
    produced_at: datetime
    segments: list[SegmentationArtifactSegment]

    @model_validator(mode="after")
    def validate_sorted_non_overlapping_segments(self) -> "SegmentationArtifactSurahPayload":
        previous_end = 0
        for segment in self.segments:
            if segment.start_ayah_number <= previous_end:
                raise ValueError("segments must be sorted and non-overlapping")
            previous_end = segment.end_ayah_number
        return self
