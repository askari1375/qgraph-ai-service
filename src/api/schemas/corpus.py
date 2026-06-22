from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JsonObject = dict[str, Any]


class CorpusAyahTranslation(BaseModel):
    model_config = ConfigDict(extra="allow")

    language_code: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @field_validator("language_code", "source_id", "source_name", "text", mode="before")
    @classmethod
    def validate_non_empty_text(cls, value: Any) -> str:
        if isinstance(value, bool) or value is None:
            raise ValueError("value must be a non-empty string")
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("value must be a non-empty string")
        return cleaned


class CorpusAyah(BaseModel):
    model_config = ConfigDict(extra="allow")

    surah_number: int = Field(ge=1, le=114)
    ayah_number: int = Field(ge=1)
    ayah_global_number: int = Field(ge=1)
    text_ar: str = Field(min_length=1)
    translations: list[CorpusAyahTranslation]

    @field_validator("text_ar", mode="before")
    @classmethod
    def validate_text_ar(cls, value: Any) -> str:
        if isinstance(value, bool) or value is None:
            raise ValueError("text_ar must be a non-empty string")
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("text_ar must be a non-empty string")
        return cleaned


class QuranCorpusSnapshot(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = Field(min_length=1)
    corpus_snapshot_id: str = Field(min_length=1)
    corpus_snapshot_hash: str = Field(min_length=1)
    produced_at: datetime
    filters: JsonObject
    counts: JsonObject
    translation_sources: list[JsonObject]
    surahs: list[JsonObject]
    ayahs: list[CorpusAyah]

    @field_validator("schema_version", "corpus_snapshot_id", "corpus_snapshot_hash", mode="before")
    @classmethod
    def validate_snapshot_text(cls, value: Any) -> str:
        if isinstance(value, bool) or value is None:
            raise ValueError("value must be a non-empty string")
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("value must be a non-empty string")
        return cleaned

    @model_validator(mode="after")
    def validate_unique_ayahs(self) -> "QuranCorpusSnapshot":
        seen: set[tuple[int, int]] = set()
        duplicates: list[str] = []
        for ayah in self.ayahs:
            key = (ayah.surah_number, ayah.ayah_number)
            if key in seen:
                duplicates.append(f"{ayah.surah_number}:{ayah.ayah_number}")
            seen.add(key)
        if duplicates:
            joined = ", ".join(duplicates)
            raise ValueError(f"ayahs must not contain duplicate surah/ayah pairs: {joined}")
        return self
