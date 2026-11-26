"""Backend-specific ``location`` models. A ``location`` crosses the connection
boundary as a JSON object (dict); each backend parses it with its own model so
the dict is validated without a hand-rolled check.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    ValidationError,
    field_validator,
)

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)
_NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _require_relative_key(value: str, *, label: str) -> str:
    text = value.replace("\\", "/")
    if text.startswith("/"):
        raise ValueError(f"{label} must be relative (no leading '/'), got {value!r}")
    segments = text.split("/")
    n = len(segments)
    for i in range(n):
        if segments[i] in {"", ".", ".."}:
            raise ValueError(
                f"{label} must have no empty, '.' or '..' segment, got {value!r}"
            )
    return text


class S3Location(BaseModel):
    """``bucket`` plus a relative object key / prefix within it."""

    model_config = _STRICT

    bucket: _NonEmptyStr
    path_within_bucket: _NonEmptyStr

    @field_validator("path_within_bucket")
    @classmethod
    def _key(cls, value: str) -> str:
        return _require_relative_key(value, label="path_within_bucket")


class HuggingFaceLocation(BaseModel):
    """A Hugging Face repo plus a relative path inside it. (Backend deferred.)"""

    model_config = _STRICT

    repo_id: _NonEmptyStr
    repo_type: Literal["dataset", "model", "space"]
    path_in_repo: _NonEmptyStr

    @field_validator("path_in_repo")
    @classmethod
    def _path(cls, value: str) -> str:
        return _require_relative_key(value, label="path_in_repo")


def parse_s3_location(location: object) -> S3Location:
    try:
        return S3Location.model_validate(location)
    except ValidationError as exc:
        raise ValueError(f"S3 location: {exc}") from exc
