"""Pydantic documents for project-level JSON. Domain rules stay on ProjectLayout."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from dl_toolbox.base.hashing.identity import SHA256_HEX_LENGTH
from dl_toolbox.base.parsing.parse_numerical import parse_positive_number

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)
_NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class MarkerDocument(BaseModel):
    model_config = _STRICT

    name: _NonEmptyStr


class IdentityDocument(BaseModel):
    model_config = _STRICT

    prefix_len: Annotated[int, Field(ge=8, le=SHA256_HEX_LENGTH)]
    human_readable_dirname: _NonEmptyStr
    hashes_dirname: _NonEmptyStr


class CheckpointDocument(BaseModel):
    model_config = _STRICT

    lock_timeout_seconds: float

    @field_validator("lock_timeout_seconds", mode="before")
    @classmethod
    def _timeout(cls, value: object) -> float:
        return parse_positive_number(value, label="lock_timeout_seconds")


class TensorBoardLoggingDocument(BaseModel):
    model_config = _STRICT

    log_dir: _NonEmptyStr


class MlflowBackendStoreDocument(BaseModel):
    model_config = _STRICT

    type: Literal["database", "filesystem"]
    database_type: Literal["sqlite", "postgres"]
    path: _NonEmptyStr
    timeout_seconds: float

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def _timeout(cls, value: object) -> float:
        return parse_positive_number(value, label="backend_store.timeout_seconds")


class MlflowArtifactStoreDocument(BaseModel):
    model_config = _STRICT

    type: Literal["filesystem"]
    root: _NonEmptyStr


class MlflowLoggingDocument(BaseModel):
    model_config = _STRICT

    backend_store: MlflowBackendStoreDocument
    artifact_store: MlflowArtifactStoreDocument
