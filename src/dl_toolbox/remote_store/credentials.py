"""``RemoteStoreCredentials``: what you need to authenticate, plus the one
``connect()`` that touches a vendor SDK or a secret. Secrets are held as
``SecretStr`` so repr/str never print them.

The on-disk form is split: the non-secret config lives in the committed
``remote.json``, and only the secret keys live in a per-backend credential file
under ``remote/credentials/`` that scaffolding git-ignores. A credential file
is scaffolded blank per backend (``blank_credential_document``) so the user
never guesses which keys a backend needs; the matching ``*CredentialsFile``
model validates it at connect time and a blank value fails with a clear message.
"""

from __future__ import annotations

import abc
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, SecretStr, StringConstraints

from dl_toolbox.base.packaging.require import require
from dl_toolbox.remote_store.backends import RemoteBackend, parse_backend
from dl_toolbox.remote_store.connection import RemoteStoreConnection

_NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class RemoteStoreCredentials(abc.ABC):
    @property
    @abc.abstractmethod
    def backend(self) -> str:
        """``RemoteBackend`` value these credentials authenticate against."""

    @abc.abstractmethod
    def connect(self) -> RemoteStoreConnection:
        """Authenticate and return a live ``RemoteStoreConnection``."""


class S3Credentials(RemoteStoreCredentials, BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint_url: _NonEmptyStr
    access_key_id: SecretStr
    secret_access_key: SecretStr
    region: _NonEmptyStr
    secure: bool
    addressing_style: Literal["path", "virtual"]

    @property
    def backend(self) -> str:
        return RemoteBackend.S3

    def connect(self) -> RemoteStoreConnection:
        boto3 = require(package="boto3")
        from botocore.config import Config
        from botocore.exceptions import BotoCoreError, ClientError

        from dl_toolbox.remote_store.s3_connection import S3Connection

        try:
            client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key_id.get_secret_value(),
                aws_secret_access_key=self.secret_access_key.get_secret_value(),
                region_name=self.region,
                use_ssl=self.secure,
                config=Config(
                    signature_version="s3v4",
                    s3={"addressing_style": self.addressing_style},
                ),
            )
        except (BotoCoreError, ClientError) as exc:
            raise ConnectionError(
                f"failed to create S3 client for {self.endpoint_url}"
            ) from exc
        return S3Connection(client=client)


class S3CredentialsFile(BaseModel):
    """The secret half of S3 credentials, as stored on disk. Non-empty."""

    model_config = ConfigDict(extra="forbid", strict=True)

    access_key_id: Annotated[str, StringConstraints(min_length=1)]
    secret_access_key: Annotated[str, StringConstraints(min_length=1)]


class HuggingFaceCredentialsFile(BaseModel):
    """The secret half of Hugging Face credentials, as stored on disk. Non-empty."""

    model_config = ConfigDict(extra="forbid", strict=True)

    token: Annotated[str, StringConstraints(min_length=1)]


def credential_file_model(backend: str) -> type[BaseModel]:
    """The ``*CredentialsFile`` model that validates one backend's secret file."""
    parsed = parse_backend(backend)
    if parsed == RemoteBackend.S3:
        return S3CredentialsFile
    return HuggingFaceCredentialsFile


def blank_credential_document(backend: str) -> dict[str, str]:
    """A blank secret file for ``backend``: its keys, every value empty.

    The keys come from the model's fields, so the scaffolded template and the
    connect-time validation can never drift.
    """
    model = credential_file_model(backend)
    return {name: "" for name in model.model_fields}
