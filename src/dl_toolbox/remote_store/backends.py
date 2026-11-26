"""``RemoteBackend`` names the remote stores, like ``TrackingBackend``."""

from __future__ import annotations

from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import one_of_message, parse_one_of


class RemoteBackend:
    S3 = "s3"
    HUGGINGFACE = "huggingface"
    BACKENDS = (S3, HUGGINGFACE)


def parse_backend(raw: object) -> str:
    backend = parse_non_empty_str(raw, label="backend")
    parsed = parse_one_of(
        backend,
        allowed=RemoteBackend.BACKENDS,
        label="backend",
        error_type=ValueError,
        error_message=one_of_message,
    )
    if not isinstance(parsed, str):
        raise TypeError(f"backend must be str, got {type(parsed)!r}")
    return parsed
