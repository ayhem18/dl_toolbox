"""``generating_process_name`` is one execution. It is not identity."""

from __future__ import annotations

from dl_toolbox.base.hashing.identity import (
    DEFAULT_ARTIFACT_PREFIX_LEN,
    is_hashed_artifact_stem,
)


def assert_generating_process_name(
    raw: object,
    *,
    prefix_len: int = DEFAULT_ARTIFACT_PREFIX_LEN,
) -> str:
    text = str(raw).strip()
    if text.lower().endswith(".json"):
        text = text[:-5].strip()
    text = text.replace("\\", "/")
    if not text:
        raise ValueError("generating_process_name is empty")
    if "/" in text or text in {".", ".."}:
        raise ValueError(f"generating_process_name must be a basename: {text!r}")
    if is_hashed_artifact_stem(text, prefix_len=prefix_len):
        raise ValueError(
            f"generating_process_name must not be a hash stem: {text!r}"
        )
    return text
