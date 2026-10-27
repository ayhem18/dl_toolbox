"""Identity hashing primitives.

The stored identity is always the full SHA-256 of canonical JSON. Filenames
use a short prefix of that digest.

Do not expose ``prefix_len`` as a per-call knob at product call sites. One
project must use one prefix length for writing stems, recognizing stems, and
every other check. Passing different values on different calls is an easy,
silent mismatch.

Callers go through ``HashManager`` (project ``identity.json``: prefix length
plus the human-readable and hashes folder names). These functions stay the
digest primitives that object uses. They are not a second public identity API.
"""

from __future__ import annotations

import hashlib
import json

DEFAULT_ARTIFACT_PREFIX_LEN = 12
SHA256_HEX_LENGTH = 64
_HEX_CHARS = frozenset("0123456789abcdef")


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def hash_payload(payload: object) -> str:
    """Full SHA-256 of the canonical JSON. Do not truncate this digest."""
    return hash_text(canonical_json(payload))


def is_full_sha256(text: str) -> bool:
    if not isinstance(text, str) or len(text) != SHA256_HEX_LENGTH:
        return False
    if text != text.lower():
        return False
    return all(char in _HEX_CHARS for char in text)


def artifact_stem(
    full_hash: str,
    *,
    prefix_len: int = DEFAULT_ARTIFACT_PREFIX_LEN,
) -> str:
    """Short configuration filename stem. The document still stores ``full_hash``."""
    if not is_full_sha256(full_hash):
        raise ValueError(f"full_hash must be a 64-char lowercase SHA-256, got {full_hash!r}")
    if prefix_len < 8:
        raise ValueError(f"prefix_len must be at least 8, got {prefix_len}")
    if prefix_len > SHA256_HEX_LENGTH:
        raise ValueError(
            f"prefix_len ({prefix_len}) exceeds hash length ({SHA256_HEX_LENGTH})"
        )
    return full_hash[:prefix_len]


def is_hashed_artifact_stem(
    stem: str,
    *,
    prefix_len: int = DEFAULT_ARTIFACT_PREFIX_LEN,
) -> bool:
    """True when ``stem`` looks like a hash prefix or a full SHA-256."""
    if not isinstance(stem, str):
        return False
    n = len(stem)
    if n not in (prefix_len, SHA256_HEX_LENGTH):
        return False
    if stem != stem.lower():
        return False
    return all(char in _HEX_CHARS for char in stem)
