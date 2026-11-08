"""``generating_process_name`` is HashManager ``validate_human_readable_name``."""

from __future__ import annotations

from dl_toolbox.base.hashing.hash_manager import HashManager


def assert_generating_process_name(raw: object, *, manager: HashManager) -> str:
    return manager.validate_human_readable_name(raw)
