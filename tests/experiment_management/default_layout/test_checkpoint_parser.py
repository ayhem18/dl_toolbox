from __future__ import annotations

import pytest

from dl_toolbox.experiment_management.default_layout.parsing.checkpoint import (
    CheckpointProjectConfigParser,
)


def test_checkpoint_parser_reads_exact_keys() -> None:
    parsed = CheckpointProjectConfigParser({"lock_timeout_seconds": 12.5})
    assert parsed.lock_timeout_seconds == 12.5


def test_checkpoint_parser_rejects_extra_keys() -> None:
    with pytest.raises(ValueError, match="keys must be"):
        CheckpointProjectConfigParser(
            {"lock_timeout_seconds": 12.5, "save_top_k": 1}
        )


def test_checkpoint_parser_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="lock_timeout_seconds"):
        CheckpointProjectConfigParser({"lock_timeout_seconds": 0})
    with pytest.raises(ValueError, match="lock_timeout_seconds"):
        CheckpointProjectConfigParser({"lock_timeout_seconds": -1})
