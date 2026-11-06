# PARKED: no contract yet. Not collected by pytest until this package is rewritten.
from __future__ import annotations

import pytest

pytest.skip("Parked: this package has no contract yet.", allow_module_level=True)

import pytest

from pydantic import ValidationError

from dl_toolbox.experiment_management.default_layout.parsing.identity import (
    IdentityConfigParser,
    identity_from_pydantic,
)


def test_identity_parser_reads_exact_keys() -> None:
    parsed = IdentityConfigParser(
        {
            "prefix_len": 12,
            "human_readable_dirname": "human_readable",
            "hashes_dirname": "hashes",
        }
    )
    assert parsed.prefix_len == 12
    assert parsed.human_readable_dirname == "human_readable"
    assert parsed.hashes_dirname == "hashes"


def test_identity_parser_rejects_extra_keys() -> None:
    with pytest.raises(ValueError, match="keys must be"):
        IdentityConfigParser(
            {
                "prefix_len": 12,
                "human_readable_dirname": "human_readable",
                "hashes_dirname": "hashes",
                "extra": 1,
            }
        )


def test_identity_parser_rejects_identical_folder_names() -> None:
    with pytest.raises(ValueError, match="distinct"):
        IdentityConfigParser(
            {
                "prefix_len": 12,
                "human_readable_dirname": "same",
                "hashes_dirname": "same",
            }
        )


def test_identity_parser_rejects_short_prefix() -> None:
    with pytest.raises(ValueError, match="prefix_len"):
        IdentityConfigParser(
            {
                "prefix_len": 4,
                "human_readable_dirname": "human_readable",
                "hashes_dirname": "hashes",
            }
        )


def test_identity_pydantic_matches_parser() -> None:
    payload = {
        "prefix_len": 12,
        "human_readable_dirname": "human_readable",
        "hashes_dirname": "hashes",
    }
    parsed = identity_from_pydantic(payload)
    original = IdentityConfigParser(payload)
    assert parsed.prefix_len == original.prefix_len
    assert parsed.human_readable_dirname == original.human_readable_dirname
    assert parsed.hashes_dirname == original.hashes_dirname
    with pytest.raises(ValidationError):
        identity_from_pydantic({**payload, "extra": 1})
    with pytest.raises(ValidationError):
        identity_from_pydantic({**payload, "prefix_len": 4})
    with pytest.raises(ValueError, match="distinct"):
        identity_from_pydantic(
            {
                "prefix_len": 12,
                "human_readable_dirname": "same",
                "hashes_dirname": "same",
            }
        )
