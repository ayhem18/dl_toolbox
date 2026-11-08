from __future__ import annotations

import pytest

from dl_toolbox.base.parsing.parse_utils import one_of_message, parse_one_of


class _CustomError(ValueError):
    pass


def test_returns_value_when_allowed() -> None:
    assert (
        parse_one_of(
            "fit",
            allowed=("fit", "inspect"),
            label="purpose",
            error_type=ValueError,
            error_message=one_of_message,
        )
        == "fit"
    )


def test_rejects_value_outside_allowed() -> None:
    with pytest.raises(ValueError) as ctx:
        parse_one_of(
            "train",
            allowed=("fit", "inspect"),
            label="purpose",
            error_type=ValueError,
            error_message=one_of_message,
        )
    message = str(ctx.value)
    assert message == "purpose must be 'fit' or 'inspect', got 'train'"


def test_three_or_more_allowed_uses_commas_and_or() -> None:
    with pytest.raises(ValueError) as ctx:
        parse_one_of(
            "other",
            allowed=("both_free", "both_match", "name_taken", "hash_taken"),
            label="pair state",
            error_type=ValueError,
            error_message=one_of_message,
        )
    assert str(ctx.value) == (
        "pair state must be 'both_free', 'both_match', "
        "'name_taken', or 'hash_taken', got 'other'"
    )


def test_error_type_is_the_raised_class() -> None:
    with pytest.raises(_CustomError) as ctx:
        parse_one_of(
            "nope",
            allowed=("create", "require"),
            label="existence",
            error_type=_CustomError,
            error_message=one_of_message,
        )
    assert type(ctx.value) is _CustomError
    assert "existence must be 'create' or 'require'" in str(ctx.value)


def test_custom_error_message_callable() -> None:
    def _overwrite_message(
        *,
        value: object,
        allowed: object,
        label: str,
        allowed_text: str,
    ) -> str:
        del allowed, label
        return f"overwrite policy must be {allowed_text}, got {value!r}"

    with pytest.raises(ValueError) as ctx:
        parse_one_of(
            "x",
            allowed=("off", "overwrite_name"),
            label="overwrite policy",
            error_type=ValueError,
            error_message=_overwrite_message,
        )
    assert str(ctx.value) == (
        "overwrite policy must be 'off' or 'overwrite_name', got 'x'"
    )


def test_non_str_value_fails_membership() -> None:
    with pytest.raises(ValueError, match="purpose"):
        parse_one_of(
            None,
            allowed=("fit", "inspect"),
            label="purpose",
            error_type=ValueError,
            error_message=one_of_message,
        )


def test_rejects_string_as_allowed() -> None:
    with pytest.raises(TypeError, match="allowed"):
        parse_one_of(
            "a",
            allowed="abc",
            label="letter",
            error_type=ValueError,
            error_message=one_of_message,
        )


def test_rejects_empty_allowed() -> None:
    with pytest.raises(ValueError, match="allowed must be non-empty"):
        parse_one_of(
            "fit",
            allowed=(),
            label="purpose",
            error_type=ValueError,
            error_message=one_of_message,
        )


def test_rejects_non_callable_error_message() -> None:
    with pytest.raises(TypeError, match="error_message must be callable"):
        parse_one_of(
            "fit",
            allowed=("fit", "inspect"),
            label="purpose",
            error_type=ValueError,
            error_message="{label} must be {allowed}, got {value!r}",
        )
