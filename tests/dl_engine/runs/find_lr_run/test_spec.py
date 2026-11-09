from __future__ import annotations

import inspect

import pytest

from dl_toolbox.dl_engine.runs.lr_finder_run.spec import (
    FindLrError,
    FindLrSpec,
    log_spaced_lrs,
    parse_find_lr_spec,
)
from tests.dl_engine.runs.find_lr_run.support import full_ds_spec, range_spec


def test_constructors_have_no_defaults() -> None:
    for name in ("range", "full_ds"):
        params = inspect.signature(getattr(FindLrSpec, name)).parameters
        for param_name, param in params.items():
            if param_name == "cls":
                continue
            assert param.default is inspect.Parameter.empty, param_name
            assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_direct_constructor_is_error() -> None:
    with pytest.raises(FindLrError, match="not a constructor"):
        FindLrSpec()


def test_range_to_dict_roundtrip() -> None:
    spec = range_spec()
    parsed = parse_find_lr_spec(spec.to_dict())
    assert parsed.kind == FindLrSpec.RANGE
    assert parsed.start_lr == spec.start_lr
    assert parsed.end_lr == spec.end_lr
    assert parsed.num_iter == spec.num_iter
    assert parsed.smooth_f == spec.smooth_f
    assert parsed.diverge_th == spec.diverge_th
    assert not hasattr(parsed, "num_iterations_per_lr")


def test_full_ds_to_dict_roundtrip() -> None:
    spec = full_ds_spec(num_iter=4, num_iterations_per_lr=3)
    parsed = parse_find_lr_spec(spec.to_dict())
    assert parsed.kind == FindLrSpec.FULL_DS
    assert parsed.num_iterations_per_lr == 3
    assert not hasattr(parsed, "smooth_f")
    assert not hasattr(parsed, "diverge_th")


def test_parse_rejects_wrong_keys_for_kind() -> None:
    payload = range_spec().to_dict()
    payload["num_iterations_per_lr"] = 1
    with pytest.raises(FindLrError, match="Extra inputs are not permitted"):
        parse_find_lr_spec(payload)
    payload = full_ds_spec(num_iter=4, num_iterations_per_lr=1).to_dict()
    payload["smooth_f"] = 0.05
    with pytest.raises(FindLrError, match="Extra inputs are not permitted"):
        parse_find_lr_spec(payload)


def test_parse_rejects_unknown_kind() -> None:
    with pytest.raises(FindLrError, match="kind must be"):
        parse_find_lr_spec(
            {
                "kind": "walk",
                "start_lr": 1e-4,
                "end_lr": 1e-1,
                "num_iter": 4,
                "smooth_f": 0.05,
                "diverge_th": 4.0,
            }
        )


def test_log_spaced_lrs_are_inclusive_and_increasing() -> None:
    spec = range_spec()
    lrs = log_spaced_lrs(spec)
    assert len(lrs) == spec.num_iter
    assert lrs[0] == pytest.approx(spec.start_lr)
    assert lrs[-1] == pytest.approx(spec.end_lr)
    for i in range(1, len(lrs)):
        assert lrs[i] > lrs[i - 1]


def test_range_rejects_inverted_short_and_diverge() -> None:
    with pytest.raises(FindLrError, match="start_lr"):
        FindLrSpec.range(
            start_lr=1e-1,
            end_lr=1e-4,
            num_iter=4,
            smooth_f=0.05,
            diverge_th=4.0,
        )
    with pytest.raises(FindLrError, match="num_iter"):
        FindLrSpec.range(
            start_lr=1e-4,
            end_lr=1e-1,
            num_iter=1,
            smooth_f=0.05,
            diverge_th=4.0,
        )
    with pytest.raises(FindLrError, match="diverge_th"):
        FindLrSpec.range(
            start_lr=1e-4,
            end_lr=1e-1,
            num_iter=4,
            smooth_f=0.05,
            diverge_th=1.0,
        )


def test_range_rejects_int_floats() -> None:
    with pytest.raises(FindLrError, match="start_lr must be a float"):
        FindLrSpec.range(
            start_lr=1,
            end_lr=10.0,
            num_iter=4,
            smooth_f=0.05,
            diverge_th=4.0,
        )


def test_full_ds_rejects_non_positive_passes() -> None:
    with pytest.raises(FindLrError, match="num_iterations_per_lr"):
        FindLrSpec.full_ds(
            start_lr=1e-4,
            end_lr=1e-1,
            num_iter=4,
            num_iterations_per_lr=0,
        )
