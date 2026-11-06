# PARKED: no contract yet. Not collected by pytest until this package is rewritten.
from __future__ import annotations

import pytest

pytest.skip("Parked: this package has no contract yet.", allow_module_level=True)

import pytest

from dl_toolbox.dl_engine.lr_finder import (
    LrFinderError,
    LrFinderSpec,
    parse_lr_finder_spec,
    parse_num_iterations_per_lr,
)
from dl_toolbox.dl_engine.lr_finder.spec import log_spaced_lrs, validate_lr_finder_spec
from tests.dl_engine.lr_finder.support import make_spec


def test_parse_lr_finder_spec_requires_exact_keys() -> None:
    spec = parse_lr_finder_spec({"lr_finder": make_spec().to_dict(), "task": "tiny"})
    assert spec.start_lr == pytest.approx(1e-4)
    assert spec.num_iter == 4
    with pytest.raises(LrFinderError, match="must contain"):
        parse_lr_finder_spec({"task": "tiny"})
    with pytest.raises(ValueError, match="lr_finder keys"):
        parse_lr_finder_spec({"lr_finder": {"start_lr": 1e-4}})


def test_parse_num_iterations_per_lr_defaults_to_one() -> None:
    run_config = {"lr_finder": make_spec().to_dict()}
    assert parse_num_iterations_per_lr(run_config) == 1
    payload = dict(make_spec().to_dict())
    payload["num_iterations_per_lr"] = 3
    assert parse_num_iterations_per_lr({"lr_finder": payload}) == 3
    spec = parse_lr_finder_spec({"lr_finder": payload})
    assert spec.num_iter == 4


def test_log_spaced_lrs_are_inclusive_and_increasing() -> None:
    spec = make_spec()
    lrs = log_spaced_lrs(spec)
    assert len(lrs) == spec.num_iter
    assert lrs[0] == pytest.approx(spec.start_lr)
    assert lrs[-1] == pytest.approx(spec.end_lr)
    for i in range(1, len(lrs)):
        assert lrs[i] > lrs[i - 1]


def test_validate_rejects_inverted_range_and_short_sweep() -> None:
    with pytest.raises(LrFinderError, match="start_lr"):
        validate_lr_finder_spec(
            LrFinderSpec(
                start_lr=1e-1,
                end_lr=1e-4,
                num_iter=4,
                smooth_f=0.05,
                diverge_th=4.0,
            )
        )
    with pytest.raises(LrFinderError, match="num_iter"):
        validate_lr_finder_spec(
            LrFinderSpec(
                start_lr=1e-4,
                end_lr=1e-1,
                num_iter=1,
                smooth_f=0.05,
                diverge_th=4.0,
            )
        )
    with pytest.raises(LrFinderError, match="diverge_th"):
        parse_lr_finder_spec(
            {
                "lr_finder": {
                    "start_lr": 1e-4,
                    "end_lr": 1e-1,
                    "num_iter": 4,
                    "smooth_f": 0.05,
                    "diverge_th": 1.0,
                }
            }
        )
