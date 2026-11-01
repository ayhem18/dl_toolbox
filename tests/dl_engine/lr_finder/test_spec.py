from __future__ import annotations

import pytest

from dl_toolbox.dl_engine.lr_finder import LrFinderError, LrFinderSpec, parse_lr_finder_spec
from dl_toolbox.dl_engine.lr_finder.spec import validate_lr_finder_spec
from tests.dl_engine.lr_finder.support import make_spec


def test_parse_lr_finder_spec_requires_exact_keys() -> None:
    spec = parse_lr_finder_spec({"lr_finder": make_spec().to_dict(), "task": "tiny"})
    assert spec.start_lr == pytest.approx(1e-4)
    assert spec.num_iter == 4
    with pytest.raises(LrFinderError, match="must contain"):
        parse_lr_finder_spec({"task": "tiny"})
    with pytest.raises(ValueError, match="lr_finder keys"):
        parse_lr_finder_spec({"lr_finder": {"start_lr": 1e-4}})


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
