# PARKED: no contract yet. Not collected by pytest until this package is rewritten.
from __future__ import annotations

import pytest

pytest.skip("Parked: this package has no contract yet.", allow_module_level=True)

from tqdm import tqdm

from dl_toolbox.dl_engine_utils.progress import range_progress


def test_disabled_progress_is_still_tqdm() -> None:
    steps = range_progress(4, enabled=False)
    assert isinstance(steps, tqdm)
    assert list(steps) == [0, 1, 2, 3]
    steps.set_postfix(lr="1e-3")
    steps.close()


def test_enabled_progress_yields_the_same_indices() -> None:
    steps = range_progress(3, enabled=True, desc="lr_finder", unit="step")
    values = list(steps)
    steps.close()
    assert values == [0, 1, 2]
