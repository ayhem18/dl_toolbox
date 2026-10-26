from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from dl_toolbox.base.input_output.json_io import read_json, write_json

from _support import N_TRIALS, random_generating_process_name, random_json_object


def test_written_object_round_trips_and_leaves_no_tmp(
    tmp_path: Path, rng: random.Random
) -> None:
    for i in range(N_TRIALS):
        path = tmp_path / f"nested_{i}" / "config.json"
        payload = random_json_object(rng)
        payload["generating_process_name"] = random_generating_process_name(rng)
        write_json(path, payload)
        assert read_json(path) == payload
        assert not path.with_name(path.name + ".tmp").exists()


def test_non_objects_are_rejected(tmp_path: Path, rng: random.Random) -> None:
    samples = (
        [1, 2, 3],
        "not-an-object",
        12,
        True,
        None,
    )
    for i, sample in enumerate(samples):
        path = tmp_path / f"bad_{i}.json"
        path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            read_json(path)
    for i in range(N_TRIALS):
        path = tmp_path / f"rand_bad_{i}.json"
        path.write_text(json.dumps(rng.randint(0, 1000)) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            read_json(path)
