from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.input_output.json_io import (
    JsonTextForm,
    dump,
    flatten,
    parse_object,
    read,
    write,
)

from tests.base.base_support import N_TRIALS, random_generating_process_name, random_json_object


def _existing_directory(path: Path) -> Directory:
    return Directory.from_absolute(path=path, policy=DirectoryParsePolicy.CREATE)


def test_written_object_round_trips_and_leaves_no_tmp(
    tmp_path: Path, rng: random.Random
) -> None:
    for i in range(N_TRIALS):
        directory = _existing_directory(tmp_path / f"nested_{i}")
        payload = random_json_object(rng)
        payload["generating_process_name"] = random_generating_process_name(rng)
        written = write(
            directory=directory,
            name="config.json",
            payload=payload,
            form=JsonTextForm.DOCUMENT,
        )
        assert read(file=written) == payload
        assert not written.path.with_name(written.path.name + ".tmp").exists()


def test_cyrillic_is_written_as_utf8_glyphs(tmp_path: Path) -> None:
    directory = _existing_directory(tmp_path)
    relative = "Сортированные снимки/Мерник/04 TR-SFCC-ZEPP-DRT-11386.DCM"
    written = write(
        directory=directory,
        name="members.json",
        payload={"dcm_file_path": relative, "members": {relative: {}}},
        form=JsonTextForm.DOCUMENT,
    )
    text = written.path.read_text(encoding="utf-8")
    assert "Сортированные снимки" in text
    assert "\\u0421" not in text
    assert read(file=written)["dcm_file_path"] == relative


def test_non_objects_are_rejected(tmp_path: Path, rng: random.Random) -> None:
    samples = (
        [1, 2, 3],
        "not-an-object",
        12,
        True,
        None,
    )
    directory = _existing_directory(tmp_path)
    for i, sample in enumerate(samples):
        path = directory.path / f"bad_{i}.json"
        path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            read(file=File.from_absolute(path=path))
    for i in range(N_TRIALS):
        path = directory.path / f"rand_bad_{i}.json"
        path.write_text(json.dumps(rng.randint(0, 1000)) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            parse_object(path.read_text(encoding="utf-8"))


def test_dump_forms() -> None:
    payload = {"b": 1, "a": "я"}
    document = dump(payload, form=JsonTextForm.DOCUMENT)
    assert document.endswith("\n")
    assert "я" in document
    canonical = dump(payload, form=JsonTextForm.CANONICAL)
    assert canonical == '{"a":"\\u044f","b":1}'
    pretty = dump(payload, form=JsonTextForm.ASCII_PRETTY)
    assert "\\u044f" in pretty
    assert pretty.startswith("{")


def test_flatten_nested_bools_lists_and_none() -> None:
    params = flatten(
        {
            "name": "dummy_conv",
            "model": {"name": "unet_resnet34", "params": {"pretrained": False}},
            "runtime": {"devices": 1, "workers": None},
            "tags": ["a", "b"],
        }
    )
    assert params["name"] == "dummy_conv"
    assert params["model.name"] == "unet_resnet34"
    assert params["model.params.pretrained"] == "false"
    assert params["runtime.devices"] == "1"
    assert params["runtime.workers"] == "null"
    assert params["tags"] == '["a", "b"]'
    assert flatten({"a": 1}) == flatten({"a": "1"})


def test_write_does_not_mkdir(tmp_path: Path) -> None:
    missing = Directory.from_absolute(
        path=tmp_path / "absent", policy=DirectoryParsePolicy.ALLOW_MISSING
    )
    with pytest.raises(ValueError, match="must exist"):
        write(
            directory=missing,
            name="config.json",
            payload={"k": 1},
            form=JsonTextForm.DOCUMENT,
        )
