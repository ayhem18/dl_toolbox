"""JSON object: parse, dump, flatten, File read, Directory write."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
    DirectorySpec,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.file_management.join import resolve_under_project


class JsonTextForm:
    DOCUMENT = "document"
    CANONICAL = "canonical"
    ASCII_PRETTY = "ascii_pretty"


def parse_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text)!r}")
    value = json.loads(text)
    return _as_json_object(value)


def dump(payload: object, *, form: str) -> str:
    payload = _as_json_object(payload)
    if form == JsonTextForm.DOCUMENT:
        return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if form == JsonTextForm.CANONICAL:
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
    if form == JsonTextForm.ASCII_PRETTY:
        return json.dumps(payload, indent=2, ensure_ascii=True)
    raise ValueError(
        "form must be "
        f"{JsonTextForm.DOCUMENT!r}, {JsonTextForm.CANONICAL!r}, or "
        f"{JsonTextForm.ASCII_PRETTY!r}, got {form!r}"
    )


def flatten(payload: object) -> dict[str, str]:
    payload = _as_json_object(payload)
    return _flatten(payload, prefix="")


def read(*, file: File) -> dict[str, Any]:
    if not isinstance(file, File):
        raise TypeError(f"file must be File, got {type(file)!r}")
    return parse_object(file.path.read_text(encoding="utf-8"))


def write(
    *,
    directory: Directory,
    name: str,
    payload: object,
    form: str,
) -> File:
    if not isinstance(directory, Directory):
        raise TypeError(f"directory must be Directory, got {type(directory)!r}")
    if directory.spec != DirectorySpec.EXISTS:
        raise ValueError(f"directory must exist: {directory.path}")
    text = dump(payload, form=form)
    path = resolve_under_project(directory.path, name, label="name")
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    return File.from_absolute(path=path)


def _as_json_object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object, got {type(value).__name__}")
    keys = list(value)
    n = len(keys)
    for i in range(n):
        key = keys[i]
        if not isinstance(key, str) or not key:
            raise ValueError(f"JSON object key must be a non-empty str, got {key!r}")
    return value


def _flatten(payload: dict, *, prefix: str) -> dict[str, str]:
    keys = list(payload)
    n = len(keys)
    params: dict[str, str] = {}
    for i in range(n):
        key = keys[i]
        if not isinstance(key, str) or not key:
            raise ValueError(f"JSON object key must be a non-empty str, got {key!r}")
        name = prefix + key
        value = payload[key]
        if isinstance(value, dict):
            nested = _flatten(value, prefix=name + ".")
            nested_keys = list(nested)
            n_nested = len(nested_keys)
            for j in range(n_nested):
                nested_key = nested_keys[j]
                params[nested_key] = nested[nested_key]
            continue
        if isinstance(value, (list, tuple)):
            params[name] = json.dumps(list(value), ensure_ascii=True)
            continue
        if isinstance(value, bool):
            params[name] = "true" if value else "false"
            continue
        if value is None:
            params[name] = "null"
            continue
        params[name] = str(value)
    return params


def read_json(path: Path) -> dict[str, Any]:
    """Path-holding call sites parse File here. Not a sixth algorithm."""
    return read(file=File.from_absolute(path=path))


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Path-holding call sites parse Directory here. form=document. Missing parent is CREATE."""
    directory = Directory.from_absolute(
        path=path.parent, policy=DirectoryParsePolicy.CREATE
    )
    written = write(
        directory=directory,
        name=path.name,
        payload=dict(payload),
        form=JsonTextForm.DOCUMENT,
    )
    return written.path
