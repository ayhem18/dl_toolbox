"""Locate a toolbox project by its marker file. Never use cwd."""

from __future__ import annotations

import keyword
from pathlib import Path
from typing import Any

from dl_toolbox.base.file_management.parent_walk import find_file_in_parents
from dl_toolbox.base.input_output.json_io import read_json
from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import require_exact_keys, require_key
from dl_toolbox.data_contract.names import assert_generating_process_name
from dl_toolbox.project.layout import (
    PROJECT_MARKER_FILENAME,
    TRACKING_BACKEND_FILENAME,
    TRACKING_BACKENDS,
)


def find_project_root(*, start: Path) -> Path:
    marker = find_file_in_parents(PROJECT_MARKER_FILENAME, start=start)
    return marker.parent


def load_project_document(project_root: Path) -> dict[str, Any]:
    if not project_root.is_absolute():
        raise ValueError(f"project_root must be an absolute path, got {project_root}")
    path = project_root / PROJECT_MARKER_FILENAME
    document = parse_mapping(read_json(path), label=PROJECT_MARKER_FILENAME)
    if "package" in document:
        require_exact_keys(document, ("name", "package"), label=PROJECT_MARKER_FILENAME)
        assert_package_name(document["package"])
    else:
        require_exact_keys(document, ("name",), label=PROJECT_MARKER_FILENAME)
    assert_generating_process_name(require_key(document, "name", label=PROJECT_MARKER_FILENAME))
    return document


def read_tracking_backend(project_root: Path) -> str:
    if not project_root.is_absolute():
        raise ValueError(f"project_root must be an absolute path, got {project_root}")
    path = project_root / TRACKING_BACKEND_FILENAME
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) != 1:
        raise ValueError(
            f"{TRACKING_BACKEND_FILENAME} must contain exactly one line, got {len(lines)}"
        )
    return parse_tracking_backend(lines[0])


def parse_tracking_backend(raw: object) -> str:
    backend = parse_non_empty_str(raw, label=TRACKING_BACKEND_FILENAME)
    if backend not in TRACKING_BACKENDS:
        raise ValueError(
            f"{TRACKING_BACKEND_FILENAME} must be one of {sorted(TRACKING_BACKENDS)}, "
            f"got {backend!r}"
        )
    return backend


def assert_package_name(raw: object) -> str:
    name = parse_non_empty_str(raw, label="package")
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"package must be a Python identifier, got {name!r}")
    if name != name.lower() or name.startswith("_"):
        raise ValueError(f"package must be a lowercase identifier, got {name!r}")
    return name
