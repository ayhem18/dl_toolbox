"""Locate a toolbox project by its marker file. Never use cwd.

Replacement patterns (File / Directory). The public return is still ``Path``
because the engine is outside this folder.

1. ``find_project_root`` — walk + parent

   Before::

       marker = find_file_in_parents(PROJECT_MARKER_FILENAME, start=start)
       # start may be a file; the walk silently used parent
       return marker.parent

   After::

       origin = File.from_absolute(path=start)  # or Directory MUST_EXIST
       marker = File.from_parent_walk(
           start=origin, basename=PROJECT_MARKER_FILENAME
       )
       return marker.parent().path  # named convert, not Path.parent

2. ``load_project_document`` — absolute + join + read

   Before::

       if not project_root.is_absolute():
           raise ValueError(...)
       path = project_root / PROJECT_MARKER_FILENAME
       read_json(path)

   After::

       root = Directory.from_absolute(
           path=project_root, policy=DirectoryParsePolicy.MUST_EXIST
       )
       marker = File.from_under(
           root=root, relative_path=PROJECT_MARKER_FILENAME
       )
       read_json(marker.path)

3. ``read_tracking_backend`` — same join as (2) for ``.tracking-backend``.
"""

from __future__ import annotations

import keyword
from pathlib import Path
from typing import Any

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.input_output.json_io import read_json
from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import require_exact_keys, require_key
from dl_toolbox.data_contract.names import assert_generating_process_name
from dl_toolbox.experiment_management.default_layout.layout.names import (
    PROJECT_MARKER_FILENAME,
    TRACKING_BACKEND_FILENAME,
    TRACKING_BACKENDS,
)


def _existing_file_or_directory(start: Path) -> File | Directory:
    """Path-edge role claim. Later code should pass File or Directory."""
    if start.is_file():
        return File.from_absolute(path=start)
    return Directory.from_absolute(
        path=start, policy=DirectoryParsePolicy.MUST_EXIST
    )


def find_project_root(*, start: Path) -> Path:
    origin = _existing_file_or_directory(start)
    marker = File.from_parent_walk(
        start=origin, basename=PROJECT_MARKER_FILENAME
    )
    return marker.parent().path


def load_project_document(project_root: Path) -> dict[str, Any]:
    root = Directory.from_absolute(
        path=project_root, policy=DirectoryParsePolicy.MUST_EXIST
    )
    marker = File.from_under(root=root, relative_path=PROJECT_MARKER_FILENAME)
    document = parse_mapping(read_json(marker.path), label=PROJECT_MARKER_FILENAME)
    if "package" in document:
        require_exact_keys(document, ("name", "package"), label=PROJECT_MARKER_FILENAME)
        assert_package_name(document["package"])
    else:
        require_exact_keys(document, ("name",), label=PROJECT_MARKER_FILENAME)
    assert_generating_process_name(require_key(document, "name", label=PROJECT_MARKER_FILENAME))
    return document


def read_tracking_backend(project_root: Path) -> str:
    root = Directory.from_absolute(
        path=project_root, policy=DirectoryParsePolicy.MUST_EXIST
    )
    backend_file = File.from_under(
        root=root, relative_path=TRACKING_BACKEND_FILENAME
    )
    text = backend_file.path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) != 1:
        raise ValueError(
            f"{TRACKING_BACKEND_FILENAME} must contain exactly one line, got {len(lines)}"
        )
    return parse_tracking_backend(lines[0])


def parse_tracking_backend(
    raw: object, *, label: str = TRACKING_BACKEND_FILENAME
) -> str:
    backend = parse_non_empty_str(raw, label=label)
    if backend not in TRACKING_BACKENDS:
        raise ValueError(
            f"{label} must be one of {sorted(TRACKING_BACKENDS)}, got {backend!r}"
        )
    return backend


def assert_package_name(raw: object) -> str:
    name = parse_non_empty_str(raw, label="package")
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"package must be a Python identifier, got {name!r}")
    if name != name.lower() or name.startswith("_"):
        raise ValueError(f"package must be a lowercase identifier, got {name!r}")
    return name
