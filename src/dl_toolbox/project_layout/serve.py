"""Serve the vendor tracking UI from a ProjectLayout.

run_mlflow and run_tensorboard attach with from_marker, then spawn the
vendor CLI on the store paths the layout already read.

"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
    DirectorySpec,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.parsing.parse_numerical import parse_positive_int
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.project_layout.names import (
    MLFLOW_RUN_FILENAME,
    TENSORBOARD_RUN_FILENAME,
)
from dl_toolbox.project_layout.project_layout import ProjectLayout, TrackingBackend

MLFLOW_UI_HOST = "127.0.0.1"
MLFLOW_UI_PORT = 5000
TENSORBOARD_UI_HOST = "127.0.0.1"
TENSORBOARD_UI_PORT = 6006


def mlflow_run_script() -> str:
    return (
        '"""Open MLflow UI on this project\'s tracking store.\n'
        "\n"
        f"Named {MLFLOW_RUN_FILENAME} so scripts/ does not shadow the mlflow "
        "package.\n"
        "\n"
        "Store paths come from ProjectLayout (logging JSON).\n"
        "\n"
        f"  python {MLFLOW_RUN_FILENAME}\n"
        f"  python {MLFLOW_RUN_FILENAME} --port 5001\n"
        '"""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "from dl_toolbox.base.file_management.file import File\n"
        "from dl_toolbox.project_layout.serve import run_mlflow\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(\n"
        "        run_mlflow(\n"
        "            start=File.from_absolute(path=Path(__file__).resolve()),\n"
        "            argv=sys.argv[1:],\n"
        "        )\n"
        "    )\n"
    )


def tensorboard_run_script() -> str:
    return (
        '"""Open TensorBoard on this project\'s log directory.\n'
        "\n"
        f"Named {TENSORBOARD_RUN_FILENAME} so scripts/ does not shadow the "
        "tensorboard package.\n"
        "\n"
        "Store paths come from ProjectLayout (logging JSON).\n"
        "\n"
        f"  python {TENSORBOARD_RUN_FILENAME}\n"
        f"  python {TENSORBOARD_RUN_FILENAME} --port 6007\n"
        '"""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "from dl_toolbox.base.file_management.file import File\n"
        "from dl_toolbox.project_layout.serve import run_tensorboard\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(\n"
        "        run_tensorboard(\n"
        "            start=File.from_absolute(path=Path(__file__).resolve()),\n"
        "            argv=sys.argv[1:],\n"
        "        )\n"
        "    )\n"
    )


def run_mlflow(*, start: File, argv: Sequence[str]) -> int:
    if not isinstance(start, File):
        raise TypeError(f"start must be File, got {type(start)!r}")
    host, port = _parse_ui_argv(
        argv=argv,
        description="Serve this project's MLflow tracking store in the UI.",
        default_host=MLFLOW_UI_HOST,
        default_port=MLFLOW_UI_PORT,
    )
    layout = ProjectLayout.from_marker(
        start=start, tracking_backend=TrackingBackend.MLFLOW
    )
    return serve_mlflow(layout=layout, host=host, port=port)


def run_tensorboard(*, start: File, argv: Sequence[str]) -> int:
    if not isinstance(start, File):
        raise TypeError(f"start must be File, got {type(start)!r}")
    host, port = _parse_ui_argv(
        argv=argv,
        description="Serve this project's TensorBoard log directory.",
        default_host=TENSORBOARD_UI_HOST,
        default_port=TENSORBOARD_UI_PORT,
    )
    layout = ProjectLayout.from_marker(
        start=start, tracking_backend=TrackingBackend.TENSORBOARD
    )
    return serve_tensorboard(layout=layout, host=host, port=port)


def serve_mlflow(*, layout: ProjectLayout, host: str, port: int) -> int:
    if not isinstance(layout, ProjectLayout):
        raise TypeError(f"layout must be ProjectLayout, got {type(layout)!r}")
    bind_host = parse_non_empty_str(host, label="host")
    bind_port = parse_positive_int(port, label="port")
    db_path = layout.mlflow_db_path()
    Directory.from_absolute(
        path=db_path.parent, policy=DirectoryParsePolicy.CREATE
    )
    artifact_root = _ensure_directory(layout.mlflow_artifact_root())
    uri = f"sqlite:///{db_path.resolve().as_posix()}"
    env = dict(os.environ)
    env["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    sys.stdout.write(f"MLflow UI: http://{bind_host}:{bind_port}/\n")
    sys.stdout.flush()
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            uri,
            "--default-artifact-root",
            str(artifact_root.path.resolve()),
            "--host",
            bind_host,
            "--port",
            str(bind_port),
        ],
        cwd=str(layout.root.path),
        env=env,
    )


def serve_tensorboard(*, layout: ProjectLayout, host: str, port: int) -> int:
    if not isinstance(layout, ProjectLayout):
        raise TypeError(f"layout must be ProjectLayout, got {type(layout)!r}")
    bind_host = parse_non_empty_str(host, label="host")
    bind_port = parse_positive_int(port, label="port")
    log_dir = _ensure_directory(layout.tensorboard_log_dir())
    sys.stdout.write(f"TensorBoard: http://{bind_host}:{bind_port}/\n")
    sys.stdout.flush()
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "tensorboard",
            "--logdir",
            str(log_dir.path),
            "--reload_multifile",
            "true",
            "--host",
            bind_host,
            "--port",
            str(bind_port),
        ],
        cwd=str(layout.root.path),
    )


def _ensure_directory(directory: Directory) -> Directory:
    if directory.spec == DirectorySpec.EXISTS:
        return directory
    return Directory.from_absolute(
        path=directory.path, policy=DirectoryParsePolicy.CREATE
    )


def _parse_ui_argv(
    *,
    argv: Sequence[str],
    description: str,
    default_host: str,
    default_port: int,
) -> tuple[str, int]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--host",
        default=default_host,
        help=f"Bind address (default {default_host}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help=f"Bind port (default {default_port}).",
    )
    args = parser.parse_args(list(argv))
    host = parse_non_empty_str(args.host, label="host")
    port = parse_positive_int(args.port, label="port")
    return host, port
