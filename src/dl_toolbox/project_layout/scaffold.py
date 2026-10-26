"""Create the opinionated toolbox project tree. The CLI passes cwd as parent."""

from __future__ import annotations

import shutil
from pathlib import Path

from dl_toolbox.base.input_output.json_io import write_json
from dl_toolbox.data_contract.names import assert_generating_process_name
from dl_toolbox.loggers.mlflow.default_configurations import DEFAULT_SQLITE_TIMEOUT_SECONDS
from dl_toolbox.project_layout.discover import assert_package_name, parse_tracking_backend
from dl_toolbox.project_layout.layout import (
    BACKEND_MLFLOW,
    BACKEND_PLACEHOLDER,
    BACKEND_TENSORBOARD,
    BACKEND_WANDB,
    DEFAULT_TRACKING_BACKEND,
    EXPERIMENTS_DIRNAME,
    LOGGING_CONFIGS_DIRNAME,
    MLFLOW_ARTIFACTS_DIRNAME,
    MLFLOW_DIRNAME,
    PLACEHOLDER_DIRNAME,
    PROJECT_LEVEL_CONFIGS_DIRNAME,
    PROJECT_MARKER_FILENAME,
    RESERVED_CHILD_NAMES,
    RUNS_DIRNAME,
    SCRIPTS_DIRNAME,
    STORAGE_DIRNAME,
    TENSORBOARD_DIRNAME,
    TRACKING_BACKEND_FILENAME,
    TRACKING_BACKENDS,
    WANDB_DIRNAME,
)

_README_BY_RELATIVE = {
    f"{STORAGE_DIRNAME}/{RUNS_DIRNAME}": (
        "One directory per training run, named by human_readable_name.\n"
        "The run manager is the only writer under this folder.\n"
    ),
    f"{PROJECT_LEVEL_CONFIGS_DIRNAME}/{LOGGING_CONFIGS_DIRNAME}": (
        "One JSON file per tracking backend. Edit .tracking-backend to switch; "
        "unused configs stay here so you do not recreate them.\n"
    ),
    f"{STORAGE_DIRNAME}/{TENSORBOARD_DIRNAME}": (
        "TensorBoard comparison index. Each child folder is one training run.\n"
        "Open TensorBoard on this directory to compare runs.\n"
    ),
    f"{STORAGE_DIRNAME}/{MLFLOW_DIRNAME}": (
        "Project-level MLflow store. tracking.db and artifacts/ are shared by every run.\n"
    ),
    f"{STORAGE_DIRNAME}/{MLFLOW_DIRNAME}/{MLFLOW_ARTIFACTS_DIRNAME}": (
        "MLflow artifact root for this project. Run-specific files live under MLflow run ids.\n"
    ),
    f"{STORAGE_DIRNAME}/{WANDB_DIRNAME}": (
        "Local W&B directory for this project. The cloud project name is the "
        "toolbox project name in dl_toolbox_project.json.\n"
    ),
    f"{STORAGE_DIRNAME}/{PLACEHOLDER_DIRNAME}": (
        "Filesystem dump used by the placeholder logger. Each child folder is one run.\n"
    ),
    SCRIPTS_DIRNAME: (
        "Train and inspect entry points. They discover the project from this file's path.\n"
    ),
    EXPERIMENTS_DIRNAME: (
        "Run configuration JSON recipes. human_readable_name is not hashed.\n"
    ),
}


def setup_project(
    project_name: str,
    *,
    parent: Path,
    package: str | None = None,
    tracking_backend: str = DEFAULT_TRACKING_BACKEND,
) -> Path:
    """Create ``parent / project_name``. ``parent`` must be an existing absolute directory."""
    name = assert_generating_process_name(project_name)
    backend = parse_tracking_backend(tracking_backend)
    if not parent.is_absolute():
        raise ValueError(f"parent must be an absolute path, got {parent}")
    if not parent.is_dir():
        raise FileNotFoundError(f"parent directory does not exist: {parent}")
    package_name = None if package is None else assert_package_name(package)
    if package_name is not None and package_name in RESERVED_CHILD_NAMES:
        raise ValueError(
            f"package {package_name!r} collides with a reserved project folder"
        )
    root = parent / name
    if root.exists():
        raise FileExistsError(f"project directory already exists: {root}")
    root.mkdir(exist_ok=False)
    try:
        _write_tree(
            root,
            project_name=name,
            package_name=package_name,
            tracking_backend=backend,
        )
    except Exception:
        shutil.rmtree(root)
        raise
    return root


def _write_tree(
    root: Path,
    *,
    project_name: str,
    package_name: str | None,
    tracking_backend: str,
) -> None:
    marker: dict[str, str] = {"name": project_name}
    if package_name is not None:
        marker["package"] = package_name
    write_json(root / PROJECT_MARKER_FILENAME, marker)
    (root / TRACKING_BACKEND_FILENAME).write_text(
        tracking_backend + "\n", encoding="utf-8"
    )

    logging_configs = (
        root / PROJECT_LEVEL_CONFIGS_DIRNAME / LOGGING_CONFIGS_DIRNAME
    )
    logging_configs.mkdir(parents=True)
    backends = tuple(sorted(TRACKING_BACKENDS))
    n_backends = len(backends)
    for i in range(n_backends):
        backend = backends[i]
        write_json(
            logging_configs / f"{backend}.json",
            _backend_config(backend),
        )

    relative_dirs = (
        f"{STORAGE_DIRNAME}/{RUNS_DIRNAME}",
        f"{PROJECT_LEVEL_CONFIGS_DIRNAME}/{LOGGING_CONFIGS_DIRNAME}",
        *_backend_storage_dirs(tracking_backend),
        SCRIPTS_DIRNAME,
        EXPERIMENTS_DIRNAME,
    )
    n_dirs = len(relative_dirs)
    for i in range(n_dirs):
        (root / Path(*relative_dirs[i].split("/"))).mkdir(parents=True, exist_ok=True)

    n_readmes = len(relative_dirs)
    for i in range(n_readmes):
        relative = relative_dirs[i]
        if relative not in _README_BY_RELATIVE:
            continue
        path = root / Path(*relative.split("/")) / "README.md"
        path.write_text(_README_BY_RELATIVE[relative], encoding="utf-8")

    if package_name is not None:
        _write_package(root, package_name=package_name)


def _backend_storage_dirs(tracking_backend: str) -> tuple[str, ...]:
    if tracking_backend == BACKEND_TENSORBOARD:
        return (f"{STORAGE_DIRNAME}/{TENSORBOARD_DIRNAME}",)
    if tracking_backend == BACKEND_PLACEHOLDER:
        return (f"{STORAGE_DIRNAME}/{PLACEHOLDER_DIRNAME}",)
    if tracking_backend == BACKEND_WANDB:
        return (f"{STORAGE_DIRNAME}/{WANDB_DIRNAME}",)
    if tracking_backend == BACKEND_MLFLOW:
        return (
            f"{STORAGE_DIRNAME}/{MLFLOW_DIRNAME}",
            f"{STORAGE_DIRNAME}/{MLFLOW_DIRNAME}/{MLFLOW_ARTIFACTS_DIRNAME}",
        )
    raise ValueError(f"unknown tracking backend {tracking_backend!r}")


def _backend_config(tracking_backend: str) -> dict:
    if tracking_backend == BACKEND_TENSORBOARD:
        return {"log_dir": f"{STORAGE_DIRNAME}/{TENSORBOARD_DIRNAME}"}
    if tracking_backend == BACKEND_PLACEHOLDER:
        return {"log_dir": f"{STORAGE_DIRNAME}/{PLACEHOLDER_DIRNAME}"}
    if tracking_backend == BACKEND_WANDB:
        return {"dir": f"{STORAGE_DIRNAME}/{WANDB_DIRNAME}"}
    if tracking_backend == BACKEND_MLFLOW:
        return {
            "backend_store": {
                "type": "database",
                "database_type": "sqlite",
                "path": f"{STORAGE_DIRNAME}/{MLFLOW_DIRNAME}/tracking.db",
                "timeout_seconds": DEFAULT_SQLITE_TIMEOUT_SECONDS,
            },
            "artifact_store": {
                "type": "filesystem",
                "root": f"{STORAGE_DIRNAME}/{MLFLOW_DIRNAME}/{MLFLOW_ARTIFACTS_DIRNAME}",
            },
        }
    raise ValueError(f"unknown tracking backend {tracking_backend!r}")


def _write_package(root: Path, *, package_name: str) -> None:
    package_root = root / package_name
    src_package = package_root / "src" / package_name
    src_package.mkdir(parents=True)
    (src_package / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "pyproject.toml").write_text(
        _package_pyproject(package_name),
        encoding="utf-8",
    )


def _package_pyproject(package_name: str) -> str:
    return (
        "[project]\n"
        f'name = "{package_name}"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.12"\n'
        "dependencies = [\n"
        '    "dl-toolbox",\n'
        "]\n"
        "\n"
        "[build-system]\n"
        'requires = ["setuptools>=61"]\n'
        'build-backend = "setuptools.build_meta"\n'
        "\n"
        "[tool.setuptools.packages.find]\n"
        'where = ["src"]\n'
    )
