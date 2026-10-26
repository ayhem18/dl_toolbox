"""Create the opinionated toolbox project tree. The CLI passes cwd as parent."""

from __future__ import annotations

import shutil
from pathlib import Path

from dl_toolbox.data_contract.json_io import write_json
from dl_toolbox.data_contract.names import assert_generating_process_name
from dl_toolbox.loggers.mlflow.default_configurations import DEFAULT_SQLITE_TIMEOUT_SECONDS
from dl_toolbox.project.discover import assert_package_name
from dl_toolbox.project.layout import (
    DEFAULT_TRACKING_BACKEND,
    EXPERIMENTS_DIRNAME,
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
    WANDB_DIRNAME,
)

_README_BY_RELATIVE = {
    f"{STORAGE_DIRNAME}/{RUNS_DIRNAME}": (
        "One directory per training run, named by human_readable_name.\n"
        "The run manager is the only writer under this folder.\n"
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
        "Local W&B directory for this project. The cloud project name is in "
        "project_level_configs/wandb.json.\n"
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
) -> Path:
    """Create ``parent / project_name``. ``parent`` must be an existing absolute directory."""
    name = assert_generating_process_name(project_name)
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
        _write_tree(root, project_name=name, package_name=package_name)
    except Exception:
        shutil.rmtree(root)
        raise
    return root


def _write_tree(root: Path, *, project_name: str, package_name: str | None) -> None:
    marker: dict[str, str] = {"name": project_name}
    if package_name is not None:
        marker["package"] = package_name
    write_json(root / PROJECT_MARKER_FILENAME, marker)
    (root / TRACKING_BACKEND_FILENAME).write_text(
        DEFAULT_TRACKING_BACKEND + "\n", encoding="utf-8"
    )

    configs = root / PROJECT_LEVEL_CONFIGS_DIRNAME
    configs.mkdir()
    write_json(configs / "tensorboard.json", {"log_dir": f"{STORAGE_DIRNAME}/{TENSORBOARD_DIRNAME}"})
    write_json(
        configs / "placeholder.json",
        {"log_dir": f"{STORAGE_DIRNAME}/{PLACEHOLDER_DIRNAME}"},
    )
    write_json(
        configs / "wandb.json",
        {"project": project_name, "dir": f"{STORAGE_DIRNAME}/{WANDB_DIRNAME}"},
    )
    write_json(
        configs / "mlflow.json",
        {
            "experiment": project_name,
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
        },
    )

    relative_dirs = (
        f"{STORAGE_DIRNAME}/{RUNS_DIRNAME}",
        f"{STORAGE_DIRNAME}/{TENSORBOARD_DIRNAME}",
        f"{STORAGE_DIRNAME}/{MLFLOW_DIRNAME}/{MLFLOW_ARTIFACTS_DIRNAME}",
        f"{STORAGE_DIRNAME}/{WANDB_DIRNAME}",
        f"{STORAGE_DIRNAME}/{PLACEHOLDER_DIRNAME}",
        SCRIPTS_DIRNAME,
        EXPERIMENTS_DIRNAME,
    )
    n_dirs = len(relative_dirs)
    for i in range(n_dirs):
        (root / Path(*relative_dirs[i].split("/"))).mkdir(parents=True, exist_ok=True)

    readme_keys = list(_README_BY_RELATIVE)
    n_readmes = len(readme_keys)
    for i in range(n_readmes):
        relative = readme_keys[i]
        path = root / Path(*relative.split("/")) / "README.md"
        path.write_text(_README_BY_RELATIVE[relative], encoding="utf-8")

    if package_name is not None:
        _write_package(root, package_name=package_name, project_name=project_name)


def _write_package(root: Path, *, package_name: str, project_name: str) -> None:
    package_root = root / package_name
    src_package = package_root / "src" / package_name
    src_package.mkdir(parents=True)
    (src_package / "__init__.py").write_text("", encoding="utf-8")
    class_name = _run_manager_class_name(package_name)
    (src_package / "run_manager.py").write_text(
        _run_manager_source(class_name=class_name, project_name=project_name),
        encoding="utf-8",
    )
    (package_root / "pyproject.toml").write_text(
        _package_pyproject(package_name),
        encoding="utf-8",
    )


def _run_manager_class_name(package_name: str) -> str:
    parts = package_name.split("_")
    n = len(parts)
    pieces: list[str] = []
    for i in range(n):
        part = parts[i]
        if not part:
            continue
        pieces.append(part[:1].upper() + part[1:])
    return "".join(pieces) + "RunManager"


def _run_manager_source(*, class_name: str, project_name: str) -> str:
    return (
        "from pathlib import Path\n"
        "\n"
        "from dl_toolbox.project.default_run_manager import DefaultRunManager\n"
        "\n"
        "\n"
        f"class {class_name}(DefaultRunManager):\n"
        f"    PROJECT_NAME = {project_name!r}\n"
        "\n"
        "    def __init__(self, *, start: Path) -> None:\n"
        "        super().__init__(self.PROJECT_NAME, start=start)\n"
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
