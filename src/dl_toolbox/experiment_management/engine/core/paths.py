"""Injected locations. Core does not walk the project tree."""

from __future__ import annotations

from pathlib import Path

from dl_toolbox.base.parsing.parse_path import parse_absolute_path


class EnginePaths:
    """The three directories the engine touches. Layout is the caller's job.

    Identity JSON and run files are kept apart on purpose. Reclaiming a name
    deletes the (name, hash) pair under ``run_identities_root`` and the folder
    under ``run_artifacts_root``. Logger stores (TensorBoard, MLflow, …) are
    not these paths.
    """

    def __init__(
        self,
        *,
        project_root: Path,
        run_identities_root: Path,
        run_artifacts_root: Path,
    ) -> None:
        """
        project_root:
            Project directory. ExperimentConfigManager.build resolves relative
            paths in experiment_config (for example data_root) against this.

        run_identities_root:
            Directory that stores claimed experiment configurations as
            (name, hash) pairs. IdentityManager uses it to decide whether a
            pair already exists; claim writes a new pair here. Default layout:
            storage/runs/, with human_readable/{name}.json and
            hashes/{stem}.json. The identity lock (identity.lock) is also under
            this directory.

        run_artifacts_root:
            Parent of per-run working directories. One child folder per claimed
            name. That child is run_artifacts_dir(name): checkpoints,
            catalog.json, inspection_staging, inspection_runs. Default layout:
            storage/run_artifacts/{name}.
        """
        self.project_root = parse_absolute_path(str(project_root), label="project_root")
        self.run_identities_root = parse_absolute_path(
            str(run_identities_root), label="run_identities_root"
        )
        self.run_artifacts_root = parse_absolute_path(
            str(run_artifacts_root), label="run_artifacts_root"
        )

    def run_artifacts_dir(self, name: str) -> Path:
        """Working directory for one run: run_artifacts_root / name.

        Claim mkdir's it. Open requires it to exist. Checkpointer.bind uses it
        as run_dir. Inspect writes inspection_runs/ under it.
        """
        return self.run_artifacts_root / name
