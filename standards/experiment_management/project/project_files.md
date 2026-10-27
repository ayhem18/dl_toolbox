# Project files

Project-level. Not in the run recipe. Switching TensorBoard to MLflow must not change a run hash: the hash is the two sub-configs, and these files are not among them.

```
project_root/
  dl_toolbox_project.json
  .tracking-backend
  project_level_configs/logging/{backend}.json
  project_level_configs/identity.json
  storage/runs/{human_readable, hashes}/
  storage/run_artifacts/{name}/
  storage/{tensorboard|mlflow|wandb|placeholder}/
  experiments/
  scripts/
```

## 1. Input

| | |
|---|---|
| Expected | Marker file, backend selector, one store JSON per backend, `identity.json` (`prefix_len`, `human_readable_dirname`, `hashes_dirname`). Paths in logging JSON are relative to the project root. |
| Comes from | `dl-toolbox setup-project`. Runtime walks from `__file__` to the marker. Never `cwd`. |
| Conditions | `dl_toolbox_project.json.name` matches `DefaultRunManager(project_name=...)`. `.tracking-backend` is one of `tensorboard`, `mlflow`, `wandb`, `placeholder`. |

**Potential issues**

- Store paths that escape the project root.
- `wandb.project` / `mlflow.experiment` / `run_name` / `tags` in logging JSON. Those are [Logger](../logging/logger.md) fields, in `logger_config`.
- Walking for a bare `storage/` folder: picks a sibling project.
- Putting `prefix_len` or index folder names in the run recipe. Those knobs are [`HashManager`](hash_manager.md).
- Changing `identity.json` after the first name/hash write. Stems already on disk use the old prefix.

## 2. Processing

```mermaid
flowchart TD
  start["start = Path(__file__)"] --> marker["find dl_toolbox_project.json"]
  marker --> name["project_name"]
  marker --> backend["read .tracking-backend"]
  backend --> json["read logging/{backend}.json"]
  json --> resolve["resolve paths under project root"]
  resolve --> lman["LoggingManager"]
  marker --> identity["read identity.json"]
  identity --> hman["HashManager"]
```

Only the selected backend gets a store folder. Config files for every backend stay on disk. `setup-project` also writes `identity.json` (defaults: prefix 12, folders `human_readable` / `hashes`).

## 3. Output

| | |
|---|---|
| Expected | `project_name`, selected `LoggingManager`, `HashManager`. |
| Consumers | [DefaultRunManager](default_run_manager.md) holds the paths and constructs the managers. [AbstractRunManager](abstract_run_manager.md) consumes them. [Logger](../logging/logger.md) takes the `LoggingManager`. [run identity](../recipe/run_identity.md) uses `HashManager` on `storage/runs` and payload under `storage/run_artifacts/{name}`. |
