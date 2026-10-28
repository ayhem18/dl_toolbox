# DefaultExpEngine

Same core. Specifies components from the default tree.

Code: `experiment_management/engine/default`. Layout and parsers: `experiment_management/default_layout` (`layout/` + `parsing/`).

## 1. Input

`project_name`, `start` path, `ExperimentConfigManager`, optional `tracking_backend`. Walks to `dl_toolbox_project.json`. `.tracking-backend` is the project default. `tracking_backend=` overrides it for this process and does not rewrite the file.

## 2. Processing

```mermaid
flowchart LR
  start["start path"] --> disc["default_layout.layout.discover"]
  disc --> files["identity.json / checkpoint.json / logging/backend.json"]
  files --> par["default_layout.parsing"]
  par --> mgrs["HashManager / LoggingManager / CheckpointManager"]
  mgrs --> core["ExpEngine"]
```

Two jobs, in `default_layout`:

1. **layout/** — names, discover, scaffold (`setup_project` writes `identity.json`, `checkpoint.json`, `logging/{backend}.json`).
2. **parsing/** — those files → constructor fields. Does not construct managers.

`DefaultExpEngine` constructs Fold-1 objects and calls `ExpEngine.__init__`.

```python
from dl_toolbox.experiment_management.engine.default.engine import DefaultExpEngine

DefaultExpEngine(
    project_name,
    start=Path(__file__),
    experiment=MnistVitExperimentConfigManager(),
    tracking_backend="mlflow",  # optional; omit to use .tracking-backend
)
```

The project does not subclass this class to teach it a loss.

Default tree (project root):

| Folder | Job |
|---|---|
| `scripts/` | Durable CLI (`train`, `inspect`, vendor UI) |
| `experiments/` | Disposable experimental scripts |
| `run_recipes/` | Run configuration JSON. `human_readable_name` is not hashed |

`checkpoint.json`:

```json
{ "lock_timeout_seconds": 600.0 }
```

## 3. Output

An `ExpEngine` ready for `train` / `inspect`.
