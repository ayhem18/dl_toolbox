# dl-toolbox

Shared deep learning code used across projects. A utility moves into this package after it has already worked in a real project and has tests that lock that behavior in.

The installable library is `src/dl_toolbox`. `standards/` and `stories/` stay in the git repository and are not part of the wheel.

## Layout

```
dl_toolbox/                  repo root; pyproject.toml lives here
  src/dl_toolbox/            import dl_toolbox
  tests/                     tests for code that has been promoted
  standards/                 evaluation and engineering contracts
  stories/                   write-ups of methods that transferred
```

`src/` is not required for a GitHub install. Pip and uv install whatever `pyproject.toml` at the repository root declares. The `src` layout is here so running Python from the repo root cannot import an unpackaged checkout, and so `standards/` and `stories/` cannot be picked up as packages.

## Use it from another project

From Git, pin a tag or commit:

```toml
[project]
dependencies = ["dl-toolbox"]

[tool.uv.sources]
dl-toolbox = { git = "https://github.com/ayhem18/dl_toolbox", tag = "v0.1.0" }
```

While developing the toolbox next to a project, point at this checkout:

```toml
[tool.uv.sources]
dl-toolbox = { path = "../../../dl_toolbox", editable = true }
```

Adjust the relative path to wherever the consumer lives under `DL_projects`.

## Parsing and geometry

Copied from `defect_dl` (the fuller set; `seam_embedder` has a subset). Originals stay in those projects until they import from here.

```python
from dl_toolbox.base.parsing.parse_numerical import parse_int, parse_positive_int
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.geometry.contours import contours_bbox
from dl_toolbox.base.geometry.masks import rasterize_contours
from dl_toolbox.base.geometry.bboxes import generate_tile_boxes_xyxy, contours_bbox_xyxy_int
```

`dl_toolbox.base.parsing` is stdlib-only. Geometry needs the extra:

```toml
[project]
dependencies = [
    "dl-toolbox[geometry]",
]
```

## Logging

`dl_toolbox` owns the logger contract and the persistence mechanics. A project owns directories and run identity.

```
dictionary  ->  parser  ->  manager  ->  logger
   (setup)     (validate)  (store)     (training API)
```

- **Toolbox:** `Logger` ABC, `{backend}_parser`, `{backend}_manager`, `{backend}_logger`, and `default_configurations.py` (dicts only).
- **Project:** store paths. W&B `project` / MLflow `experiment` live on the Logger, not on the toolbox project name. Run names stay on the logger.

```python
from dl_toolbox.dl_engine.loggers.mlflow.mlflow_logger import MlflowLogger
from dl_toolbox.dl_engine.loggers.mlflow.mlflow_manager import MlflowManager
from dl_toolbox.dl_engine.loggers.base_logger import flatten_mapping

mlflow_dir = storage_dir / "mlflow"
manager = MlflowManager(
    db_path=mlflow_dir / "mlflow.sqlite",
    artifact_root=mlflow_dir / "artifacts",
    timeout_seconds=30,
)
logger = MlflowLogger(
    manager=manager,
    experiment="pretrain_mim",
    run_name="local_swin_tiny_mim",
    tags={"task": "pretrain_mim"},
)
logger.setup()
logger.log_params(flatten_mapping(run_config))
logger.log_scalars({"loss/train": 0.12}, step=0)
logger.close()
```

TensorBoard and placeholder follow the same split: the project passes `log_dir`, the manager writes there.

```python
from dl_toolbox.dl_engine.loggers.factory import get_logger
from dl_toolbox.dl_engine.loggers.tensorboard.default_configurations import default_tensorboard_setup
from dl_toolbox.dl_engine.loggers.tensorboard.tensorboard_manager import TensorBoardManager

manager = TensorBoardManager(default_tensorboard_setup(log_dir=artifacts_dir / "tensorboard"))
logger = get_logger("tensorboard", manager=manager)
```

Logging backends are separate extras. Install only the one you use:

```toml
[project]
dependencies = [
    "dl-toolbox[dl]",           # seed_everything (numpy, torch)
    "dl-toolbox[geometry]",     # contours, masks, bboxes
    "dl-toolbox[tensorboard]",
    # "dl-toolbox[wandb]",
    # "dl-toolbox[mlflow]",
]
```

`log_scalars(..., step=)` takes `step` as a keyword-only argument.

## What gets added

1. Use the idea in a project until the behavior is clear.
2. Write the result down in `stories/` if the method is worth repeating.
3. If the same code should be called from more than one project, move it into `src/dl_toolbox` with tests.
4. Tag a release before other projects depend on that change.
