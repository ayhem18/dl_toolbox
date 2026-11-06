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

`dl_toolbox` owns the logger contract. A project owns store URIs and run identity. Fit and inspect call `Logger.setup` / `log_*` / `close`. They do not pass URIs into those methods.

```python
from dl_toolbox.dl_engine.loggers.logger import (
    EXISTENCE_CREATE,
    STREAM_FIT,
    flatten_mapping,
)
from dl_toolbox.dl_engine.loggers.mlflow import MlflowLogger

mlflow_dir = storage_dir / "mlflow"
logger = MlflowLogger(
    backend_store_uri=f"sqlite:///{(mlflow_dir / 'mlflow.sqlite').as_posix()}",
    artifact_store_uri=(mlflow_dir / "artifacts").as_posix(),
    experiment_name="pretrain_mim",
    run_name="local_swin_tiny_mim",
    tags={"task": "pretrain_mim"},
)
logger.setup(stream=STREAM_FIT, existence=EXISTENCE_CREATE)
logger.log_params(flatten_mapping(run_config))
logger.log_scalars({"loss/train": 0.12}, step=0)
logger.close()
```

TensorBoard takes `log_dir` and `run_name`. Inspect uses `setup(stream=inspect, existence=require)` on the same `run_name`. W&B is not in this package yet.

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
