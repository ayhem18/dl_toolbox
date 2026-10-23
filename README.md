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
dl-toolbox = { git = "https://github.com/<you>/dl_toolbox.git", tag = "v0.1.0" }
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
from dl_toolbox.parsing.parse_numerical import parse_int, parse_positive_int
from dl_toolbox.parsing.parse_str import parse_non_empty_str
from dl_toolbox.geometry.contours import contours_bbox
from dl_toolbox.geometry.masks import rasterize_contours
from dl_toolbox.geometry.bboxes import generate_tile_boxes_xyxy, contours_bbox_xyxy_int
```

`dl_toolbox.parsing` is stdlib-only. Geometry needs the extra:

```toml
[project]
dependencies = [
    "dl-toolbox[geometry]",
]
```

## Currently copied from mypt

```python
from dl_toolbox.seed import seed_everything
from dl_toolbox.loggers.tensorboard_logger import TensorBoardLogger

seed_everything(42)
logger = TensorBoardLogger(log_dir="tb_logs")
logger.log_scalar(0, "loss", 0.12)
```

Logging backends are separate extras. Install only the one you use:

```toml
[project]
dependencies = [
    "dl-toolbox[dl]",           # seed_everything (numpy, torch)
    "dl-toolbox[geometry]",     # contours, masks, bboxes
    "dl-toolbox[tensorboard]",  # TensorBoardLogger
    # "dl-toolbox[wandb]",
    # "dl-toolbox[mlflow]",     # extra is declared; logger comes next
]
```

`get_logger("tensorboard" | "wandb" | "placeholder", log_dir=...)` builds a backend. WandB-only settings such as `project` and `run_name` go in `**kwargs` so every logger keeps the same constructor: `(log_dir, **kwargs)`. Time-series methods take `step` first.

## What gets added

1. Use the idea in a project until the behavior is clear.
2. Write the result down in `stories/` if the method is worth repeating.
3. If the same code should be called from more than one project, move it into `src/dl_toolbox` with tests.
4. Tag a release before other projects depend on that change.
