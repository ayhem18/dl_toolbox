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

## What gets added

1. Use the idea in a project until the behavior is clear.
2. Write the result down in `stories/` if the method is worth repeating.
3. If the same code should be called from more than one project, move it into `src/dl_toolbox` with tests.
4. Tag a release before other projects depend on that change.
