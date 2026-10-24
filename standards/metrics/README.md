# Metrics

The mental framework is [philosophy.md](philosophy.md) and [philosophy_in_code.md](philosophy_in_code.md). Copy those two files into a project’s documentation so that project stays self-contained.

Executable types are `dl_toolbox.dl_engine.metrics`. They are not a trained product: they do not define a run, a logger, or a monitor key.

The classification helpers in `classification/classification.py` currently violate the grain rule (loss is reduced per batch, then averaged). Do not copy that file as the spec. Reimplementation follows these notes.

## Toolbox

| Module | Job |
|---|---|
| `metric_object` | Named measurement: keys, batch `metric_function`, combiner, named reducers. |
| `accumulator` | Fold one split of one epoch; emit batch/epoch scalars. |
| `contracts` | Tensor mapping keys and checks. |
| `classification` | Task helpers (loss, accuracy) that must return **batch** measurements. |

## Project

A project still owns:

- `prepare_tensors`, `forward_fn`, `loss_fn` (backward), `build_metric_inputs`
- which `MetricSpec`s to log at batch vs epoch
- any metric whose keys are not `output` / `target`

The training loop calls the accumulator. It does not implement combine or reduce.
