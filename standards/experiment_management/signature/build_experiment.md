# build_experiment

Abstract method on `DefaultRunManager`. The project implements the body. Toolbox specifies the return type because [`manage_run`](../project/default_run_manager.md) stores it on [`ExpSignatureObj`](exp_signature_obj.md).

One class (the project run-manager subclass). One function. Data contract, loss, metrics, optimizer all live here. Scripts do not wire fit.

## 1. Input

| | |
|---|---|
| Expected | `experiment_config`: a mapping. Schema is the caller’s. One half of the **configuration**. |
| Comes from | [run recipe](../recipe/run_recipe.md). `manage_run` passes it in. Toolbox does not parse the keys. |
| Conditions | Must be an object. Extra/missing keys are the project parser’s job. The name is not an argument. |

**Potential issues**

- Parsing `experiment_config` inside `dl_toolbox`. The toolbox cannot know the task.
- Leaving loss / loaders / optimizer in `scripts/train.py`. Then inspect cannot rebuild the net from the same function.
- Returning objects the fit signature does not name.
- Putting the monitor in `dl_toolbox_config`. The name is a `MetricSpec` this function creates. The project parser checks it against `val_metrics`.

## 2. Processing

The project body: load data (data contract if the task has one), build the net, optimizer, `loss_fn`, `prepare_tensors_fn`, metrics. Compose `monitor` (`key` + `mode`) from the project’s own monitor JSON and the `val_metrics` list.

Toolbox does not specify those steps. It specifies the **shape** of the result. Seed is already applied when this runs.

## 3. Output

Fit-user args. Everything `fit_classification` needs **except** what the toolbox already built.

| Field | Role |
|---|---|
| `loss_fn` | Backward. Not the loss metric. |
| `prepare_tensors_fn` | Raw batch → tensor mapping. |
| `optimizer` | Bound to the model. |
| `model` | `nn.Module`. |
| `train_loader`, `val_loader` | Data. |
| `forward_fn` | Net + batch → output. |
| `build_metric_inputs_fn` | Output + batch → metric tensors. |
| `train_metrics`, `val_metrics` | `MetricSpec` lists. See [metrics](../../metrics/philosophy.md). |
| `monitor` | `MonitorConfig`: composed log `key` (`metric_log_name`) + `mode`. Project-owned JSON; toolbox-shaped return. |

| | |
|---|---|
| Consumers | `manage_run` puts this on `ExpSignatureObj.experiment` and binds `monitor` onto `FitConfig` and `CheckpointManager`. [`run_experiment`](../loops/run_experiment.md) unpacks it. Inspect reuses the builder for the net, then loads weights from the catalog. |
