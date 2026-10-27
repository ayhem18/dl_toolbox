# run_experiment

Default entry point. Consumes [`ExpSignatureObj`](../signature/exp_signature_obj.md). Calls [fit](fit.md). Project scripts become two lines.

```python
signature = DefaultRunManager(
    project_name="mnist_vit",
    start=Path(__file__),
    experiment_config_manager=MnistVitExperimentConfigManager(),
).manage_run(recipe)
run_experiment(signature)
```

## 1. Input

| | |
|---|---|
| Expected | One `ExpSignatureObj`. |
| Comes from | `AbstractRunManager.manage_run` (on `DefaultRunManager`) after the project’s `ExperimentConfigManager` has built. |
| Conditions | `experiment` has the fit-user args including `monitor`. `fit_config`, `checkpoint_manager`, and `logger` are present. Seed already applied. |

**Potential issues**

- Calling `fit_classification` from the script and bypassing this function. Then inspect and fit drift.
- Inspect calling `run_experiment`. Inspect is a different entry.

## 2. Processing

Unpack. Do not parse JSON. Do not build the model.

```python
def run_experiment(signature: ExpSignatureObj):
    e = signature.experiment
    return fit_classification(
        e.model, e.train_loader, e.val_loader, e.optimizer,
        prepare_tensors_fn=e.prepare_tensors_fn,
        forward_fn=e.forward_fn,
        loss_fn=e.loss_fn,
        build_metric_inputs_fn=e.build_metric_inputs_fn,
        train_metrics=e.train_metrics,
        val_metrics=e.val_metrics,
        config=signature.fit_config,
        checkpoint_manager=signature.checkpoint_manager,
        logger=signature.logger,
    )
```

## 3. Output

| | |
|---|---|
| Expected | Whatever [fit](fit.md) returns (Lightning `Trainer`). Side effects: checkpoints, catalog, logs. |
| Consumers | The train script. Humans looking at the vendor UI and `run_dir`. |
