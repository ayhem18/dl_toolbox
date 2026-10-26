# Fit

Light Lightning loop. Does not own checkpoint policy or store paths. Does not know inspect.

## 1. Input

| | |
|---|---|
| Expected | Fit-user args from [`build_experiment`](../signature/build_experiment.md). `FitConfig` (trainer, bound monitor, logger, early stopping — **no checkpoint fields**). [`CheckpointManager`](../checkpoint/checkpoint_manager.md). |
| Comes from | [`run_experiment`](run_experiment.md) unpacking [`ExpSignatureObj`](../signature/exp_signature_obj.md). |
| Conditions | `max_epochs >= 1`. Monitor key must appear in val epoch scalars. Logger is a toolbox `Logger`. Seed already applied. |

**Potential issues**

- Building `ModelCheckpoint(...)` from `FitConfig.checkpoint` (current code smell). Register `checkpoint_manager.as_callback()` instead.
- Reading inspect config. Fit does not.
- Lightning `Trainer(logger=...)`. Toolbox logger is `epoch_logger`; Trainer stays `logger=False`.
- Taking `seed` as a fit argument. [`seed_config`](../training/seed.md) ran in `manage_run`.
- `FitConfig.early_stopping is None`. Off is `EarlyStoppingConfig.NONE`.

## 2. Processing

```mermaid
flowchart TD
  start["on_fit_start: logger.setup, log params"] --> train["train epoch: accumulate, log scalars"]
  train --> val["val epoch: accumulate, log scalars, require monitor"]
  val --> cb["CheckpointManager callback writes epoch=N.ckpt + catalog"]
  cb --> end["on_fit_end / finally: logger.close"]
```

`L.Trainer` gets `max_epochs`, `accelerator`, `devices`, `precision`, `enable_progress_bar` from [`trainer_config`](../training/trainer.md). Early stopping is always on `FitConfig`. Fit registers the Lightning callback only when `name` is `PATIENCE`. Same monitor as the checkpoint manager, bound from the experiment.

## 3. Output

| | |
|---|---|
| Expected | Trained weights under `run_dir/checkpoints/`. `catalog.json`. Logged scalars and the recipe as an artifact. Logger closed. |
| Consumers | [inspect](inspect.md) reads the catalog. Humans read the vendor UI. Resume is later. |
