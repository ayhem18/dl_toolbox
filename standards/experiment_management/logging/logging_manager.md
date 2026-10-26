# LoggingManager

Project-level store. One sqlite + artifact root, one W&B dir, one TensorBoard parent dir.

Not a run. Not an MLflow experiment. Not a W&B project. Those fields belong on the [Logger](logger.md) via `logger_config`.

```
project dict → Manager parser → LoggingManager
run dict     → Logger parser  → Logger(manager=...)
```

## 1. Input

| | |
|---|---|
| Expected | Store dict for the selected backend. Paths already absolute and under the project root. |
| Comes from | [DefaultRunManager](../project/default_run_manager.md) reading `logging/{backend}.json` after [project files](../project/project_files.md). |
| Conditions | Schema is backend-specific and exact. No `project`, no `experiment`, no `run_name`, no `tags`. |

**Potential issues**

- Treating the manager as “the log folder” (TensorBoard-shaped). MLflow is a database plus an artifact root.
- Opening a SummaryWriter / `wandb.init` on the manager itself: that is the [Logger](logger.md).
- TensorBoard / placeholder: manager holds the **parent** dir. Per-run child is `log_dir / logger_config.run_name`.

## 2. Processing

Parser validates. Manager creates directories, sets tracking URI, pings the backend. It does not open a run.

| Backend | Manager holds |
|---|---|
| MLflow | `backend_store`, `artifact_store` |
| W&B | `dir` |
| TensorBoard | `log_dir` = `storage/tensorboard` |
| placeholder | `log_dir` = `storage/placeholder` |

## 3. Output

| | |
|---|---|
| Expected | A store object. Still no vendor experiment/project/run. |
| Consumers | [Logger](logger.md). One manager per project, reused across runs. |
