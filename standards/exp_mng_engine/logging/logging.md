# Logging

Two lifetimes. Do not add a fourth class.

| | LoggingManager | Logger |
|---|---|---|
| Lifetime | project | this run |
| File | `logging/{backend}.json` | `logger_config` |
| Holds | store | vendor run identity |

## 1. Input

Manager: typed store fields (absolute paths). Logger: manager + run fields (`run_name`, and backend extras).

Inspect must not call `Logger.setup` / `start_run` while fit might still be open.

## 2. Processing

```python
class LoggingManager(ABC):
    def publish_directory(self, local_dir: Path, *, dest: str, logger: Logger) -> None: ...
```

Fit opens a live run (`logger.setup`). Inspect publishes a committed folder:

- filesystem backends: copy to `{log_dir}/{run_name}/{dest}`
- MLflow: client `log_artifacts(run_id, ...)` — not `start_run`
- W&B: local copy under `{dir}/{run_name}/{dest}` (no live `wandb.init` from inspect)

`dest` for inspect is `inspection/epoch=N`. Exclusive: refuse if the target exists.

Open-run state (SummaryWriter, wandb run) belongs on the **Logger**, not the manager. Today TensorBoard / placeholder still hold it on the manager; moving that is a follow-up, not required for the core engine to orchestrate.

## 3. Output

Vendor run for fit. Published inspect tree after commit. `RunSession` holds both `logger` and `logging_manager`.
