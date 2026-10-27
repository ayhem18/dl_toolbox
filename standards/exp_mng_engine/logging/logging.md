# Logging

Two lifetimes. Do not add a fourth class.

| | LoggingManager | Logger |
|---|---|---|
| Lifetime | project | this run / this process |
| File | `logging/{backend}.json` | `logger_config` |
| Holds | store | open-run state (`SummaryWriter`, MLflow run, …) and every `log_*` |

## 1. Input

Manager: typed store fields (absolute paths). Logger: manager + run fields (`run_name`, and backend extras).

Fit calls `logger.setup()` (default `stream="fit"`). Inspect calls `logger.setup(stream="inspect")` in the inspect process, then `log_image` / `log_table` / `log_text` with `step=epoch`, then `close()`. Inspect must not share the fit process’s writer.

## 2. Processing

```python
class LoggingManager(ABC):
    def prepare(self) -> None: ...

class Logger(ABC):
    def setup(self, *, stream: str = "fit") -> None: ...
    def log_scalars(self, values, *, step: int) -> None: ...
    def log_image(self, tag, image, *, step: int) -> None: ...
    def log_table(self, tag, rows, *, step: int) -> None: ...
    def log_text(self, tag, text, *, step: int) -> None: ...
    def close(self) -> None: ...
```

`LoggingManager` creates directories and vendor clients/writers. It does not log. TensorBoard `create_writer(run_name, filename_suffix=)` is store setup; inspect uses `filename_suffix=".inspect"`. MLflow `client()` is store setup; inspect loads the run id the fit logger recorded.

There is no `log_artifact` and no `publish_directory`. Copying files into the log dir is not logging.

TensorBoard `--reload_multifile=true` is required while fit and inspect both write event files under `{log_dir}/{run_name}/`.

MLflow inspect `setup(stream="inspect")` attaches to the fit run through `MlflowClient` and the recorded run id. It does not `start_run` / `end_run`. Fit `close()` is the only `set_terminated`. W&B attach-by-run-id is a later Logger change.

## 3. Output

Vendor event stream for fit scalars. Inspect images/tables at `step=epoch` on the same TensorBoard or MLflow run. Disk payload stays under `inspection_runs/epoch=N`.
