# RunSession

Shared handle for fit and inspect. Assembled by `ExpEngine`.

## 1. Input

Every field required. Keyword-only.

| Field | Role |
|---|---|
| `identity` | `RunIdentity` (`run_dir`, name, hash) |
| `fit_config` | trainer + monitor + early stopping |
| `checkpointer` | run façade |
| `logger` | run logger. Inspect calls `setup(stream="inspect")` in its own process. |
| `logging_manager` | project store |
| `seed` | the integer that was applied |
| `experiment` | `ExperimentBuild` |

## 2. Processing

None. Value object. `run_dir` / `human_readable_name` / `hash` are properties of `identity`.

## 3. Output

`train` unpacks experiment + fit_config + checkpointer + logger. `inspect` unpacks checkpointer + logger + experiment (for the project’s callable).
