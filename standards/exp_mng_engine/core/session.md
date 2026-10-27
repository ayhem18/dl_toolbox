# RunSession

Shared handle for fit and inspect. Assembled by `ExpEngine`. Replaces `ExpSignatureObj` on the new path.

## 1. Input

Every field required. Keyword-only.

| Field | Role |
|---|---|
| `identity` | `RunIdentity` (`run_dir`, name, hash) |
| `fit_config` | trainer + monitor + early stopping |
| `checkpointer` | run façade |
| `logger` | run logger (not set up on inspect) |
| `logging_manager` | inspect publish |
| `seed` | the integer that was applied |
| `experiment` | `ExperimentBuild` |

## 2. Processing

None. Value object. `run_dir` / `human_readable_name` / `hash` are properties of `identity`.

## 3. Output

`train` unpacks experiment + fit_config + checkpointer + logger. `inspect` unpacks checkpointer + logging_manager + experiment (for the project’s callable).
