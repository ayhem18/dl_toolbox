# Data contract

The mental framework is [philosophy.md](philosophy.md) and [philosophy_in_code.md](philosophy_in_code.md). Copy those two files into a project’s documentation so that project stays self-contained.

Executable helpers are `dl_toolbox.data_contract`. They are stdlib-only. They are not a data product: they do not define a frame, an item, or a lock.

## Toolbox

| Module | Job |
|---|---|
| `repo` | Walk parents from an **absolute** `start` to find a named directory or file. Never `Path.cwd()`. |
| `hashing` | Canonical JSON, full SHA-256 identity, filename prefix. Product call sites use [`HashManager`](../experiment_management/project/hash_manager.md) from `project_level_configs/identity.json`. |
| `json_io` | Atomic JSON-object read/write. |
| `names` | `generating_process_name` is a basename that is not a hash stem. |
| `payload` | Join a relative path under an absolute directory; the file must exist and stay under it. |
| `configs` | Resolve and write collection configurations (`{prefix}.json` plus `{generating_process_name}.json`). |
| `process_records` | Claim a `generating_process_name` and signature hash via `HashManager`, write process configuration last, delete an incomplete run. Payload directories stay the caller’s. |
| `named_freeze` | Delete a finished `generating_process_name` (process pair, collection alias, payload directory the caller passes). Find which name holds a process hash. Not `process()`. |

## Project

A project still owns:

- the data entity, the data collection, the generating process, the oracle, and the named recipe
- `Collection.configs_dir()` and, if file-backed, `Collection.artifacts_dir(generating_process_name)`
- member indexes, payload load methods, `related_collections()`
- dump layout and iteration folder names

The process calls toolbox helpers. Consumers `Collection.load`. They do not import a strategy to find bytes.
