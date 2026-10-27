# IdentityManager

Project-level exclusive `(name, hash)` pair. Default implementation is `HashManager`. The digest is SHA-256 of `{dl_toolbox_config, experiment_config}`. The name is a label. It is not hashed.

## 1. Input

| | |
|---|---|
| Expected | `run_identities_root` (injected by `EnginePaths`). Basename `name`. Full SHA-256. |
| Comes from | Default: `identity.json` → prefix length and folder names → `HashManager`. Core never reads that file. |
| Conditions | `assert_name`: basename, not a hash stem. |

**Potential issues**

- Hashing the whole recipe file (includes the name).
- Asking identity to delete run artifacts. `reclaim_name` is the engine: `release_name` then `rmtree`.
- Force stealing `hash_taken` or wiping `both_match`. Force only overwrites `name_taken`.

## 2. Processing

```python
class IdentityManager(ABC):
    def hash_payload(self, payload: object) -> str: ...
    def assert_name(self, name: object) -> str: ...
    def lookup(self, directory, *, name, configuration_hash) -> NameHashLookup: ...
    def claim(self, directory, *, name, configuration_hash) -> None: ...
    def release_name(self, directory, *, name) -> dict | None: ...
```

Four-way lookup: `both_free` / `both_match` / `name_taken` / `hash_taken`.

`get_new_run_or_overwrite` (fit): decision tree is the contract in [get_new_run_or_overwrite.md](get_new_run_or_overwrite.md).

`get_existing_run` (inspect): require `both_match` and the run artifacts directory exists.

`HashManager` keeps its existing method names and implements this ABC as aliases (`lookup` → `lookup_name_hash_json`, `claim` → `write_name_hash_json_unique`).

## 3. Output

A `RunIdentity` on the session. JSON pair under `run_identities_root`. Run artifacts directory is the caller’s path.
