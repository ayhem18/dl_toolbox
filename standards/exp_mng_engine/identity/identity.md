# IdentityManager

Project-level exclusive `(name, hash)` pair. Default implementation is `HashManager`. The digest is SHA-256 of `{dl_toolbox_config, experiment_config}`. The name is a label. It is not hashed.

## 1. Input

| | |
|---|---|
| Expected | Records directory (injected by `EnginePaths`). Basename `name`. Full SHA-256. |
| Comes from | Default: `identity.json` → prefix length and folder names → `HashManager`. Core never reads that file. |
| Conditions | `assert_name`: basename, not a hash stem. |

**Potential issues**

- Hashing the whole recipe file (includes the name).
- Asking identity to delete payload. `reclaim_name` is the engine: `release_name` then `rmtree`.
- Force stealing `hash_taken`.

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

Claim (fit): `both_free` → write the JSON pair + mkdir payload, under a records-dir lock.

Open (inspect): require `both_match` and payload exists.

`HashManager` keeps its existing method names and implements this ABC as aliases (`lookup` → `lookup_name_hash_json`, `claim` → `write_name_hash_json_unique`).

## 3. Output

A `RunIdentity` on the session. Records JSON pair under the records directory. Payload is the caller’s path.
