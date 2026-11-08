# Process records (proposal)

Draft. Feature `.mdc` later: `.cursor/rules/dl_toolbox_structure/data_contract/process_records.mdc`.
Code: `src/dl_toolbox/data_contract/process_records.py`, `named_freeze.py`. Tests: `tests/data_contract/test_process_records.py`, `test_named_freeze.py`.

Design of the three objects: `standards/data_contract/philosophy.md`. Collection configuration: `standards/contract/proposals/collection_configuration.md`. The pair itself is HashManager (`.cursor/rules/dl_toolbox_structure/base/hashing/hash_manager.mdc`).

---

## Context

A generating process is one execution: the user names a `generating_process_name`, the class runs, a frozen collection appears. Two questions have to stay answered after that execution: was this signature already run, and under which name. HashManager is those questions for a `.json` pair. Training already uses it. A data process uses the same pair as its **completion record**.

The order is the other way around from a training run. Training writes the pair, then mkdirs artifacts. A data process must not mark the execution complete until payload (if any) and the collection configuration are on disk. `lookup` is the claim. `create_pair` is the last write. `generating_process_signature` is HashManager `extra`. The index document keys are `name` and `hash`. `name` is that execution's `generating_process_name`.

Payload is a different Directory. File-backed collection kinds own `artifacts_dir(generating_process_name)`. HashManager does not mkdir it. JSON-only collections have no payload Directory; those processes never pass one. Creating that folder without overwrite, and deleting it when the run fails, is this contract's payload story. Directory `CREATE` is `exist_ok=True` (Directory contract tension). Exclusive create here is `mkdir` without `exist_ok` on a Directory whose spec is `MISSING`.

Collection configuration is a third home: the collection kind's `configs_dir` (collection-configuration proposal). This contract writes it by calling `write_collection_configuration`. Incomplete cleanup and `remove_named_freeze` delete those Files by `File.from_under`, then unlink. They do not walk payload to discover a collection.

The identity index is a Directory HashManager created. Callers parse that Directory (`ALLOW_MISSING` or `EXISTS`) and call `create_identity_index` themselves. This contract does not take a Path and parse it. `lookup` / `create_pair` / `drop_pair` require an identity index. A Directory that merely exists is not one.

`remove_named_freeze` and `process_hash_occupant` are how a finished name is deleted so a later `process()` can claim it. They are not `process()`. `process()` has no overwrite flag.

### What process records need in order to operate

1. a HashManager already constructed
2. the identity-index Directory for this process kind
3. for file-backed processes: the payload Directory (`Collection.artifacts_dir(generating_process_name)`)
4. the collection kind's `configs_dir` Directory, when writing or deleting collection configuration

### What process records can do

- **claim_generating_process:** `lookup` the pair. The state must be `BOTH_FREE`.
- **create_payload_directory:** exclusive mkdir of a payload Directory whose spec is `MISSING`.
- **write_process_configuration:** `create_pair` with `extra={"generating_process_signature": ...}`.
- **delete_incomplete_run:** `drop_pair`, delete collection configuration Files if they exist, rmtree payload if it exists.
- **remove_named_freeze:** `drop_pair`, delete collection configuration Files, rmtree payload. Missing names are a no-op that returns false.
- **process_hash_occupant:** `lookup` of this hash; `HASH_TAKEN` or `BOTH_MATCH` yields the stored `name`.

---

## Terminology

Directory, File: `.cursor/rules/dl_toolbox_structure/base/file_management/directory.mdc`, `file.mdc`.
JSON: `.cursor/rules/dl_toolbox_structure/base/input_output/json_io.mdc`.
HashManager, pair states, `create_identity_index`, `lookup`, `create_pair`, `drop_pair`, `validate_human_readable_name`, `hash_payload`: `.cursor/rules/dl_toolbox_structure/base/hashing/hash_manager.mdc`.
Collection configuration: `standards/contract/proposals/collection_configuration.md`.

| Term | Meaning |
|---|---|
| `generating_process_name` | HashManager `human_readable_name` of this execution |
| `generating_process_signature` | process identity. HashManager `extra` on the index document |
| `configuration_hash` | HashManager `hash_payload` of `generating_process_signature` |
| identity index | Directory HashManager `create_identity_index` returned |
| `payload` | Directory `Collection.artifacts_dir(generating_process_name)`. File-backed only |
| `configs_dir` | Directory that is the collection kind's configuration folder |
| `claim_generating_process` | `lookup`; require `BOTH_FREE` |
| `create_payload_directory` | exclusive mkdir of `payload` |
| `write_process_configuration` | HashManager `create_pair` with the signature as `extra` |
| `delete_incomplete_run` | drop the pair, the collection Files, and `payload` after a failed `process()` |
| `remove_named_freeze` | drop a finished name so a later `process()` can claim it |
| `process_hash_occupant` | which `generating_process_name` currently holds this `configuration_hash` |

---

## Input Specification

A `HashManager` is already constructed.

`claim_generating_process`, `write_process_configuration`, and `delete_incomplete_run` take an identity index: a Directory that exists and is an identity index. The caller already ran `create_identity_index`.

`generating_process_name` is a raw label; `validate_human_readable_name` produces the basename. `configuration_hash` is a full SHA-256.

`write_process_configuration` takes `generating_process_signature`: a JSON object. It becomes `extra`. It must not contain `name` or `hash`.

`payload` is a Directory. File-backed behaviors take it. JSON-only behaviors omit it. At `create_payload_directory`, spec is `MISSING`. After a successful create, spec is `EXISTS`. `delete_incomplete_run` / `remove_named_freeze` rmtree it only when spec is `EXISTS`.

`configs_dir` is a Directory. `delete_incomplete_run` / `remove_named_freeze` delete collection Files under it when spec is `EXISTS`. Spec `MISSING` means no collection configuration was written.

`remove_named_freeze` / `process_hash_occupant` take an identity-index Directory that may be `MISSING` (no process pair yet) or `EXISTS` (must then be an identity index). They do not `create_identity_index`.

---

## Main functionalities

### `claim_generating_process(manager, directory, *, generating_process_name, configuration_hash)`

`lookup` the pair. The state must be `BOTH_FREE`.

**input:** `manager`, `directory`, `generating_process_name`, `configuration_hash`.

**assumptions on the input:** `directory` is an identity index. `configuration_hash` is a full SHA-256.

**steps:**

```
name = manager.validate_human_readable_name(generating_process_name)
lookup = manager.lookup(
    directory, human_readable_name=name, hash=configuration_hash,
)
if lookup.state is not BOTH_FREE:
    BOTH_MATCH → FileExistsError: this pair already exists
    NAME_TAKEN → FileExistsError: name exists with occupant hash
    HASH_TAKEN → FileExistsError: hash already saved as occupant name
return name
```

This is the locked HashManager read. It does not `create_pair`. File-backed processes then call `create_payload_directory`. JSON-only processes do not.

**side effects:** none on the index JSON. HashManager acquires and releases the exclusive lock for the lookup.

**outputs:** the validated `generating_process_name`. Later `write_process_configuration` of this pair is `BOTH_MATCH`.

### `create_payload_directory(payload)`

exclusive mkdir of a payload Directory whose spec is `MISSING`.

**input:** `payload`.

**assumptions on the input:** `payload` is a Directory. Spec is `MISSING`.

**steps:**

```
Directory.from_absolute(path=payload.path.parent, policy=CREATE)
if payload.spec is EXISTS: FileExistsError
payload.path.mkdir(exist_ok=False)
return Directory.from_absolute(path=payload.path, policy=MUST_EXIST)
```

Parent `CREATE` is allowed to already exist (the collection kind's payload root). The `generating_process_name` folder is the exclusive create.

**side effects:** `payload` now exists as a directory. A second call is `FileExistsError`.

**outputs:** the same path as a Directory with spec `EXISTS`.

### `write_process_configuration(manager, directory, *, generating_process_name, configuration_hash, generating_process_signature)`

`create_pair` with `extra={"generating_process_signature": ...}`.

**input:** `manager`, `directory`, `generating_process_name`, `configuration_hash`, `generating_process_signature`.

**assumptions on the input:** after the locked read inside `create_pair`, pair state is `BOTH_FREE`. `generating_process_signature` is a JSON object.

**steps:**

```
name_path, hash_path = manager.create_pair(
    directory,
    human_readable_name=generating_process_name,
    hash=configuration_hash,
    extra={"generating_process_signature": generating_process_signature},
)
return File.from_absolute(path=name_path)
```

**side effects:** HashManager writes the hash file first, then the name file, under the exclusive lock.

**outputs:** the name-side File. Later `lookup` of this pair is `BOTH_MATCH`. The document keys are `name`, `hash`, `generating_process_signature`.

### `delete_incomplete_run(manager, directory, *, generating_process_name, configs_dir, payload)`

drop the pair, the collection Files, and `payload` after a failed `process()`.

**input:** `manager`, `directory`, `generating_process_name`, `configs_dir`. File-backed calls also take `payload`. JSON-only calls omit `payload`.

**assumptions on the input:** `directory` is an identity index. `configs_dir` is a Directory. If `payload` is passed, it is a Directory.

**steps:**

```
manager.drop_pair(directory, human_readable_name=generating_process_name)
if payload was passed and payload.spec is EXISTS:
    rmtree payload.path
if configs_dir.spec is EXISTS:
    delete_collection_configuration(
        configs_dir, generating_process_name=generating_process_name, manager=manager,
    )
```

`drop_pair` returns `None` when the name was never stored (write-last: the usual incomplete case). Collection Files are absent until `write_collection_configuration` succeeded.

**side effects:** process pair gone if it had been written. Collection Files unlinked if they existed. Payload tree removed if it existed.

**outputs:** none.

### `remove_named_freeze(manager, directory, *, generating_process_name, configs_dir, payload)`

drop a finished name so a later `process()` can claim it.

**input:** `manager`, `directory`, `generating_process_name`, `configs_dir`. File-backed calls also take `payload`. JSON-only calls omit `payload`.

**assumptions on the input:** `directory` is an identity index, or spec `MISSING`. `configs_dir` is a Directory. If `payload` is passed, it is a Directory.

**steps:**

```
name = manager.validate_human_readable_name(generating_process_name)
dropped = None
if directory.spec is EXISTS:
    dropped = manager.drop_pair(directory, human_readable_name=name)
collection_removed = False
if configs_dir.spec is EXISTS:
    collection_removed = delete_collection_configuration(
        configs_dir, generating_process_name=name, manager=manager,
    )
payload_exists = payload was passed and payload.spec is EXISTS
if dropped is None and not collection_removed and not payload_exists:
    return False
if payload_exists:
    rmtree payload.path
return True
```

**side effects:** same three homes as `delete_incomplete_run`, when those objects existed.

**outputs:** `True` when something was there, `False` when the name was already absent.

### `process_hash_occupant(manager, directory, *, configuration_hash)`

which `generating_process_name` currently holds this `configuration_hash`.

**input:** `manager`, `directory`, `configuration_hash`.

**assumptions on the input:** `configuration_hash` is a full SHA-256. `directory` is an identity index, or spec `MISSING`.

**steps:**

```
if directory.spec is MISSING: return None
probe = a legal generating_process_name used only for this lookup
lookup = manager.lookup(
    directory, human_readable_name=probe, hash=configuration_hash,
)
if lookup.state is HASH_TAKEN: return lookup.document["name"]
if lookup.state is BOTH_MATCH: return probe
return None
```

`NAME_TAKEN` means this hash is free (HashManager checks `HASH_TAKEN` first). `BOTH_FREE` means the hash is free.

**side effects:** none on the index JSON. HashManager lock for the lookup.

**outputs:** the occupant `generating_process_name`, or `None`.

---

## Secondary functionalities

### Collection configuration delete

`delete_collection_configuration` is the collection-configuration proposal. `delete_incomplete_run` and `remove_named_freeze` call it when `configs_dir` exists.

### Exclusive payload mkdir

Directory `CREATE` is `mkdir(parents=True, exist_ok=True)`. `create_payload_directory` is not that parse. Spec `MISSING`, then `mkdir(exist_ok=False)`, then the living Directory is `EXISTS`. The parent of `payload` may be created with `CREATE` (shared kind root).

### `generating_process_name`

`manager.validate_human_readable_name`. Same object as HashManager's human readable name.

---

## Tensions

- **`claim_generating_process` then `write_process_configuration` are two HashManager lock acquisitions.** Write-last is the completion-record rule. Another process can observe `BOTH_FREE` in the gap, do the same work, and lose at `create_pair`. Payload exclusive mkdir only serializes the folder, not the pair.
- **`process_hash_occupant` needs a hash-only lookup.** HashManager `lookup` requires a name. The probe name is a legal `generating_process_name`. If that probe is the real occupant, the state is `BOTH_MATCH` and the probe is the correct answer. If the probe is `NAME_TAKEN` by a different hash, this hash is free.
- **`drop_pair` then collection unlink then payload rmtree are not one lock.** Another process can claim the name after `drop_pair` and before payload is gone. Freeze is not `process()`.
- **A crash between collection hashed write and alias write** is the collection-configuration tension. Cleanup finds the alias with `File.from_under`; if only the hashed File exists, this delete does not see it.
- **File has no delete; Directory spec goes stale after rmtree.** Unlink and rmtree are the disk ops. The File / Directory objects held from before those ops are not re-parsed.
