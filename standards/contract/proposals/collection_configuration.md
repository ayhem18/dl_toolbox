# Collection configuration (proposal)

Draft. Feature `.mdc` later: `.cursor/rules/dl_toolbox_structure/data_contract/collection_configuration.mdc`.
Code: `src/dl_toolbox/data_contract/configs.py`. Tests: `tests/data_contract/test_configs.py`.

Design of the three objects (entity, collection, process): `standards/data_contract/philosophy.md`. This note is the on-disk collection configuration that consumers `load`.

---

## Context

A later stage is allowed to ask questions of a frozen set: how many members, what is this entity, what are the bytes for that id. It is not allowed to call the generating process to find those answers. The collection already built its identity in memory (`to_dict`, the hash). That document has to live on disk in a place that belongs to the collection *kind*, so `Collection.load` is a read of a known folder, not a walk of payload trees.

That folder is one Directory. The document is stored twice in it: once under a prefix of the collection hash, once under the `generating_process_name` of the execution that produced this set. Consumers load by hash or by that name. They are looking at the same JSON object.

This is a different home from the process pair. HashManager (`.cursor/rules/dl_toolbox_structure/base/hashing/hash_manager.mdc`) keeps `(name, hash)` in two subdirectories of an identity index, under an exclusive lock, as the completion record of an execution. The collection configuration is the identity of the frozen set. Both files sit in the kind's folder. A HashManager identity index here would make `Collection.load` ask about process records. The process writes this document; the collection does not have `save`.

Disk is File and Directory (`.cursor/rules/dl_toolbox_structure/base/file_management/file.mdc`, `directory.mdc`). JSON is `read_object` / `write_object` (`.cursor/rules/dl_toolbox_structure/base/input_output/json_io.mdc`). A missing configuration is not a File. `File.from_under` either returns a File or `FileNotFoundError`. Existence is that parse, not `is_file` on a Path the caller still holds.

The hashed filename stem is the same prefix HashManager uses for hash-file basenames: the first `prefix_len` characters of the full digest. `prefix_len` lives on the HashManager the project already constructed. `generating_process_name` is HashManager `validate_human_readable_name`. This folder is still not an identity index.

### What collection configuration needs in order to operate

1. a HashManager (for `prefix_len` and `validate_human_readable_name`)
2. the collection kind's configuration Directory
3. the collection's document (`to_dict`), its `collection_hash`, and its `generating_process_name`

### What collection configuration can do

- **write_collection_configuration:** take the in-memory collection's document and write both JSON files under the kind's Directory. Hashed file first, alias second.
- **load_collection_configuration:** given an identity (name, full hash, or prefix), parse the File under that Directory and return the document.
- **delete_collection_configuration:** given a `generating_process_name`, unlink the alias File and the hashed File under `configs_dir`.

---

## Terminology

Directory, File: `.cursor/rules/dl_toolbox_structure/base/file_management/directory.mdc`, `file.mdc`.
JSON: `.cursor/rules/dl_toolbox_structure/base/input_output/json_io.mdc`.
HashManager: `.cursor/rules/dl_toolbox_structure/base/hashing/hash_manager.mdc`.
Design: `standards/data_contract/philosophy.md`.

| Term | Meaning |
|---|---|
| `configs_dir` | Directory that is the collection kind's configuration folder. `Collection.configs_dir()` |
| collection document | JSON object `collection.to_dict()`. Keys include `hash`, `generating_process_name`, `generating_process_signature`, and the member identities |
| `collection_hash` | full 64-character lowercase SHA-256 stored as `document["hash"]` |
| `generating_process_name` | HashManager `validate_human_readable_name` of the execution that produced this set. Context on the document. Not identity |
| hashed file | File `{prefix}.json` under `configs_dir`. The document still stores the full `collection_hash` |
| alias file | File `{generating_process_name}.json` under `configs_dir`. Same document as the hashed file |
| `prefix` | first `manager.prefix_len` characters of `collection_hash` |
| `identity` | the key `load_collection_configuration` takes: a `generating_process_name`, a full `collection_hash`, or a `prefix` |
| `write_collection_configuration` | write both files from the in-memory collection |
| `load_collection_configuration` | parse one File under `configs_dir` from `identity` |
| `delete_collection_configuration` | unlink alias and hashed Files for one `generating_process_name` |

The two files of a healthy write are equal JSON objects.

---

## Input Specification

A `HashManager` is already constructed. Every prefix and every "does this `identity` look like a prefix" check uses `manager.prefix_len`. `generating_process_name` goes through `manager.validate_human_readable_name`.

`configs_dir` is a Directory. `write_collection_configuration` accepts it missing or existing: missing becomes `DirectoryParsePolicy.CREATE`, then both writes. `write_object` requires a Directory that exists. `load_collection_configuration` requires `configs_dir` with spec `EXISTS`.

`write_collection_configuration` takes the collection and `manager`. The collection has `generating_process_name`, `collection_hash`, `to_dict()`, and `configs_dir()` returning that Directory. `to_dict()["hash"]` equals `collection_hash`. `to_dict()["generating_process_name"]` equals the validated name.

`load_collection_configuration` takes `configs_dir`, `identity`, and `manager`. `identity` is a non-empty string after strip. A missing File is `FileNotFoundError` from `File.from_under`, not an empty document.

The collection document is read with `[]`: `hash` is a full SHA-256. Other keys are the collection's identity plus `generating_process_name`. Missing `hash` on an existing File is a corrupt configuration.

`read_object` / `write_object` are the JSON contract. Form for both files is `JsonTextForm.DOCUMENT`. Hashed file is written first, alias second.

---

## Main functionalities

### `write_collection_configuration(collection, *, manager)`

take the in-memory collection's document and write both JSON files under the kind's Directory.

**input:** `collection`, `manager`.

**assumptions on the input:** Input Specification.

**steps:**

```
name = manager.validate_human_readable_name(collection.generating_process_name)
document = collection.to_dict()
if document["hash"] != collection.collection_hash: error
if document["generating_process_name"] != name: error
configs_dir = collection.configs_dir()
if configs_dir.spec is MISSING:
    configs_dir = Directory.from_absolute(path=configs_dir.path, policy=CREATE)
hashed_name = {prefix of collection.collection_hash}.json
alias_name = {name}.json
write_object(directory=configs_dir, name=hashed_name, payload=document, form=DOCUMENT)
write_object(directory=configs_dir, name=alias_name, payload=document, form=DOCUMENT)
return the hashed File
```

`prefix` is the first `manager.prefix_len` characters of `collection.collection_hash` (Secondary functionalities).

**side effects:** two Files under `configs_dir`. Write order is hashed file first, alias second. `write_object` is one-file atomic. The pair of files is not one rename.

**outputs:** the hashed File. Later `load_collection_configuration` of the name, the full hash, or the prefix returns the same document.

### `load_collection_configuration(configs_dir, identity, *, manager)`

given an identity (name, full hash, or prefix), parse the File under that Directory and return the document.

**input:** `configs_dir`, `identity`, `manager`.

**assumptions on the input:** `configs_dir` exists. `identity` is a non-empty string after strip.

**steps:**

```
text = strip identity
if text is empty: error
file = File.from_under(root=configs_dir, relative_path={text}.json)
    except FileNotFoundError:
        if text is a full SHA-256:
            file = File.from_under(root=configs_dir, relative_path={prefix of text}.json)
        else:
            error: no configuration
document = read_object(file=file)
if "hash" missing or not a full SHA-256: error
if file stem is a prefix (length prefix_len or 64, lowercase hex):
    document["hash"] must start with that stem
if text is a full SHA-256:
    document["hash"] must equal text
if text is a prefix:
    document["hash"] must start with text
return file, document
```

Each candidate path is parsed as File at most once. `is_file` then `read` of the same path is the inner waste this call must not do.

**side effects:** none. Files unchanged.

**outputs:** the File that was read, and the collection document. `Collection.from_payload` is the collection's reconstruction; this call is the disk read.

### `delete_collection_configuration(configs_dir, *, generating_process_name, manager)`

given a `generating_process_name`, unlink the alias File and the hashed File under `configs_dir`.

**input:** `configs_dir`, `generating_process_name`, `manager`.

**assumptions on the input:** `configs_dir` exists. `generating_process_name` is a raw label; `validate_human_readable_name` produces the basename.

**steps:**

```
name = manager.validate_human_readable_name(generating_process_name)
alias = File.from_under(root=configs_dir, relative_path={name}.json)
    except FileNotFoundError: return False
document = read_object(file=alias)
hashed_name = {prefix of document["hash"]}.json
hashed = File.from_under(root=configs_dir, relative_path=hashed_name)
    except FileNotFoundError: skip
    else: unlink hashed.path
unlink alias.path
return True
```

**side effects:** alias File deleted when it existed. Hashed File deleted when it existed and `document["hash"]` names it. After unlink those File objects are stale.

**outputs:** `True` when the alias existed, `False` when `File.from_under` for the alias was `FileNotFoundError`.

---

## Secondary functionalities

### Hashed filename

The hashed File's basename stem is the first `manager.prefix_len` characters of the full digest, the same prefix HashManager uses for hash-file basenames (`artifact_stem` with that `prefix_len`). The document stores the full digest.

A stem looks like a prefix when it is lowercase hex of length `manager.prefix_len` or 64 (`is_hashed_artifact_stem` with that `prefix_len`). `load_collection_configuration` uses that check on the File's stem and on `identity`.

### `generating_process_name`

`manager.validate_human_readable_name`. A `generating_process_name` must not look like a prefix. The alias filename is `{name}.json`.

### Directory parse on write

`write_object` does not mkdir. If `configs_dir` is missing, `DirectoryParsePolicy.CREATE` runs before the two writes. After that, spec is `EXISTS`.

---

## Tensions

- **A crash between the hashed write and the alias write** leaves a hashed File without an alias. `load_collection_configuration` by hash succeeds; by `generating_process_name` is `FileNotFoundError`. `write_object` is atomic per File; the pair is not one rename.
- **`DirectoryParsePolicy.CREATE` is `exist_ok=True`.** Two processes can both create `configs_dir` and both write the same alias. Process-records claim is supposed to make the name unique first. That claim is a different lock (HashManager's identity index). The gap is still there.
- **File has no delete.** Incomplete-run cleanup and `remove_named_freeze` unlink `File.path` after a successful `File.from_under`. After unlink the File object is stale. That is the same move HashManager `drop_pair` already makes on index JSON.
