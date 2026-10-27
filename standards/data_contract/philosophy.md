# Data architecture philosophy

This note is the **design contract**. How the Python should look so that design stays visible is [philosophy_in_code.md](philosophy_in_code.md). Executable helpers live in `dl_toolbox.data_contract`. A project still writes its own entities, collections, and processes.

If an entity or collection cannot be written without a default, a `None`, or a slot this note does not name, this contract is wrong or incomplete. Revise it here before writing more code.

Data work is readable when every piece is one of three objects, files are one of two kinds, and non-tree joins are frozen as a **lock** instead of being re-solved at every later stage.

A **level** is those three objects for one kind: the data entity, the data collection, and the generating process that produces that collection. Frame-level frame, frame-level CLAHE, lock, crop — each is one level. The three objects of a level are not interchangeable.

## Assumption

Once a piece of data has an identifier, its content does not change.

This contract does not detect or repair mutation under a stable id. If the bytes change and the id stays, that is a data-integrity failure outside this design. The contract tracks which frozen sets exist and how they were asked for. Load checks that the configuration hash matches the identity document. It does not checksum payload bytes.

## What this is not

Experiment tracking (models, losses, seeds, run comparison) is later context. It is not a level of this contract.

## Design decisions

### 1. Three objects, never mixed

**Data entity.** One thing a later stage is allowed to ask questions of: a frame, a mask, a crop. It holds the fields those questions need. `to_dict()` is its identity. A question that needs bytes is asked of the collection. It does not take the collection’s artifact directory. It does not store an absolute path. It does not load payload. It does not write the collection configuration. It does not compute the collection hash. It does not hold pixels.

**Collection.** A frozen set of entities of one kind. It is built in memory. It creates its own identity: the generating-process signature plus the identities of its members. It owns member access, that identity hash, and reads of member payload. Where its configuration belongs is a property of that collection *kind*, not of one instance. Consumers load a collection by hash or `generating_process_name` alias. They never call the generating process. They never walk payload folders to discover a collection.

A collection that **points at** another collection (a hash already on it) exposes an instance method that calls that other class’s `load`. Consumers do not guess the Python class. `related_collections()` returns a dict of zero-argument callables; calling one entry is what loads. Collections with exactly one parent also have `parent_collection()`. `FrameCollection.related_collections()` is `{}`. A lock has one callable per filled slot, keyed by class constants on `Lock`. `Collection.load` remains the reconstruction path. Do not add `get()`.

**Process.** The only place work happens. The user specifies a strategy and args; the oracle returns the class that implements the steps. Instantiation does not take `generating_process_name`. `process(generating_process_name=...)` is one execution of that class. The process writes payload **if this process has bytes**, using relative paths it owns under the collection’s artifact directory. It constructs entities and the collection in memory, asks the collection for its configuration document and for the collection kind’s configuration folder, and writes that file. Then it writes its own configuration. Where that process configuration belongs is a property of that process *kind*, not of one execution. A `generating_process_name` is one execution of that process. It is not process identity.

If a function is two of these at once, it will be hard to debug. If you can name which of the three you are holding, the design is readable.

### 2. Two kinds of files

| Kind | What it is |
|---|---|
| **Payload** | Artifacts: images, masks, rasters, caches, restaged trees. Actual bytes. |
| **Configuration** | Identity, usually a JSON file. |

A file-backed **collection kind** owns the payload root. `Collection.artifacts_dir(generating_process_name)` is a static method of that kind: it returns the directory for one generating-process execution. The generating process does not own that directory. It asks the collection class for it, writes bytes there, and names files with **relative** paths. Those relative paths are identity on the entities. A JSON-only collection (a lock, a split) has no `artifacts_dir`.

Process and collection **both** have a configuration. Same kind of file. Different owner, different question, different home.

The home of each configuration is a property of that *kind*, not of one instance or one execution. The process is the writer of both. The collection builds its configuration in memory. It does not write its file. The process asks the collection for `configs_dir` and for `to_dict`, then writes.

| Configuration | Question it answers | Home (property of the kind) |
|---|---|---|
| **Collection** | What is this frozen set? Consumers `load` this file. | The collection kind’s folder / `{hash-prefix}.json` (optional `{generating_process_name}.json` alias) |
| **Process** | Which strategy ran, with which signature, under which `generating_process_name`? | The process kind’s pair under [`HashManager`](../exp_mng_engine/identity/identity.md): `{human_readable_dirname}/{name}.json` and `{hashes_dirname}/{hash-prefix}.json` (defaults `human_readable/` and `hashes/`) |

The collection configuration is the identity the in-memory collection already has. The process configuration stores `generating_process_signature` plus the `generating_process_name` of that execution. They are not interchangeable.

Write order:

1. The oracle maps strategy plus args to a class. `generating_process_name` is not an oracle argument.
2. `process(generating_process_name=...)` claims the `generating_process_name` and the generating-process signature. It does not write the process configuration yet.
3. If this collection is file-backed, the process calls `Collection.artifacts_dir(generating_process_name)` and creates that directory without overwrite.
4. The process writes payload (if any) under that directory. Relative paths are its decision. It passes those relative paths into the data-entity constructors.
5. It constructs the entities and the collection in memory. The collection joins relative paths to `artifacts_dir(generating_process_name)` and indexes the files.
6. The collection produces its configuration document (`to_dict`). That document stores `generating_process_name` as context. It is not identity.
7. The process asks the collection class for `configs_dir` and writes the collection configuration.
8. The process writes its own configuration. That file is the completion record.

An incomplete run deletes leftover payload, the collection configuration it started, and any process-configuration files. If only the collection configuration exists, the run did not finish. Writing the process configuration before payload would mark an unfinished execution as complete. Do not do it.

Bytes live under the collection’s artifact directory for that `generating_process_name`. A configuration filename is a short hash prefix; the document stores the full hash. Putting a hash in byte paths is the chicken-and-egg (hash needs paths, paths need the hash directory). Do not do it.

A path field on a **file-backed data entity** is relative to the collection’s `artifacts_dir(generating_process_name)`. The entity does not take that directory as a constructor argument, does not store it as a field, and does not receive it in `from_payload`. The collection joins the relative path when it indexes members: the file must exist and stay under that directory. The joined absolute path stays in the collection’s index, beside the member, under the same id. It is not a field on the entity. Payload reads are methods on the collection (`load_image(member_id)`, and the reads this level names). They use that index. JSON-only entities have no path field. A JSON-only collection has no payload index.

The artifact directory is a property of the collection kind plus `generating_process_name`. Reconstruct it with `Collection.artifacts_dir(generating_process_name)`. Do not store a machine path or a repo-relative absolute path on the collection document. Do not pass that directory into the collection constructor as a free string that can disagree with `generating_process_name`. Do not put the directory on the entity. Do not attach the joined path to the entity as a derived attribute. Do not return a second entity that carries it. Location stays on the collection.

### 3. The collection is the identity document

The identity is created in memory on the collection, not by a round-trip through the filesystem. The process already has that document when payload (if any) is done: it takes the JSON and the collection kind’s folder, and writes the file. Consumers `load` that file. Forcing the process to write the file and read it back is not the contract.

The process has one write helper for collection configurations. The collection has `to_dict`, `from_payload`, and `load`. It does not have `save`. Tests that build a collection without a process use that same helper. A second write path is two objects mixed.

### 4. A generating_process_name is one execution

Unique per generating process. The process claims **the `generating_process_name` and the generating-process configuration** (`generating_process_signature`). After payload (if any) and the collection configuration are on disk, it writes the same process-configuration document through [`HashManager`](../exp_mng_engine/identity/identity.md) — the same exclusive pair training runs use. Folder names come from `project_level_configs/identity.json`, not from constants in the process class. Defaults:

- `human_readable/{generating_process_name}.json`
- `hashes/{hash-prefix}.json`

`configuration_hash` is the hash of `generating_process_signature`. The filename is that hash’s prefix, not the full digest. `generating_process_name` is not in that hash. Reusing a finished `generating_process_name` is an error. Reusing a finished configuration under a new `generating_process_name` is an error. `process()` has no overwrite flag that steals a `generating_process_name`.

A separate helper may delete a finished freeze so a later `process()` can claim those files: `remove_named_freeze` removes the process pair, the collection alias and hashed collection file, and a payload directory the caller already resolved from `Collection.artifacts_dir(generating_process_name)`. `process_hash_occupant` returns which `generating_process_name` currently holds a process hash. Those helpers are not `process()`. They do not read an artifact path from the collection document.

Availability is decided by reading those files, not by the filename alone: the document’s `generating_process_name`, `hash`, and `generating_process_signature` must agree. That document is the process configuration. The collection configuration is a different document. Its home belongs to the collection kind.

### 5. Identity

**Data entity.** `to_dict()` is the identity document. It returns exactly the fields that are the identity of the entity. There is no identity field missing from `to_dict()`, and no `to_dict()` key that is not identity.

The constructor takes those identity fields. The default is that every constructor argument is identity and therefore in `to_dict()`. A constructor argument that is **passed** and is **not** identity is an exception: the level note must name it and justify (1) why it is passed and (2) why it is not identity. An artifact directory is not such an exception. It is not passed to the entity.

For a file-backed entity, the relative path is identity: it names which payload file in the collection’s directory. It is not a location on disk.

**Data collection.** Identity is the generating-process signature plus the identities of its member entities (each entity’s `to_dict()`, or the lock document, which is the member). Same signature and same members → same hash. The collection configuration filename is that hash’s prefix; the document stores the full hash. It is never a payload directory name.

Every collection stores `generating_process_name`. That is how a consumer ties this frozen set to one process configuration and, if the collection is file-backed, to `artifacts_dir(generating_process_name)`. It is **not** identity. Do not hash it. Do not hash timestamps, machine paths, or “we copied files.” The artifact directory is not stored; it is rebuilt from the kind and `generating_process_name`.

**Process.** Identity is `generating_process_signature`. Same signature → same hash. A `generating_process_name` is not identity.

The process signature is one dict, used as-is: strategy plus the args that change members.

### 6. An entity is local to its collection

There is no global “this frame” (or mask, or crop) outside a collection. A child points at **this entity in this collection**: `parent_collection_hash + entity_id`.

A file-backed entity’s path is relative to that collection’s artifact directory. Bytes are asked for through the collection (it owns the folder). `member(id)` returns the entity. The collection’s load methods return the bytes for that id. The entity is not independently locatable on disk.

### 7. If a field is not required, it is not in the contract

This applies to **data entities and data collections**. No slot “for later levels.” No `created_at` because it is not identity and not needed to resolve files. No parent-hash field on a collection that has no parents. A missing required field is an error, not a fallback.

The contract is named in full words. Abbreviations are how the contract goes missing.

### 8. Fan-out, then lock

A process can **fan out** from one collection: several children, each with a single parent. A mask view is a child of a frame collection. An image auxiliary is also a child of that frame collection. Those children are siblings. Neither belongs inside the other’s identity.

If a later process can define its members from **one** collection instance, it is a child of that collection. The entity pointer is `parent_collection_hash + entity_id`.

If it needs **more than one** collection instance, those instances are locked first. The process’s parent is the lock. The entity does not carry two parent hashes.

A **lock** is a data entity: the named slots for this combination, each an exact collection hash. The collection is the frozen set of that entity (one lock document). Same slots → same lock. A lock is JSON-only. Do not invent a generic `(kind, hash)` list. The lock *kind* names which slot names are legal — a data type, not a universal schema across projects, and not something invented inside a training run. A new combination of admitted slots and exact hashes is a new lock **instance**, not a second lock type and not an optional field on an old document. Do not add `LockWithAux` / `LockWithoutAux`.

A lock is still a frozen collection. Consumers `load` it by hash. A run only chooses an instance.

The set is exact instances (hashes), not “whatever is in the folder.” A frame split only needs the view. Auxiliaries do not belong in the split’s identity. When a later stage loads the combination, its parent is the lock.

### 9. Each process is specific to the collection it produces

Frame preparation, overlap grouping, and crop sampling do not share one sequence of steps.

What is shared: the user names a strategy; the class implements that process’s steps; `process(generating_process_name=...)` claims the `generating_process_name`, writes payload if any under `Collection.artifacts_dir(generating_process_name)`, constructs the collection in memory, writes the collection configuration, then writes the process configuration. Consumers load only collections.

### 10. Promote only what is battle-tested

This pattern is meant to transfer. Extra machinery for one dump or one Kaggle layout is not the contract. No backward compatibility unless the user asks for it.

## What this is for

Print three lines and understand a run: which process and `generating_process_name`, which collection hash and size, which entity id and kind. If a later stage fails, it failed because it asked an entity a question or because a parent hash did not match—not because someone unpacked a tuple of relative paths.
