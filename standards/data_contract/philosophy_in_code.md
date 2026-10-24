# Philosophy in code

Design lives in [philosophy.md](philosophy.md). This note is how to write the Python so that design stays readable, and so it is obvious when the *design* is the thing that must change.

The rules below are **style**. They are not a second contract. They are also **not optional**. Follow them. If the next implementation is forced to break one, **stop and warn**. Do not drop the rule in the code. Do not decide it does not matter. The user decides whether to change the writing or the design.

If an **entity or collection** cannot be written without a default, a `None`, or a slot [philosophy.md](philosophy.md) does not name, that is a **design** stop: revise [philosophy.md](philosophy.md). Do not weaken the design to make an implementation fit.

## When to stop

### Design — revise [philosophy.md](philosophy.md)

Only these are design problems, and only on **data entities and data collections**:

- a default on an entity or collection constructor
- `.get()` for a required JSON key
- a `None` sentinel instead of a class constant for a domain absence
- a field that is not in the level note
- an identity field missing from `to_dict()`, or a `to_dict()` key that is not identity
- a constructor argument that is not identity, unless the level note names it and justifies why it is passed and why it is not identity
- an artifact-directory argument on an entity constructor or on entity `from_payload`
- an artifact-directory argument on a collection constructor or on collection `from_payload`
- an artifact-directory path stored in the collection configuration document
- an absolute path stored on the entity, including one attached at index time
- a payload `load` on the entity
- a second entity, returned by the collection, whose job is to carry the joined path
- collection identity that omits `generating_process_signature` or the member identities
- collection identity that includes `generating_process_name`, artifact directory, timestamps, or machine paths
- a merged index “for convenience”
- a `save` on the collection, or a second write path besides the process helper
- the process forced to write the collection file and read it back

### Style — stop and warn

If the implementation cannot follow a rule in **How the Python should look** without breaking it, stop. Say which rule and why it cannot be followed. Do not proceed. Do not treat the rule as optional.

## How the Python should look

Names such as `Frame.UNLABELED`, `non_test`, and `test` in this note are **examples for one hub**. Replace them with this project’s contract words. Do not keep them as the ritual.

### Usual classes, not dataclasses

An entity is a usual class: keyword-only `__init__` with identity fields, plus `from_payload` as the JSON constructor. `to_dict` and `from_payload` are the identity document. A constructor argument that is not in `to_dict()` is an exception the level note must name and justify. The entity constructor does not take a directory.

A collection is the same shape: keyword-only `__init__`, no defaults. It takes `generating_process_name` (not identity). It does not take an artifact directory. A file-backed collection computes that directory with `cls.artifacts_dir(generating_process_name)` inside `__init__`. It builds its identity in memory: hash of `generating_process_signature` plus each member’s `to_dict()` (or the lock document). `from_payload` unpacks a saved document into that constructor and checks `hash`. It does not read an artifact-directory key. `to_dict` does not write one. `load` reads the configuration file. Payload bytes are separate methods on the same collection, keyed by member id.

If this collection points at another, name the loader after that class (`load_frame_collection`, `load_lock_collection`). `related_collections()` values are bound methods (or `partial` when the loader takes an aux class). `parent_collection()` exists only when there is exactly one parent. Do not implement `parent_collection()` on `FrameCollection` or `LockCollection`. `from_payload` still rebuilds **this** collection only.

The collection has `to_dict`, `from_payload`, and `load`. It does not have `save`. The process ABC owns one helper that writes the collection configuration to the collection class’s folder. Tests that build a collection without a process use that same helper.

### No defaults on entities or collections

Every constructor argument on an entity or collection is required. Every JSON key the object needs is read with `[]`, not `.get()`. A missing field is an error. Membership is `in`, not `.get()` with a fallback.

If a field is not required, it is not on the class. The joined absolute path is not a field on the entity. It lives in the collection’s index.

This rule does not apply to processes.

### The contract words are the identifiers

Use the names from **this** project’s level note. Do not abbreviate domain words (`process`, `rel`, `payload` for three different things). Unclear names mean an unclear contract.

### Domain values are class constants

Absence and kind are class constants on the entity (example on one hub: `Frame.TEST`, `Frame.NON_TEST`, `Frame.UNLABELED`). Not module-level aliases, not string literals scattered in scripts, not `None` for “unlabeled.”

### Oracle returns a class

Strategy name → class. Instantiation is `build_*(strategy=..., args=...)` with a required `args` dict. The registry lives with the process package.

### The process ABC owns the shared loop; strategies own the steps

`process(generating_process_name=...)` claims the `generating_process_name`, asks the collection class for `artifacts_dir(generating_process_name)` when file-backed, writes payload **if any** at relative paths the strategy owns, constructs the collection in memory, writes the collection configuration with the ABC helper, then writes the process configuration. Strategies implement the per-item steps. Consumers never import a strategy to find bytes.

A JSON-only collection has no `artifacts_dir`. A JSON-only process does not write payload.

### Index in one pass, in the shape you query

Build the dictionaries you will look up, keyed by id. Member access is O(1). Do not merge-sort a third list. Hash members after calling each entity’s `to_dict()`. Do not invent a parallel member sequence. Do not hash a raw incoming document that the entity did not emit.

Collection identity is `generating_process_signature` plus those member documents. Process identity is `generating_process_signature` only. `generating_process_name` is not in either hash.

### Resolve paths on the collection, once, when indexing members

A file-backed entity stores a path relative to the collection’s artifact directory. That relative path is identity. The entity is not passed the directory. The entity does not store the joined absolute path. The entity does not load payload.

The collection owns the artifact directory. When it indexes a member it joins that directory to the entity’s relative path, requires an existing absolute file that stays under the directory, and keeps that absolute path in its own index, keyed by the member id. Payload reads are methods on the collection. They take that id and use the index. `member(id)` returns the entity. JSON-only entities have no path field. A JSON-only collection has no payload index and no payload load.

The home of a collection configuration is a static method on the collection class (`configs_dir`). The home of a file-backed collection’s payload is a static method on the **same** class (`artifacts_dir(generating_process_name)`). The home of a process configuration is a static method on the process class. Do not put those directories on the instance. Do not put `artifacts_dir` on the process. JSON-only collections have `configs_dir` only.

### Keyword-only construction

Entity constructors, collection constructors, and process entry points are keyword-only (`process(generating_process_name=...)`).

### Preallocate when `n` is known

No `list.append` in those loops.

### Package layout matches the three objects

Entity and collection modules at the level root. Process modules in a process package (ABC, strategies, oracle, named recipe). Imports name the module.

### A generating_process_name and a process configuration are claimed exclusively

The process class owns two config folders: `human_readable/` and `hashes/`. `assert_generating_process_name_available` checks `Collection.artifacts_dir(generating_process_name)` (if file-backed) and both process-configuration files. It reads the documents; it does not trust the filename. `mkdir` without overwrite.

An incomplete run deletes leftover payload, the collection configuration it started, and any process-configuration files, so the same `generating_process_name` can be retried. The process configuration is the completion record. A finished `generating_process_name` or a finished `generating_process_signature` is not overwritten by `process()`. No `--force` on `process()`. A cleanup helper may delete those files first (`remove_named_freeze`, `process_hash_occupant`); that helper is not `process()`.

### Scripts are thin

Load a named recipe, call `process(generating_process_name=...)`, print hash / artifact directory / config path. No identity logic in the CLI.

### No backward compatibility unless the user asks

Old types do not shape the new types.
