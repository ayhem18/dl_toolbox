# What a contract is

This note is the definition. A feature `.mdc` is not a contract because it sits under `.cursor/rules/`. It is a contract when the objects below exist in code, and later call sites use them instead of re-checking assumptions.

Image is the reference. File-management helpers listed in a table are not.

If an entity can be built without going through parse, or a spec can hold an illegal combination, this note is not implemented. Revise the code or this note. Do not mark the folder `CONTRACT` in the structure index.

## Start from the entity

The itch is often a **functionality**: `cv2.imread` in every dataset, `abs_path` on every argument, `process_path` as a general helper.

The contract is the **entity** that functionality keeps reconstructing. Photometric IO became `Image`. Path processing becomes a path entity, not a more clever `process_path`.

A functionality is a smell. It is not the starting object.

## Scope

A contract is **one entity** and the objects around it. A folder may hold several contracts. Image and Mask share a directory and stay distinct. A lock is not a path. Do not fold every helper in `file_management/` into one entity so the folder can be marked `CONTRACT`.

A contract **pays for itself** when it removes duplicated checks *and* stops an assumption from being re-derived at call sites. If the wrapper is ceremony and callers still pass raw `Path`, it does not pay.

## Parse policy is not spec

Spec is what is true of the entity **now**. How it was obtained is parse. Image does not store “came from a PNG”. `CREATE` is a parse policy (missing directory → mkdir). After parse, the entity is a directory that exists. Putting `CREATE` on the living spec is a stale claim.

## Six objects

### 1. Entity = native payload + spec

The entity wraps a native Python object (`numpy.ndarray`, `pathlib.Path`) plus declared properties. It does not subclass the native type. Later code holds the entity. It does not hold the raw array or path and remember the properties in comments.

Image: `Image` is `ImageSpec` plus an ndarray. Spatial size lives on the array, not on the spec.

### 2. Spec = combination of properties

Each property answers one question (color mode, sample type, layout). The spec is their product. Dependent fields as free slots (`channels` next to `color_space`, `dir_ok` next to `file_ok`) make illegal states representable.

Named constants, not ad-hoc strings. No default spec: the caller names the combination.

Image: `ColorMode` × `SampleType` × `Layout`. Gray is `(Y_dim, X_dim)`. `FILE` + `CREATE` is the path equivalent of `uint8` + range `[0, 1]`: a combination the spec constructor must reject.

### 3. Validation is owned once

Two checks, both here:

- **Spec consistency:** the combination is legal.
- **Payload matches spec:** the native object is that combination.

What the payload cannot prove is still a claim. RGB vs BGR is not in the pixels. File vs directory is a role, not a guess after `is_file()`. Validation checks consistency of the claim; it does not invent the claim.

Call sites do not repeat these checks.

### 4. Sources are named

A source is an outside object the entity can be built from. Image: file, ndarray. Each source is a different question. Do not pretend they share one `from_anything`.

### 5. Parse APIs are owned, and the required arguments are the source contract

The module owns how a source becomes an entity. The arguments the caller **must** pass are the information that source cannot supply:

| Source | Caller must pass | Source assigns |
|---|---|---|
| ndarray | `spec` (RGB vs BGR is a claim) | nothing photometric |
| file | path only | OpenCV-native spec (gray uint8 or BGR uint8 YXC) |

The inverse belongs here too: entity → source (`ImageIO.write`). Per-source constraints live on that API (lossy codec, native spec for encode).

### 6. Conversion is named

`convert(entity, target_spec)` maps one legal spec to another. Each pair is a defined map or an error. Silent color swap, silent `/ 255`, silent `is_file() → parent` are conversions without a name.

`strict` is part of the map, not a default.

## When it is a contract

All of these are true:

- You can name the entity, the spec properties, the sources, the parse APIs, and the convert maps.
- A payload that skipped parse is a bug, not an alternative API.
- Later code does not re-validate the spec.
- The feature `.mdc` states those objects. It does not list modules.

Image meets this. Path work is not one folder-contract. File and Directory are separate contracts. ExclusiveDirLock is still a draft (`standards/contract/proposals/file_management.md`). A feature `.mdc` is written only when the six objects exist in code.
