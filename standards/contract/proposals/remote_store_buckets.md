# Remote-store bucket names & creation (proposal)

Short companion to `remote_store.md`. Two concerns: (1) how a project name
becomes a bucket name, (2) when a bucket may be created.

## 1. Bucket name from project name

Deliberately strict and small, so it is trivial to implement, test, and debug.

### What counts as a valid bucket name

A bucket name is valid iff it matches:

```
^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$
```

that is: 3–63 characters, only lowercase letters, digits, and `-`, starting and
ending with a letter or digit. No dots, no underscores, no uppercase. (This is a
strict subset of the S3/MinIO rule, chosen so there are no `..` / IPv4 / case
edge cases to reason about.)

The bucket name lives in `remote.json`, is user-editable, and is authoritative.
It is **validated on every use**; an invalid name fails fast against this regex
with a clear message — not deep inside a `PutObject` as an opaque
`InvalidBucketName`.

### The transformation T (project name → default bucket)

`T` runs **once, at scaffold time**, only to choose the default `remote.json`
bucket. It is never applied at runtime: the stored name is the truth, and
runtime only *validates* it.

```
T(name) = name.strip().lower().replace("_", "-")
```

Lowercase, and every `_` becomes `-`. Nothing else.

### The three categories

Project names are expected to be drawn from `[A-Za-z0-9_-]`.

- **Already valid** — the name is already a valid bucket (`T(name) == name`): used
  verbatim. e.g. `ccf-tabformer`, `fraud2024`.
- **Fixable** — not valid, but `T(name)` is: the default bucket is `T(name)`,
  deterministically. e.g. `ccf_tabformer → ccf-tabformer`,
  `CCF_Tabformer → ccf-tabformer`.
- **Not fixable** — `T(name)` is still invalid: it contains a character outside
  `[a-z0-9-]` after transform (e.g. a dot or space), is shorter than 3 or longer
  than 63, or starts/ends with `-`. No default is derived; scaffold writes
  `"bucket": ""`, and the first remote operation fails with "set a valid bucket
  name in remote.json". e.g. `ab`, `_tmp` (→ `-tmp`), `my.project`, a 70-char
  name.

## 2. Bucket creation

- **Only `push` may create a bucket.** `push_file` / `push_directory` ensure the
  (validly named) bucket exists — creating it if missing — then upload. Push is
  what establishes the remote truth, including its container.
- **`pull` and `exists` never create.** They only read; a missing bucket or
  object is reported or raised, never created. Pull is strictly read-only.
