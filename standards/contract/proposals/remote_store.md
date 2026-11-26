# RemoteStore (proposal)

Draft. Not a `.mdc` yet. Mark `CONTRACT` and mirror this into
`.cursor/rules/dl_toolbox_structure/remote_store/` only once the definition in
`standards/contract/philosophy.md` is met in code.

## Context

One of the ML servers this work runs on is now reachable only through a
jumpserver, and the usual protocols (sftp, rsync, direct git over the old
remote) are blocked or rewritten before they reach it. The old loop — develop
locally, `git push`, pull on the server, run, `rsync` the artifacts and
checkpoints back — no longer works without routing every change through the
security team. So we stop fighting that boundary and introduce a storage layer
we control: a MinIO (S3-compatible) object store on a server we *can* reach,
used as the single source of truth for data, checkpoints, and run artifacts.
Hugging Face is a second backend of the same shape for the public, dataset-like
pieces.

The package already knows how to give a thing an **identity**: `HashManager`
turns a configuration into a `(human_readable_name, hash)` pair, and the data
contract's first assumption is that *once a piece of data has an identifier, its
content does not change*. `ProjectLayout` already answers "where does this live
on disk" without any caller reconstructing the tree. RemoteStore is the same two
questions pointed at a remote: an object is addressed by an **identity** (the
content hash, backend-independent) and a **location** (a backend-specific
namespace — for S3 a bucket and a key; for Hugging Face a repo and a path).

This layer does **not** re-decide identity, re-hash on every call, reconcile, or
detect conflicts. The remote is the source of truth. `push` overwrites the
remote; `pull` overwrites the local file. There is no immutability invariant to
defend here — that assumption governs the *data* layer, not this *transport*
layer. The only thing this layer protects is that a half-finished push does not
advertise itself as present (see the index invariant below).

Two abstractions carry it, mirroring `IdentityManager`(ABC)/`HashManager` and
`TrackingBackend`:

- `RemoteStoreCredentials` (ABC) — what you need to authenticate, plus one
  `connect()` that is the only surface that touches a vendor SDK or a secret.
- `RemoteStoreConnection` (ABC) — the live connection; it carries every
  operation (`object_exists`, `push_file`/`pull_file`,
  `push_directory`/`pull_directory`).

On top of those sits `ProjectRemoteMirror`: the project-aware layer that owns the
mapping between the local `storage/` tree and the remote. It fixes the bucket at
the project level, derives identity from the file bytes through `HashManager`,
and speaks only in paths relative to `storage/`. The raw `RemoteStoreConnection`
knows backends and nothing about dl_toolbox; the mirror knows the project and
nothing about boto3. (Proposed name: `ProjectRemoteMirror`, composing a
connection — not a per-backend `ProjectLayoutS3Connection` subclass, because the
mirroring logic is identical across backends and only the transport differs.)

## Terminology

Directory, File, JSON `read_object`/`write_object`, HashManager, ProjectLayout:
their own contracts. This section names only what is specific to RemoteStore.

| Term | Meaning |
|---|---|
| `identity` | the content address of one object: the full 64-char lowercase SHA-256 of its bytes. Backend-independent. Minted on push via `HashManager`/`identity.hash_file`; read from the identity document on exists/pull |
| `location` | where an object lives on one backend, as a JSON object (dict) parsed by that backend's `...Location` model. Backend-specific keys |
| `RemoteBackend` | the backend names: `RemoteBackend.S3`, `RemoteBackend.HUGGINGFACE`, `RemoteBackend.BACKENDS`. Class variables, like `TrackingBackend` |
| `S3Location` | `{bucket, path_within_bucket}` |
| `HuggingFaceLocation` | `{repo_id, repo_type, path_in_repo}` |
| `RemoteStoreCredentials` | ABC: `backend`, `connect()`. The only place a secret or vendor SDK is touched |
| `S3Credentials` | endpoint, access key, secret key, region, TLS, addressing style. Keys held as `SecretStr` so repr/str redact |
| `RemoteStoreConnection` | ABC: the live connection and every operation |
| `S3Connection` | the boto3/MinIO implementation |
| `object_exists` | does the object at `(identity, location)` exist. **File-only.** Reads the index entry (a single object key), not a payload tree |
| `push_file` / `pull_file` | overwrite the remote object from a local file / overwrite the local file from the remote object |
| `push_directory` / `pull_directory` | transfer a tree under a location prefix. No `object_exists` for a tree |
| `ProjectRemoteMirror` | project-aware layer: composes a `RemoteStoreConnection`, fixes the bucket at project level, derives identity from file bytes, speaks paths relative to `storage/` |
| index entry | the single object whose presence means "this identity is completely stored". Written **last**. For a directory it is the identity/configuration file the data contract already writes last |

## Input Specification

`identity` is a full 64-char lowercase SHA-256 (`identity.is_full_sha256`).

`location` is a JSON object (dict). Each backend parses it with its Pydantic
`...Location` model (`extra="forbid"`, `frozen`, `strict`), raising `ValueError`
on a bad shape — never a hand-rolled dict check. `S3Location`: `bucket` and
`path_within_bucket` are non-empty strings; `path_within_bucket` is a posix key
with no `..` segment and no leading `/`.

Local ends are `base.file_management` objects, so this layer never re-checks
absoluteness or existence. `push_file` takes a `File` (an existing absolute
file) as `source`; `pull_file` takes a `destination` `Directory` (exists) and a
`name`, and returns the written `File`. `push_directory` takes a `Directory`
(exists) as `source`; `pull_directory` takes a `destination` `Directory`
(exists). Nested output folders are created through `Directory.from_under`, and
object keys resolve with `resolve_under_project` — the shared helpers, not a
hand-rolled `..`/leading-`/` guard.

`S3Credentials` (connect-time): `endpoint_url` (non-empty; MinIO's URL),
`access_key_id` and `secret_access_key` (`SecretStr`), `region` (non-empty, e.g.
`us-east-1`), `secure` (bool), `addressing_style` (`"path"` for MinIO). It is
built from two on-disk halves (below), never stored whole.

On-disk, under `project_level_configs/remote/`: a committed `remote.json` (the
non-secret config — `backend`, `bucket`, and for S3 `endpoint_url`, `region`,
`secure`, `addressing_style`, parsed by `S3RemoteDocument`) and a
`credentials/<backend>.json` secret file per backend (`S3CredentialsFile`:
`access_key_id`, `secret_access_key`; both non-empty at connect time).
`scaffold` writes a blank credential file for **every** backend so the keys are
never a guess, and `.gitignore` excludes `remote/credentials/`.

`ProjectRemoteMirror`: a `RemoteStoreConnection`, a `HashManager`, the project's
`storage/` Directory (must exist), and a `bucket` (non-empty, from
`remote.json`). Every path argument is relative to `storage/`, with no `..` and
no leading `/`; it resolves under `storage/`.

## Main functionalities

### `RemoteStoreCredentials.connect()` → `RemoteStoreConnection`

The only call that authenticates. Lazily imports the backend SDK through
`base.packaging.require` (so importing the package never requires boto3), builds
the client, and returns a live `RemoteStoreConnection`. Auth failure surfaces
here as a `ConnectionError` with a **redacted** message (no key bytes).

### `RemoteStoreConnection.object_exists(*, identity, location)` → `bool`

Does the single object at `(identity, location)` exist. **File-only**: the
caller never passes a directory prefix. Returns `True`/`False`; it does not
download the object. This is the resolve-or-fetch gate ("pull it, or produce
it"). On S3: `head_object` on the key; a 404 is `False`, other errors raise.

### `push_file(*, identity, location, source)` → `None`

Overwrite the remote object from the `source` `File`, unconditionally. On S3:
`upload_file` to the key, recording `identity` as object metadata. Side effect:
the remote object is replaced.

### `pull_file(*, identity, location, destination, name)` → `File`

Overwrite `name` under the `destination` `Directory` from the remote object,
unconditionally, and return the written `File`. Side effect: the local file is
replaced.

### `push_directory(*, identity, location, source)` / `pull_directory(..., destination)`

Transfer a tree under the location prefix. `push_directory` walks the `source`
`Directory` and uploads each file under `path_within_bucket + "/" + <relative
posix>`. `pull_directory` lists the prefix and downloads each object into
`destination`, recreating subdirectories (via `Directory.from_under`) and
overwriting. No `object_exists` for a tree; a tree's presence
is decided by `object_exists` on the single identity/configuration **file** that
the data contract writes last.

### `ProjectRemoteMirror` — push/pull/exists by path within `storage/`

`push_file(*, local_file)`: `identity = identity.hash_file(local_file.path)`
(the bytes are in hand); `location = {bucket, path_within_bucket=<rel to
storage>}`; delegate to the connection.

`object_exists(*, relative_path, expected_hash)` and
`pull_file(*, relative_path, expected_hash)`: `expected_hash` comes from the
identity document the caller already holds (the bytes are **not** in hand before
a pull). `pull_file` re-hashes the downloaded bytes and raises if they do not
equal `expected_hash` before the file is accepted.

`push_directory` / `pull_directory`: the tree variants, by relative directory
path.

## Secondary functionalities

- `RemoteBackend.S3` / `HUGGINGFACE` / `BACKENDS` — the backend names, class
  variables; `_parse_backend` validates with `parse_one_of`.
- `_require_s3_location(location)` → `S3Location` — `model_validate`, wrapping
  `ValidationError` as `ValueError`.
- `_object_key(location)` / `_prefix(location)` — the one place a location
  becomes an S3 key/prefix. **Shared** by `object_exists`, `push_*`, `pull_*`, so
  the write key and the check key can never drift (the `prefix_len` lesson from
  HashManager).
- normalizing a local path to a `path_within_bucket` reuses
  `resolve_under_project` (bounds under `storage/`, rejects `..`/absolute); the
  mirror does not hand-roll that guard.

## Tensions

- **No atomic multi-object commit on S3.** A crashed `push_directory` leaves a
  partial prefix. The guard is write-order at the mirror: push the payload tree
  first, the identity/configuration file **last**. `object_exists` checks that
  last file, so a half-finished push reads as absent and a retry completes it.
- **Hugging Face is a second backend, not yet implemented here.** The ABCs and
  `HuggingFaceLocation` exist so it slots in; `HuggingFaceCredentials` /
  `HuggingFaceConnection` are deferred. HF repos are git+LFS, so a directory push
  is one atomic commit there — the opposite of S3, and worth revisiting the
  write-order guard per backend.
- **Remote GC vs local `drop_pair`.** Dropping a pair locally does not un-mirror
  it. The remote is append-mostly; deletion is a separate explicit op, never a
  side effect of the mirror.
- **Credentials placement.** Secrets live in `remote/credentials/<backend>.json`
  inside the repo, git-ignored, and outside `storage/` so the mirror never
  pushes them. The non-secret config (`remote.json`) stays committed — the split
  keeps endpoint/bucket/region versioned while quarantining only the secret. The
  residual risk (anyone with filesystem access reads the creds file) is inherent
  to a local credential file.
- **`ProjectRemoteMirror` name and shape.** Chosen over `ProjectLayoutS3Connection`:
  backend-neutral, composes a connection. The one backend-specific thing left is
  that the mirror builds an S3-shaped `location` dict (see the note in
  `project_mirror.py`).
- **Wiring into `ProjectLayout` (done).** `scaffold` writes `remote/remote.json`
  plus a blank `remote/credentials/<backend>.json` for every backend and the
  `.gitignore` entry; `setup-project --remote-backend` selects which backend
  `remote.json` records. `from_start` stays unchanged (no remote read, so old
  projects still load); `ProjectLayout.remote_mirror()` reads both halves
  lazily, validates, connects, and returns the mirror. Only S3 is wired —
  `remote_mirror()` refuses any other `backend` with a clear message. The CLI
  exposes `remote-push PATH [--hash]`, `remote-pull PATH --hash [--dir]`, and
  `remote-exists PATH --hash`: each locates the project from cwd (reading
  `.tracking-backend` for the backend it does not itself use), builds the
  mirror, and operates on `PATH` under `storage/`. A file push prints the
  computed identity; pull verifies it; a directory needs `--hash` (its identity
  is not a single file's content).
- **`remote.json` shape is per-backend.** Only the S3 shape has a document
  (`S3RemoteDocument`) today; a Hugging Face `remote.json` is scaffolded as a
  placeholder, and the read logic must grow a branch when that backend lands.
