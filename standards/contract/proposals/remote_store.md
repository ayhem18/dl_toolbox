# RemoteStore (proposal)

Draft. Not a `.mdc` yet. Mark `CONTRACT` and mirror this into
`.cursor/rules/dl_toolbox_structure/remote_store/` only once the definition in
`standards/contract/philosophy.md` is met in code.

## Context

One ML server is now reachable only through a jumpserver, with sftp/rsync/direct
git blocked or rewritten. So we stop fighting that boundary and keep the source
of truth in object storage we control: a MinIO (S3-compatible) bucket on a server
we can reach. The loop becomes: on the ML server `push` checkpoints / the mlflow
store to the bucket; locally `pull` them back. Hugging Face is a second backend
of the same shape, for public dataset-like pieces (deferred).

The layer is deliberately small — a dumb, reliable mirror, not a
distributed-systems artifact. For a single object:

- **push** — copy the local bytes to the remote; the remote now holds the truth.
- **pull** — copy the remote bytes to the local path, overwriting.

An object is addressed by its **location** alone (a backend-specific dict; for S3
`{bucket, path_within_bucket}`). There is **no hashing and no verification in this
layer.** Content addressing, where a project wants it, lives in how the local
paths under `storage/` are named (`HashManager` names them); the mirror reflects
those paths verbatim. Identity never enters the remote-store layer.

A **directory** is not a real object — it is a key prefix, a convenience wrapper:
push overwrites every file under the prefix, pull downloads them. Object stores
have no atomic multi-object swap, so a directory push is made *completable*, not
transactional, by a **manifest** written strictly last (listing the files it
wrote); its presence marks the push done. A directory pull reads the manifest and
fetches exactly its files, so a half-finished push (manifest absent) is never
pulled as complete. Pulls are otherwise non-atomic by design — the remote truth is
never at risk, so a partial local pull is just re-run.

Two abstractions, mirroring `IdentityManager`(ABC)/`HashManager` and
`TrackingBackend`:

- `RemoteStoreCredentials` (ABC) — what you need to authenticate, plus the one
  `connect()` that touches a vendor SDK or a secret.
- `RemoteStoreConnection` (ABC) — the live connection and the five transport ops.

On top sits `ProjectRemoteMirror`: the default layout. It composes one connection,
fixes the bucket at the project level, and speaks only in paths relative to
`storage/`. The connection knows backends and nothing about dl_toolbox; the mirror
knows the project and nothing about boto3.

## Terminology

Directory, File, JSON, HashManager, ProjectLayout: their own contracts.

| Term | Meaning |
|---|---|
| `location` | where an object lives on one backend, a JSON object (dict) parsed by that backend's `...Location` model. The **only** address; no identity |
| `RemoteBackend` | `RemoteBackend.S3`, `.HUGGINGFACE`, `.BACKENDS` — class variables |
| `S3Location` | `{bucket, path_within_bucket}` |
| `RemoteStoreCredentials` / `S3Credentials` | ABC + S3 impl; `connect()` is the only secret/SDK surface; keys held as `SecretStr` |
| `RemoteStoreConnection` / `S3Connection` | ABC + boto3 impl; the five ops below |
| `exists` | is there a single object at this key. File-granular |
| `push_file` / `pull_file` | overwrite the remote object from a local file / overwrite the local file from the remote object |
| `push_directory` / `pull_directory` | overwrite / download every file under a key prefix |
| `DIRECTORY_MANIFEST_NAME` | `.dltoolbox-manifest.json`; the per-directory manifest, written last on push, required on pull |
| `ProjectRemoteMirror` | default layout: composes a connection, fixes the bucket, maps a `storage/`-relative path to a `location` |

## Input Specification

`location` is a dict, parsed by the backend's Pydantic `...Location` model
(`extra="forbid"`, `frozen`, `strict`). `S3Location`: `bucket`,
`path_within_bucket` non-empty; `path_within_bucket` a relative posix key (no
`..`, no leading `/`).

`S3Credentials` (connect-time): `endpoint_url`, `access_key_id`,
`secret_access_key` (`SecretStr`), `region`, `secure`, `addressing_style`. Built
from the committed `remote.json` (non-secret config) plus the git-ignored
`remote/credentials/<backend>.json` (secrets, non-empty at connect). `scaffold`
writes a blank credential file for every backend; `.gitignore` excludes
`remote/credentials/`.

Local ends are `base.file_management` objects: `push_file` takes a `File`;
`pull_file` a `destination` Directory + `name`; `push_directory`/`pull_directory`
a Directory. The connection re-checks nothing those types already guarantee.

`ProjectRemoteMirror`: a `RemoteStoreConnection`, the project `storage/` Directory
(exists), and a `bucket`. Every path is relative to `storage/`, no `..`, no
leading `/`; it resolves under `storage/` via `resolve_under_project`. No
`HashManager` — the mirror does not hash.

## Main functionalities

### `RemoteStoreConnection` (location-keyed, no identity)

- `exists(*, location) -> bool` — HEAD the key (S3: a 404 is `False`).
- `push_file(*, location, source: File)` — `upload_file` to the key.
- `pull_file(*, location, destination, name) -> File` — `download_file`, return it.
- `push_directory(*, location, source)` — upload each file under the prefix, then
  put the manifest `{files: [...]}` at `prefix + DIRECTORY_MANIFEST_NAME` **last**.
  A source file named `DIRECTORY_MANIFEST_NAME` is rejected (reserved).
- `pull_directory(*, location, destination)` — read the manifest (absent →
  `FileNotFoundError`), download each listed file, recreating subdirectories.

### `ProjectRemoteMirror` (by path within `storage/`)

`push_file(local_file) -> str` returns the storage-relative path; `pull_file`,
`exists`, `push_directory`, `pull_directory` all take `relative_path`. Each turns
the path into `{bucket, path_within_bucket}` and delegates. Nothing is hashed or
verified.

### CLI

`dl-toolbox remote-push PATH` (file or directory, inferred), `remote-pull PATH
[--dir]`, `remote-exists PATH`. Each locates the project from cwd (reading
`.tracking-backend` for the backend it does not itself use), builds the mirror,
and operates on `PATH` under `storage/`. No `--hash` anywhere.

## Secondary functionalities

- `RemoteBackend.S3` / `HUGGINGFACE` / `BACKENDS`; `parse_backend` via `parse_one_of`.
- `parse_s3_location` → `S3Location`, wrapping `ValidationError` as `ValueError`.
- `_object_key` / `_object_prefix` — the one place a location becomes an S3
  key/prefix, shared by every op.
- `S3CredentialsFile` / `HuggingFaceCredentialsFile`, `blank_credential_document`,
  `credential_file_model` — the on-disk secret shapes; blank templates derived
  from the models so scaffold and connect can't drift.

## Tensions

- **No atomic multi-object swap on S3.** A directory push is completable, not
  transactional: the manifest written last is the commit point. A crash before it
  leaves no manifest, so a pull refuses and you re-run. Hugging Face (git+LFS) has
  a real atomic commit — worth revisiting the manifest there.
- **No verification.** The layer trusts the remote (it *is* the truth) and the
  backend's transport checksums. If a project wants end-to-end integrity it lives
  one layer up, in the content-addressed path names, not here.
- **`exists` is file-only.** There is no directory-existence op; a directory's
  readiness is implied by `pull_directory` succeeding (manifest present). Add one
  if resolve-or-fetch for whole directories is ever needed.
- **Hugging Face deferred.** ABCs + `S3Location`'s sibling `HuggingFaceLocation`
  exist; `HuggingFaceCredentials`/`Connection` do not. `remote.json` shape is
  per-backend (`S3RemoteDocument` only); `remote_mirror()` refuses non-S3.
- **Credentials placement.** Secrets live in `remote/credentials/<backend>.json`
  in the repo, git-ignored, outside `storage/` so the mirror never pushes them;
  the non-secret `remote.json` stays committed.
- **`ProjectRemoteMirror` name/shape.** Composition over a
  `ProjectLayoutS3Connection` subclass; the one backend-specific line is building
  the `{bucket, path_within_bucket}` dict.
