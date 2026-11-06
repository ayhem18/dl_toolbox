# ExclusiveDirLock (proposal, deferred)

File and Directory are implemented. Their contracts are:

- `.cursor/rules/dl_toolbox_structure/base/file_management/file.mdc`
- `.cursor/rules/dl_toolbox_structure/base/file_management/directory.mdc`

`ExclusiveDirLock` stays a separate contract. It is not a File or a Directory.

**Pays for:** two processes writing the same catalog.

Takes a Directory (lock path). `timeout_seconds` required. Atomic `mkdir` / `rmdir`. Crash leaves the directory.

Existing code: `base/file_management/exclusive_lock.py`. Do not fold it into File/Directory. Revisit later.
