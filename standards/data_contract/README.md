# Data contract

The mental framework is [philosophy.md](philosophy.md) and [philosophy_in_code.md](philosophy_in_code.md). Copy those two files into a project’s documentation so that project stays self-contained.

Executable helpers are `dl_toolbox.data_contract`. They are not a data product: they do not define a frame, an item, or a lock.

Drafts for those helpers: [collection configuration](../contract/proposals/collection_configuration.md), [process records](../contract/proposals/process_records.md). Disk is File and Directory. JSON is `read_object` / `write_object`. The process pair is HashManager.

## Toolbox

| Module | Job |
|---|---|
| `names` | `generating_process_name` is HashManager `validate_human_readable_name`. |
| `configs` | `write_collection_configuration`, `load_collection_configuration`, `delete_collection_configuration` under the collection kind’s Directory. |
| `process_records` | `claim_generating_process`, `create_payload_directory`, `write_process_configuration`, `delete_incomplete_run`. |
| `named_freeze` | `remove_named_freeze`, `process_hash_occupant`. Not `process()`. |
