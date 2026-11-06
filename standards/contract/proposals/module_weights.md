# ModuleWeights (proposal)

Python-object entity. Home after implementation: `dl_engine/dl_engine_utils/checkpointing/` next to the catalog checkpointer, not inside it. A folder may hold several contracts. The catalog is a directory. This is the tensors of one `nn.Module`.

**Pays for:** `write_weights` and `load_lightning_net_weights` are two halves of one object, living in two packages. Inspect re-derives `net.` and silently drops every key that does not start with that prefix. `write.py` already documents the coupling.

Draft. Do not mark `CONTRACT`. Do not write the feature `.mdc` until the six objects exist in code and call sites stop calling `torch.load` / stripping prefixes.

---

## Entity

`ModuleWeights` = `state_dict` (`Mapping[str, Tensor]`) + `KeyLayout`.

It does not subclass `dict`. Later code holds this object. It does not hold a raw `state_dict` and remember “these keys still have `net.`” in a comment.

Not this entity:

- The checkpoint **directory** (`InspectableCheckpointer`, `catalog.json`, top-k, lock). Fetch returns a **File**. This contract starts after that File exists.
- Optimizer / epoch / callbacks inside a Lightning trainer payload. Parse from that file **extracts** `state_dict`. The rest is ignored by name, not stored.
- A Lightning `Trainer`. `trainer.save_checkpoint` is a **source**. It is not a write of this entity.

---

## Spec

```
KeyLayout = bare | lightning_net
```

Named constants. No default. No free `prefix: str` slot.

| Layout | Keys |
|---|---|
| `bare` | `nn.Module.state_dict()` of the net itself (`fc.weight`, `encoder.stem.weight`, …) |
| `lightning_net` | `ClassificationLightningModule` attribute `net`: every kept key starts with `net.` |

`lightning_net` is that module’s attribute name, not a law of Lightning. A later wrapper that stores the net as `model` is a **new named layout**, not an argument.

Illegal: putting file kind (`weights_only` vs `lightning_trainer`) on the spec. That is parse. After parse, the entity is tensors plus key layout. Same rule as Image not storing “came from a PNG”.

Illegal: putting `map_location` on the spec. Device is where `torch.load` materialized the tensors. It is a parse argument.

---

## Validation (once)

**Spec consistency:** the combination is one of the two layouts.

**Payload matches spec:**

- keys are non-empty `str`; values are `Tensor`
- `bare`: no further key syntax
- `lightning_net`: every key starts with `net.` ; empty after that stem is an error

What the payload cannot prove: that `net.` means `ClassificationLightningModule.net` rather than a module that happened to name a child `net`. Layout is a claim. Validation checks consistency of the claim.

`load_state_dict(..., strict=True)` is payload-matches-**module**, at `load_into`. Not at parse. A file may hold extra keys until convert.

Call sites do not repeat these checks. They do not loop keys looking for `net.`.

---

## Sources

| Source | Caller must pass | Source assigns |
|---|---|---|
| `nn.Module` (the net) | nothing about keys | `bare` |
| File, weights-only (`{"state_dict": ...}`) | `layout`, `map_location` | tensors from disk |
| File, Lightning trainer checkpoint | `layout`, `map_location` | `payload["state_dict"]`, then that layout |

No `from_anything`. No “if `state_dict` in dict else the dict is the state” at the call site — that branch is the trainer-file parse API.

`ClassificationLightningModule.state_dict()` in memory is `lightning_net`. That is `from_module(lightning_module)` only if we name it. Prefer: take the **net** (`lightning_module.net`) as the `nn.Module` source (`bare`), or take the **File** after fit wrote it (`lightning_net`). Do not invent a third in-memory dump that re-introduces the prefix as a surprise.

---

## Parse APIs

Required arguments are what that source cannot supply.

### `from_module(*, module)`

Assume: `module` is the net (`nn.Module`), not the Lightning wrapper.

```
state = module.state_dict()
return ModuleWeights(state_dict=state, layout=bare)
```

After: in-memory copy. Disk unchanged.

### `from_weights_file(*, file, layout, map_location)`

Assume: `file` is a File. Contents are `{"state_dict": <dict of tensors>}` (what `write` emits). `layout` required. `map_location` required.

```
payload = torch.load(file, map_location=map_location, weights_only=False)
if payload is not a dict or "state_dict" not in payload: error
state = payload["state_dict"]
validate(state, layout)
return ModuleWeights(state_dict=state, layout=layout)
```

After: entity in memory. File unchanged.

### `from_trainer_file(*, file, layout, map_location)`

Assume: `file` is a File written by `Trainer.save_checkpoint` (what inspect actually fetches today).

```
payload = torch.load(...)
if payload is not a dict or "state_dict" not in payload: error
state = payload["state_dict"]
validate(state, layout)
return ModuleWeights(state_dict=state, layout=layout)
```

After: weights only. Optimizer / epoch / hyper_parameters in that file are not on the entity.

Today `load_lightning_net_weights` accepts both file kinds with one `if`. That hides two sources. Two parse APIs.

### Inverse: `write(*, directory, name, weights)`

Assume: `directory` is a Directory that exists. `name` is a relative path with no `..` (same rule as `File.from_under`). `weights` is `ModuleWeights`. JSON already owns this disk shape.

```
path = join under directory / name
torch.save({"state_dict": dict(weights.state_dict)}, path)
return File.from_absolute(path)
```

After: a File. Directory unchanged. **No mkdir.** Creating a missing parent is Directory parse (`CREATE`).

This write is **weights-only**. It is not `Trainer.save_checkpoint`. Fit under DDP still uses the trainer as the write_function of the **catalog** contract (barrier). Inspect that must load the net uses this entity’s parse, not a second `torch.load`.

`weights_writer` is a closure over an in-memory `ModuleWeights` for tests. Not a third format.

---

## Convert

`convert(weights, target_layout)`:

| From | To | Map |
|---|---|---|
| `bare` → `bare`, `lightning_net` → `lightning_net` | identity |
| `bare` → `lightning_net` | prefix every key with `net.` |
| `lightning_net` → `bare` | strip `net.` from every key; error if a key does not start with it |

No silent drop of non-matching keys. Today’s loader builds `stripped` only from keys that start with `net.` and ignores the rest. That is a convert without a name, and it is wrong: a `bare` file fails with “no `net.` weights”; a mixed dict loads a subset.

### `load_into(*, module)`

Assume: `weights.layout` is `bare`. `module` is that net.

```
missing, unexpected = module.load_state_dict(weights.state_dict, strict=True)
if missing or unexpected: error
```

If the entity is `lightning_net`, convert first. Call sites do not strip.

---

## Translation (after code exists)

| Object | Name |
|---|---|
| Entity | `ModuleWeights` |
| Spec | `KeyLayout.BARE`, `KeyLayout.LIGHTNING_NET` |
| Parse | `from_module`, `from_weights_file`, `from_trainer_file`, `write` |
| Convert | `convert(layout=...)`, `load_into(module=...)` |

Replace `dl_engine_utils.checkpoint.lightning.load_lightning_net_weights` and `dl_engine.dl_engine_utils.checkpointing.write.write_weights`. Digital Lab `val_inspect.checkpoint.load_net_weights` becomes parse + convert + `load_into`. Catalog lookup stays `InspectableCheckpointer.fetch` (or the old binding until that swap). Path joining in `rank1_checkpoint` is File, not this contract.

---

## Tensions

1. **Two file kinds.** Fit writes a trainer checkpoint. Tests write weights-only. One `torch.load` that sniffs `state_dict` is two sources pretending to share an API. Keep both parse APIs. Do not add `kind=None` that auto-detects.

2. **`net.` is a claim.** Hardcoding it in a helper made inspect work for `ClassificationLightningModule` and nowhere else. A free `prefix` argument makes illegal layouts representable (`""`, `"model."`, `"net"` without the dot). Named constant only. When the Lightning wrapper changes attribute, add a layout, do not thread a string through inspect scripts.

3. **Silent key drop.** The leftover loader drops every key that does not start with `net.`. That must become convert-or-error. If Lightning puts non-`net.` keys in `state_dict` (buffers on the wrapper), `from_trainer_file` with `lightning_net` should error until we name a filter. Do not hide the filter as `if key.startswith`.

4. **Catalog vs bytes.** `rank1_checkpoint` re-walks `catalog.json` and joins relative paths by hand. That is File + checkpointer fetch, already specified. This contract must not grow a `load_rank1(run_dir)` that opens the catalog.

5. **`map_location`.** Required on file parse. Not a default `cpu`. Not stored on the spec.

6. **`weights_only=False`.** Lightning checkpoints are not tensor-only pickles. That is a property of the trainer-file source. Do not pretend `weights_only=True` is the safe default there.

7. **Ceremony.** If after this exists, inspect still does `torch.load` and a `net.` loop, the wrapper did not pay. Gate: those two call sites plus `write_weights` use the entity.
