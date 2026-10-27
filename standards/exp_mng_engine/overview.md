# exp_mng_e

The experiment management engine. Fit, logging, and inspect run together because they share **disk**, not a process and not RAM.

This folder is the contract for the engine. A project subclasses `ExperimentConfigManager`, edits scaffolded JSON, and constructs `DefaultExpEngine`.

Fit and inspect never share a process.

## Two folds of opinion

```mermaid
flowchart TB
  subgraph fold1 [Fold 1 - how it works]
    core["Engine core: roles + orchestration"]
    impl["Default manager logic: base / dl_engine"]
  end
  subgraph fold2 [Fold 2 - how a project turns it on]
    layout["Default layout: where files live"]
    parsers["Parsers: JSON to constructor fields"]
    scaffold["Scaffold: write those files"]
  end
  impl -->|"implements roles"| core
  layout --> parsers
  parsers -->|"construct default managers"| core
  scaffold --> layout
```

- **Fold 1** — uniqueness, catalogs, vendor stores, fetch protocol. Lives in `base/` and `dl_engine/`. No project tree. No recipe JSON. `dl_engine` never imports `experiment_management`.
- **Fold 2** — which JSON files exist, where they sit, how they become those objects. Lives in the **default engine**. A project edits scaffolded JSON or subclasses `ExperimentConfigManager`. It does not reimplement catalogs.

## Core vs default

```mermaid
flowchart TB
  subgraph core [Engine core - same for every engine]
    roles["Roles"]
    orch["get_new_run_or_overwrite / get_existing_run / train / inspect / fit_state"]
    session["RunSession"]
  end
  subgraph def [Default engine - same core, specified components]
    lay["Layout"]
    par["Parsers"]
    mk["Default managers from Fold 1"]
  end
  lay --> par
  par --> mk
  mk -->|"passes built managers"| roles
  roles --> orch
  orch --> session
```

**Core** does not know `identity.json` or `storage/runs`. It receives managers and injected paths. It defines the protocol between roles.

**Default** keeps that core. It only does layout and parsing. Constructor: `ExperimentConfigManager`.

```python
ExpEngine(identity, logging, checkpoint, experiment, paths, compose_toolbox, tracking_backend)

DefaultExpEngine(project_name, start=..., experiment=...)
```

## Epoch vs rank

Epoch is the only absolute checkpoint index. Rank is a sort of currently retained files **at the moment of the call**. Inspect snapshots are stored under epoch, even if the query was `rank=1`.

## Files

| Folder | Job |
|---|---|
| [core/](core/engine.md) | `ExpEngine`, roles, `RunSession`, fit state |
| [checkpoint/](checkpoint/protocol.md) | Lock, save, prune, fetch |
| [inspect/](inspect/inspect.md) | Request → snapshot → exclusive folder |
| [identity/](identity/identity.md) | `(name, hash)` pair; lookup / claim / release |
| [identity/get_new_run_or_overwrite.md](identity/get_new_run_or_overwrite.md) | Decision tree for `get_new_run_or_overwrite` |
| [logging/](logging/logging.md) | Store vs run; inspect logs through Logger |
| [default/](default/engine.md) | `DefaultExpEngine` + `default_layout` (layout + parsers) |
