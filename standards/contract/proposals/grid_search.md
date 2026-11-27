# Grid search wrapper (proposal)

Draft. Feature `.mdc` later: `.cursor/rules/dl_toolbox_structure/dl_engine/exp_engine/grid_search.mdc`.
Code: `src/dl_toolbox/dl_engine/exp_engine/grid_search/`. Tests: `tests/dl_engine/exp_engine/grid_search/`.
GPU query: `src/dl_toolbox/dl_engine/dl_engine_utils/visible_gpus.py`.

---

## Context

A project overfit wrapper used to copy three steps: cartesian enumeration over `experiment_config`, validation of each cell, optional filtering of legal-but-uninteresting cells. Kept members were written as `exp_{i}.json` plus `index.json`, then launched one child train process per free GPU.

That logic is not a Lightning run. It belongs in `exp_engine/grid_search`. Physical GPU query and `CUDA_VISIBLE_DEVICES` binding belong in `dl_engine_utils/visible_gpus.py` so engines and the wrapper share one pool.

## Terminology

- **enumerate**: cartesian product of axis `param_values` (explicit lists, or `logspace:` / `linspace:` strings).
- **validate**: project parser. Illegal → skip reason `invalid`.
- **filter**: optional `keep`. Legal but uninteresting → skip reason `uninteresting`.
- **persist**: write keepers and the index. Skips stay in the index only.
- **GPU batch**: one child per free GPU. Parent does not train. `query_free_gpu_indices` is the floor-filtered pool.

## Shape

`GridSearchDocument` keys: `starting_recipe`, `grid_search_exp_name`, `grid_search_config`. Each axis is `param_name` (dotted path into `experiment_config`) and `param_values`.

`write_grid_search` takes the starting recipe dict, the family Directory, `validate`, `keep`, and `stamp_recipe`. The project stamps `run_group` and `human_readable_name`.

`run_recipes_on_free_gpus` takes an injected `query_free` and `popen`. The CLI builds `query_free` from `query_free_gpu_indices` and `gib_to_mib`.
