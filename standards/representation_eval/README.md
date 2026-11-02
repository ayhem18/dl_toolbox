# Representation eval

Frozen embeddings, then kNN / linear probe / 2D projections. Not a training loop and not an `*ExpEngine`.

## Toolbox

`dl_toolbox.dl_engine.representation_eval` with subpackages `common`, `knn`, `linear_probe`, `transformations`. Shared helpers (Lightning `net.` load, pred metrics, neighbor sheets) live in `dl_toolbox.dl_engine_utils`.

The project supplies `encode_fn`, label mapping, and image reload. Single-number metrics are returned as dicts for `Logger.log_scalars`. Disk gets PNGs and structured JSON only.

## Project

`InspectFn` loads `record.path`, extracts train/val banks, calls `run_representation_probes`, then `log_scalars` / `log_image` / `log_table` at `step=epoch`.
