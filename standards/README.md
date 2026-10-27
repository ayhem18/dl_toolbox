# Standards

Contracts a project can adopt before an experiment is worth comparing: metric definitions, split rules, what must be logged, and the bar a result has to clear.

These are documents. They are not installed with `dl-toolbox`. When a contract becomes executable (a metric, a check, a fixture), that code moves to `src/dl_toolbox` and the standard links to it.

| Folder | Job |
|---|---|
| [exp_mng_engine](exp_mng_engine/overview.md) | Experiment engine: core vs default, owned checkpoint protocol, fit/inspect via disk. |
| [metrics](metrics/README.md) | Batch measurement, combiner, reducer. |
| [data_contract](data_contract/README.md) | Entity, collection, process, lock. |

## Template

Copy this heading block into a new file, one standard per file.

```markdown
# <name>

## Applies to
Which task shape this covers (for example binary semantic segmentation with sparse foreground).

## Protocol
What is fixed before a run is comparable: split, resolution, threshold, aggregation.

## Report
The numbers and artifacts a run must leave behind.

## Pass bar
The condition under which a change is kept.
```
