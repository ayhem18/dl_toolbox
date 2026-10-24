# Philosophy in code

Design lives in [philosophy.md](philosophy.md). This note is how to write the Python so that design stays readable, and so it is obvious when the *design* is the thing that must change.

The rules below are **style**. They are not a second contract. They are also **not optional**. Follow them. If the next implementation is forced to break one, **stop and warn**. Do not drop the rule in the code. Do not decide it does not matter. The user decides whether to change the writing or the design.

If a **MetricObject** cannot be written without a default, a `None`, or a reduced scalar where [philosophy.md](philosophy.md) requires a batch measurement, that is a **design** stop: revise [philosophy.md](philosophy.md). Do not weaken the design to make an implementation fit.

## When to stop

### Design — revise [philosophy.md](philosophy.md)

Only these are design problems, and only on **MetricObject** (and the tensor contract it names):

- `metric_function` returns a scalar `()` or a dummy `[reduced_value, batch_count]`
- `metric_function` averages or sums away the item axis
- `combiner` adds already-reduced batch scalars (average of averages)
- a reducer returns more than one number
- the accumulator interprets the batch (reads `output` / `target` itself)
- a metric accumulates; an accumulator defines a new measurement
- required tensor keys missing from the mapping, filled with a default
- a classification helper whose grain is “one row per batch” without the level note saying so

### Style — stop and warn

If the implementation cannot follow a rule in **How the Python should look** without breaking it, stop. Say which rule and why it cannot be followed. Do not proceed. Do not treat the rule as optional.

## How the Python should look

### MetricObject is the named measurement, not the fold

`MetricObject` is a frozen dataclass: `name`, `required_keys`, `metric_function`, `combiner`, `reducers`. No running state. No split. No log names.

`MetricAccumulator` is a usual class: one split, one combined tensor per metric name, `update` / `epoch_scalars` / `reset`. It does not own reducer implementations.

### Metric function keeps the item axis

The return of `metric_function` has leading dimension `B` matching the batch (or the item axis documented on that metric).

```text
legal:   (B,), (B, 2), (B, 4)
illegal: (), (2,) as [batch_mean, 1]
```

For losses, use per-item reduction (`reduction="none"`, then mean over the item’s own logits/pixels if needed), not `F.cross_entropy(..., reduction="mean")` as the measurement.

Accuracy that needs a proper micro average stores **per-item counts** `(B, 2)` or equivalent, not a single `[correct, total]` for the whole batch unless the combiner concatenates that as **one row per batch** and the metric note has named batch as the grain (it has not). Default grain is the item.

### Combiner concatenates batches

`combiner` is `torch.cat` on dim 0 after both sides are rows of the same width. Do not `+` two scalars. Do not `stack` a mean with a one.

If a metric truly needs a running sum of counts and never the per-item vector, say so in [philosophy.md](philosophy.md) first. Concatenate-then-sum-in-the-reducer is the default: it keeps batch and epoch reducers the same function.

### Reducer in, scalar out

Each reducer is `Callable[[Tensor], Tensor]` and the tensor has `numel() == 1`. The accumulator is allowed to reject anything else.

Mean of a `(B,)` loss vector is `values.mean()` (over items, not over a list of batch means). Ratio of stacked counts sums the columns first, then divides.

### MetricSpec only names which reducers fire

`MetricSpec = (MetricObject, batch_reducer_names, epoch_reducer_names)`. Those names are keys into `metric.reducers`. The spec does not wrap a second function.

Log names stay `name/split/level/reducer` with `level` in `{batch, epoch}`.

### Keys are sliced, then the metric runs

The accumulator calls the existing tensor-mapping helpers: required keys present, values are tensors, then `metric_function(sliced)`. Classification helpers use `output` and `target`. A new task names its own keys; it does not teach the accumulator new batch layouts.

### No defaults on MetricObject fields

Every field is required. An empty `reducers` mapping is allowed only if the spec never asks for a reducer. A missing reducer name in a spec is an error, not a skip.

This rule does not apply to Lightning / logging.

### Package layout matches the two objects

- `metric_object.py` — `MetricObject`, `MetricSpec`, log-name helper
- `accumulator.py` — the fold
- `contracts.py` — tensor keys and mapping checks
- `classification/` — task helpers that obey the grain rules

Imports name the module. Classification factories do not live in `accumulator.py`.

### Scripts and the training module are thin

The Lightning module builds the tensor mapping, calls `accumulator.update`, and logs the dicts the accumulator returns. It does not implement combine or reduce.

### No backward compatibility unless the user asks

Old helpers that return `[batch_mean, 1]` do not shape the new helpers.
