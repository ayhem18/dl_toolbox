# Metrics philosophy

This note is the **design contract**. How the Python should look so that design stays visible is [philosophy_in_code.md](philosophy_in_code.md). Executable types live in `dl_toolbox.dl_engine.metrics`. A project still writes its own `metric_function`, combiner, and named reducers for the task.

If a `MetricObject` cannot be written without reducing a batch to a scalar *inside* `metric_function`, this contract is wrong or incomplete. Revise it here before writing more code.

Metrics are readable when every piece is one of two objects, a measurement is a **batch** of observations, and a scalar exists only after a **reducer**.

## Assumption

The training loop never interprets a batch. It calls `metric_function` on the tensors that metric named, folds the returned tensor with `combiner`, and asks named reducers for scalars. The loop does not know whether the metric is loss, accuracy, IoU, or Dice.

This contract does not own the optimizer loss. Backward uses a separate `loss_fn`. A loss **metric** is a measurement of the same family as accuracy: it is logged, combined, and reduced. It is not the backward tensor.

## What this is not

Experiment tracking (which logger, which monitor key, which checkpoint) is later context. It is not a level of this contract.

Input-key names (`output`, `target`) are a thin tensor contract so classification helpers can share keys. They are not the metric design.

## Design decisions

### 1. Two objects, never mixed

**MetricObject.** One named measurement. It holds:

- the tensor keys it needs
- `metric_function`: tensors of one batch → a **batch measurement**
- `combiner`: two batch measurements → one combined measurement of the same kind
- named `reducers`: a (combined) measurement → **one scalar**

It does not accumulate across batches. It does not know the split (`train` / `val`). It does not log. It does not choose which reducers fire at batch grain versus epoch grain.

**Accumulator.** The fold over one split of one epoch. It holds one combined tensor per metric name. On `update` it:

1. slices the incoming tensor mapping to that metric’s required keys
2. calls `metric_function` (one batch)
3. optionally asks batch reducers for scalars from **that batch measurement**
4. `combiner`s the batch measurement into the combined tensor

On epoch end it asks epoch reducers for scalars from the **combined** tensor. It then resets.

If a function is both objects at once, it will be hard to debug. If you can name which of the two you are holding, the design is readable.

### 2. The metric function operates on a batch and returns a batch

`metric_function` sees one batch: a mapping of tensors whose leading dimension is the batch (or an equivalent per-item axis the combiner agrees on).

It returns a **batch measurement**: a tensor whose leading dimension is that same item axis. Each row is one observation of this metric on one item (or one independently reducible unit the metric defines).

Examples of a legal batch measurement:

| Metric | Shape of one batch return | What one row is |
|---|---|---|
| Per-item CE / BCE | `(B,)` | mean loss on that item |
| Per-item accuracy counts | `(B, 2)` | `[n_correct, n_total]` on that item |
| Per-crop confusion | `(B, 4)` | `[tp, fp, fn, tn]` on that crop |
| Per-item Dice | `(B,)` | Dice on that item |

A scalar `()` or a length-2 stack `[batch_mean, 1]` is **not** a batch measurement. That is a reducer result (or a stand-in for “one batch happened”). Putting it in `metric_function` is how epoch loss becomes an **average of batch averages**.

The metric function does not:

- average over the batch
- count batches as the unit of averaging
- call a reducer
- mutate a running total

PyTorch `reduction="mean"` over the whole batch is a reducer, not a metric function. If the library call reduces, pass `reduction="none"` (or an equivalent) and keep the per-item axis.

### 3. Combined objects are batches

`combiner(previous, value)` joins two batch measurements of the **same kind**. The result is still a batch measurement: more rows, same row layout.

The default join is concatenate on the item axis (`dim=0`). Summing already-reduced batch scalars is not combining batches. It is combining reducer outputs, which this contract forbids.

After `N` batches of sizes `B_1 … B_N`, the combined tensor has `B_1 + … + B_N` rows. That is the epoch’s set of observations. Variable batch size is then automatic: a batch of 16 and a batch of 7 contribute 16 and 7 rows. They do not contribute two equally weighted means.

Accuracy already wanted this (counts that can be summed **per item**, then reduced). Loss must use the same grain. The current classification loss metric stacks `[mean_over_batch, 1]` and sums those stacks; that is the average-of-averages bug. The design failure is the grain of the measurement, not the accumulator.

### 4. The reducer operates on a (combined) batch and returns a scalar

A reducer reads only a batch measurement (one batch, or the combined epoch tensor) and returns a **rank-0** tensor: one number.

The same reducer is legal at both grains:

- **batch grain:** the argument is this batch’s measurement (`B` rows)
- **epoch grain:** the argument is the combined measurement (`sum B` rows)

The reducer does not know which grain it is. `MetricSpec` names which reducer keys fire at which grain. The accumulator applies them. The metric object does not.

A reducer may collapse rows (mean, p95, worst-k) and may collapse a row’s internals (correct / total, tp / (tp+fp+fn)). It may not return a vector of per-item scores. Those scores are the measurement; the reducer’s job is the scalar.

If two numbers are needed (mean and p95), that is two named reducers, not one reducer that returns two values.

### 5. Grain is the unit of averaging

Name the unit before writing the function.

- **Item grain** (default for classification): one row per example. Epoch mean loss is the mean of per-item losses. Epoch accuracy is total correct / total labeled items (or per-item then mean — say which).
- **Pixel / token grain** (allowed when the metric says so): one row may still be an item, but the row stores counts, and the reducer sums counts across rows. That is micro-averaging. It is not an average of per-item ratios unless a reducer named that way computes it.

Do not mix grains in one `MetricObject`. A mean of per-item Dice and a micro IoU from stacked confusion counts are two metrics, or two reducers on a count-valued measurement.

Equal batch sizes do not make average-of-batch-means equal to the item mean. The last batch of an epoch is often shorter. The contract does not special-case that; item-axis combine already handles it.

### 6. Keys in, tensors only

`metric_function` receives a mapping of tensors, sliced to `required_keys`. It does not receive the raw dataloader batch, the model, or Python labels as lists.

The training loop (or the project) builds that mapping (`output` plus whatever targets the metric named). Metrics do not parse tuples.

### 7. Promote only what is battle-tested

This pattern is meant to transfer. A classification helper that already reduces the batch inside `metric_function` is not the contract. No backward compatibility unless the user asks for it.

## What this is for

Print three lines and understand a score: which metric name, which split and grain, which reducer. If epoch loss disagrees with a hand mean over the dataset, it failed because `metric_function` reduced too early — not because the accumulator averaged the wrong list of scalars.
