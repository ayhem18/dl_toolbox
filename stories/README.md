# Stories

A story records a method that already moved a result, so the next segmentation project can reuse the decision instead of regenerating it.

These are documents. They are not installed with `dl-toolbox`. Code that more than one project should call is promoted into `src/dl_toolbox`; the story keeps the context the function cannot carry.

## Template

```markdown
# <short name>

## Problem shape
What was similar across tasks (labels, class imbalance, resolution, metric).

## What was tried
The small set of variants that were actually compared.

## What held
The choice that survived, and the standard it was judged against.

## Reuse
What to copy next time, and which `dl_toolbox` module now owns the code, if any.
```
