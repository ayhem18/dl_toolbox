# Multi-threading standard

## Goal

Use threads only where they unlock **real parallelism** — typically when work is dominated by native code (OpenCV, NumPy ufuncs on large arrays) that releases the CPython GIL. Threading Python logic in a loop does not make the pipeline faster and often makes it harder to reason about.

Reference implementation: `packages/geomap_ocr/src/geomap_ocr/image_processing/batch_filters.py` (`segment_multi_threaded`).

---

## When to use threads

| Use threads | Do not use threads |
|-------------|-------------------|
| OpenCV per-image work (`cv2.*`, `fillPoly`, `dilate`, `threshold`, `warpAffine`, …) | Pure Python loops over regions, dict building, JSON I/O |
| Large NumPy ops that stay in native code | GPU inference (use batching on the GPU instead) |
| Independent images / tiles with no shared mutable state | Tasks that must run in strict order on one mutable buffer |

**Rule of thumb:** if the task body is mostly Python object manipulation, threading will not help. Refactor so each task spends its time in OpenCV or NumPy native calls.

---

## Required pattern

### 1. One native task per function

Each task callable should do one coherent unit of native work and return a result. Do not embed orchestration inside tasks.

```python
def _preprocess_task(image_bgr: np.ndarray, params: ...) -> SegmentationPreprocessResult:
    return preprocess_segmentation_gray(image_bgr, params=params)


def _otsu_task(blurred_u8: np.ndarray) -> tuple[float, np.ndarray]:
    return segment_otsu_inv(blurred_u8)
```

`segment_multi_threaded` splits segmentation into:

1. **Phase 1** — CLAHE + blur per image (`_preprocess_task`)
2. **Phase 2** — Otsu and adaptive threshold per image (`_otsu_task`, `_adaptive_task`), `2 × n` tasks interleaved
3. **Phase 3** — mask combine per image (`_combine_task`)

Phases are sequential; work **within** a phase is parallel across images (and across Otsu/adaptive for the same image in phase 2).

### 2. Shared executor passed through orchestration

The top-level function either:

- accepts an optional `executor: ThreadPoolExecutor | None`, or
- creates a local pool for the duration of the call.

Inner helpers take the executor explicitly — they do **not** create their own pools.

```python
def segment_multi_threaded(
    images: Sequence[np.ndarray],
    *,
    executor: ThreadPoolExecutor | None = None,
    max_workers: int | None = None,
) -> list[CombinedSegmentationFrames]:
    ...
    if executor is not None:
        _run_segment_phases_on_executor(executor, n, images, params, results)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as local_executor:
            _run_segment_phases_on_executor(local_executor, n, images, params, results)
```

Callers that run several threaded stages in one pipeline (e.g. layout mask build + paint + another OpenCV pass) should create **one** executor and pass it into each stage to avoid pool churn.

### 3. Preallocate outputs; submit then join per phase

When `n` is known, preallocate result slots (`[None] * n`), submit all futures for the phase, then `.result()` in order.

```python
preprocess_results: list[SegmentationPreprocessResult | None] = [None] * n
preprocess_futures = [None] * n
for i in range(n):
    preprocess_futures[i] = exec_pool.submit(_preprocess_task, images[i], params)
for i in range(n):
    preprocess_results[i] = preprocess_futures[i].result()
```

Do not use `pool.map(lambda i: ..., range(n))` over a generic wrapper that hides phase boundaries. Phases often depend on prior phase outputs; keep that structure visible.

### 4. Do not “thread the for loop”

**Anti-pattern** — wrapping an existing sequential loop without restructuring:

```python
# BAD: still mostly Python; false sense of parallelism
def build_masks(regions_per_page):
    with ThreadPoolExecutor() as pool:
        return list(pool.map(lambda args: build_dilated_layout_mask(*args), ...))
```

`build_dilated_layout_mask` does call OpenCV, so a threaded version **can** work — but only if:

- each task is a dedicated `_mask_task(page_bgr, regions, dilate_px)` that runs OpenCV end-to-end;
- phases are explicit (footprint fill → dilate → paint as separate task types if needed);
- a single executor is passed from the 1.4 orchestrator.

Prefer refactoring into OpenCV-heavy task functions over `pool.map` on a high-level helper.

---

## Orchestrator responsibilities

Higher-level APIs (e.g. step 1.4 layout fallback) should:

1. Validate inputs and preallocate per-page output lists.
2. Optionally create one `ThreadPoolExecutor` for all CPU-native substeps.
3. Call threaded helpers with `executor=...`.
4. Call **batched GPU** text detection once masked pages are ready (threading does not replace GPU batching).

```text
1.4 orchestrator
  ├─ Phase A (CPU, threaded): mask build per page   ─┐
  ├─ Phase B (CPU, threaded): mask paint per page    ├─ same executor
  └─ Phase C (GPU, batched):  Nemotron detect_text   ─┘ separate; batch pages
  └─ Phase D (CPU, sequential): orphan filter + grouping per page (fast Python)
```

Phase D stays sequential unless profiling shows it matters; it is graph/grouping logic, not OpenCV-bound.

---

## `max_workers`

- Default: `min(n_pages, os.cpu_count() or 4)` for page-parallel OpenCV.
- Cap workers when images are huge (memory): prefer fewer workers over OOM.
- `max_workers=1` must remain correct (sequential fallback).

---

## Testing

- **Correctness:** threaded path must match sequential path (`max_workers=1` vs `max_workers>1`) on the same inputs.
- **Performance:** measure with realistic page sizes; document speedup only when verified.
- Do not assert wall-time in unit tests; compare outputs only.

---

## Checklist before adding threads

- [ ] Task body is mostly OpenCV / native NumPy, not Python loops
- [ ] Task function is standalone (`_foo_task`) with clear inputs/outputs
- [ ] Orchestrator accepts optional `executor` and does not nest pools
- [ ] Phases are explicit; dependencies between phases are obvious
- [ ] Outputs preallocated (`[None] * n`); no `append` in hot paths (see `performance_standards.md`)
- [ ] Sequential fallback (`max_workers=1` or `executor is None` local pool) tested

---

## Related

- `performance_standards.md` — preallocated lists, batched I/O
- `packages/geomap_ocr/src/geomap_ocr/image_processing/batch_filters.py` — canonical threaded batch implementation
