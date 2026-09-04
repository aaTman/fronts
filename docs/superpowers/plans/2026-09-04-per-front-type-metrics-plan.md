# Per-Front-Type Losses and Metrics — Implementation Plan

## Context

Training currently logs exactly two scalars to W&B: `hss` and `hss_hard`, both collapsed
across all six classes. The per-class information already exists inside the metric
implementations — `heidke_skill_score`, `critical_success_index` and
`probability_of_detection` in `src/fronts/layers/metrics.py` all build per-class
TP/FP/FN/TN vectors and then `reduce_sum` them into a single scalar one line before
returning. The offline evaluation pipeline (`src/fronts/evaluate.py:158-191`) already
produces a full per-front-type breakdown (`pod_{ft}`, `csi_{ft}`, `hss_{ft}`, ...), but it
is a separate numpy/xarray path that never touches the training loop.

This plan surfaces per-front-type losses and metrics during training, logs them to W&B, and
persists them to a local CSV for later plotting.

## Global Constraints

These bind every task. A violation is a review defect.

- **Reporting only.** Gradients and the scalar quantity being optimized must be
  bit-for-bit unchanged. Nothing in this plan may alter training dynamics. The existing
  aggregate `hss` / `hss_hard` metrics keep their current values and behavior.
- **Language:** English only — code, comments, docstrings, commits, tests.
- **Style:** Google Python style guide. `ruff` for formatting, `pylint` clean. Max line
  length 120. Google-convention docstrings on every public function/class.
- **Self-documenting code over comments.** Never use comment blocks as section separators
  (no `# --- Section ---`).
- **Imports:** packages and modules only (except `typing`). Full pathname module imports.
  All imports at top of file.
- **No mutable global state.**
- **TDD, no mocks.** Write the failing test first, then the code. Use real tensors and
  real fixtures. `pytest`, tests under `tests/` mirroring `src/fronts/`. Guard
  TensorFlow-touching tests with `pytest.importorskip("tensorflow")`, matching the
  existing convention in `tests/layers/test_metrics.py`.
- **Dataclasses:** avoid default field values unless absolutely necessary. Where a default
  is unavoidable (to keep existing YAML configs parsing), add a comment stating why.
- **Environment:** run everything through pixi. Tests: `pixi run test`
  (which is `PYTHONPATH=src python -m pytest tests/ -v`). Never use conda, uv, or system
  Python. To run a subset: `pixi run -e default python -m pytest tests/layers/test_metrics.py -v`.
- **Commit** after each task with a descriptive message. End commit messages with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01S643f33viuW9XfyQYipCnD
  ```

## Domain Facts (verified, do not re-derive)

- Six classes, background at index 0: `{"CF": 1, "WF": 2, "SF": 3, "OF": 4, "DL": 5}`.
- The model (`UNet3Plus`) emits ~5 deep-supervision outputs. `model.outputs[0]` is the
  **finest-resolution head**, named `sup1_softmax` (`src/fronts/model.py:604-608` reverses
  the construction order; `src/fronts/callbacks.py` uses `pred[0]` for the same reason).
- Keras names per-output metric keys `{output_layer_name}_{metric_name}`, e.g.
  `sup1_softmax_hss`. Validation keys get a `val_` prefix on the whole key:
  `val_sup1_softmax_hss`.
- `MetricsConsolidationCallback` (`src/fronts/callbacks.py:101-148`) runs before
  `WandbMetricsLogger` and rewrites the shared Keras `logs` dict in place. Its regexes:
  - `_PER_OUTPUT_LOSS_RE = re.compile(r"^sup\d+_.+_loss$")` — anchored at the end, so a
    metric named `loss_CF` (key `sup1_softmax_loss_CF`) is **not** matched and survives.
  - `_PER_OUTPUT_METRIC_RE = re.compile(r"^sup\d+_[^_]+_(?P<metric>.+)$")` — captures
    everything after the activation as the metric name, so `hss_CF` is captured intact.
    Keys are grouped by captured metric name and averaged; a group with one member
    (which is the case for metrics attached only to `sup1`) averages to itself.
- `heidke_skill_score` uses `tf.math.divide_no_nan`, so a batch containing zero observed
  pixels of a rare class yields **0.0, not NaN**. This is why per-class metrics must
  accumulate counts across batches rather than average per-batch scalars — see Task 2.
- `y_true` and `y_pred` at the metric are float32, shape `(batch, lat, lon, class)` for
  2D models and `(batch, level, lat, lon, class)` for 3D, class axis last.
- Config parsing is `dacite.from_dict` with `check_types=False`, one dataclass per
  top-level YAML key (`src/fronts/utils.py:369-393`). Adding a **required** field to a
  config dataclass breaks all 17 existing YAML configs at parse time.

---

## Task 1: Consolidate shared constants into `src/fronts/constants.py`

**Goal:** create one dependency-light home for constants currently duplicated across
modules, so `src/fronts/layers/metrics.py` can reach the front-type mapping without
importing `fronts.callbacks` (which imports `wandb` at module scope) or
`fronts.plot.plot` (which imports matplotlib and cartopy at module scope).

`src/fronts/model_1702/run_eval.py:33-42` currently duplicates `OFFICE_REGIONS` verbatim
with a comment explaining it does so to avoid the `wandb` import. This task removes the
need for that workaround.

### Create `src/fronts/constants.py`

It must import **only** the standard library and `numpy`. It must not import anything from
`fronts`. Move these in, verbatim (same values, same ordering):

| Constant | Current home |
|---|---|
| `BoundingBox` (namedtuple) | `src/fronts/utils.py:24` |
| `FRONT_TYPE_CLASS_INDEX` | `src/fronts/callbacks.py:22` (also duplicated at `evaluate.py:46`, `plot/plot.py:64`) |
| `FRONT_NAMES` | `src/fronts/plot/plot.py:56-62` |
| `FRONT_COLORS` | `src/fronts/plot/plot.py:40-46` |
| `CONTOUR_CMAPS` | `src/fronts/plot/plot.py:48-54` |
| `FRONT_CLASS_MAP` | `src/fronts/data/targets.py:6` |
| `OFFICE_REGIONS` | `src/fronts/callbacks.py:28-34` |
| `LITE_THRESHOLDS` | `src/fronts/callbacks.py:36` |

Carry the existing explanatory comments across with the constants they document (the
Unified Surface Analysis / WPC manual note on `OFFICE_REGIONS`, and the
`evaluate.py:44-45` note clarifying that `FRONT_TYPE_CLASS_INDEX` holds one-hot indices,
not raw front data codes).

Add one new constant:

```python
BACKGROUND_CLASS_KEY = "none"
```

This is the token used to label the background (class 0) in per-front-type metric names.
It must not collide with any key in `FRONT_TYPE_CLASS_INDEX`.

### Update importers

- `src/fronts/utils.py` — delete the local `BoundingBox` definition, re-export it
  (`from fronts.constants import BoundingBox`) so every existing `utils.BoundingBox`
  reference keeps working. This is a deliberate compatibility re-export; note it as such.
- `src/fronts/callbacks.py` — delete the three moved constants, import from
  `fronts.constants`. Update all internal references (lines 194, 244, 349, 378, 385).
- `src/fronts/evaluate.py` — delete its `FRONT_TYPE_CLASS_INDEX` (line 46) and the
  comment above it, import from `fronts.constants`. Update line 314.
- `src/fronts/plot/plot.py` — delete the four moved constants, import from
  `fronts.constants`. Update lines 282, 321, 328, 362, 373, 410, 490, 581.
- `src/fronts/data/targets.py` — delete `FRONT_CLASS_MAP`, import from
  `fronts.constants`. Update lines 18 (docstring reference), 27, 48.
- `src/fronts/model_1702/run_eval.py` — delete the duplicated `OFFICE_REGIONS` and the
  comment at lines 33-35 explaining the duplication, import from `fronts.constants`.
  Keep `REGION_CHOICES` working (line 44) and lines 120-121.
- `src/fronts/model_1702/case_study.py` — update `plot.FRONT_TYPE_CLASS_INDEX`,
  `plot.CONTOUR_CMAPS`, `plot.FRONT_COLORS`, `plot.FRONT_NAMES` references (lines 141,
  167, 175, 190) to import from `fronts.constants` directly.
- `src/fronts/train.py:653` — `list(fronts_callbacks.FRONT_TYPE_CLASS_INDEX)` becomes a
  `fronts.constants` reference.
- `tests/test_train.py:9` — update the `FRONT_CLASS_MAP` import. **Also fix the stale
  comment on line 38**: it reads `# [1, 2, 3, 4, 15]` but the actual dryline code is `16`.
- `tests/model_1702/test_regions.py:44` — uses `run_eval.OFFICE_REGIONS`; confirm it still
  resolves after the import change.

Do **not** leave backwards-compatibility aliases in `callbacks.py`, `evaluate.py`,
`plot/plot.py`, or `targets.py` — update every call site instead. The `utils.BoundingBox`
re-export is the single exception, because it is referenced widely as a type annotation.

### Tests

Add `tests/test_constants.py`:
- `FRONT_TYPE_CLASS_INDEX` maps exactly `{"CF": 1, "WF": 2, "SF": 3, "OF": 4, "DL": 5}`.
- Every key in `FRONT_TYPE_CLASS_INDEX` has an entry in `FRONT_NAMES`, `FRONT_COLORS`,
  and `CONTOUR_CMAPS`.
- Class indices are unique, contiguous, and none is 0 (0 is reserved for background).
- `BACKGROUND_CLASS_KEY` is not a key in `FRONT_TYPE_CLASS_INDEX`.
- `FRONT_CLASS_MAP` values are exactly the `FRONT_TYPE_CLASS_INDEX` values.
- Importing `fronts.constants` does not import `wandb`, `matplotlib`, or `tensorflow`
  (assert those module names are absent from `sys.modules` after a subprocess import, or
  use `importlib` in a clean subprocess — this is the property the whole task exists for).

The full existing suite must still pass — this task is a pure move with no behavior change.

---

## Task 2: Stateful per-front-type contingency metrics

**Goal:** add per-front-type HSS (soft), HSS (hard), CSI and POD as stateful Keras metrics
in `src/fronts/layers/metrics.py`.

### Why stateful

The obvious implementation — wrapping the existing metric closures in
`MeanMetricWrapper` with a one-hot `class_weights` vector — is wrong here. Those closures
use `divide_no_nan`, so a batch with zero observed pixels of a rare class returns `0.0`.
Averaged over an epoch, that silently drags DL and OF (the rare types this feature exists
to expose) toward zero. Accumulating the contingency table across the epoch and computing
the ratio once at `result()` time avoids this, and makes the numbers directly comparable
to the `hss_{ft}` / `csi_{ft}` / `pod_{ft}` values `evaluate.py` already produces offline.

### Implementation

Add a `tf.keras.metrics.Metric` subclass. Suggested shape (adjust naming to fit the
module's conventions):

```python
class PerClassContingencyMetric(tf.keras.metrics.Metric):
    def __init__(self, class_index, statistic, threshold=None, name=None, dtype=None):
        ...
```

- `class_index: int` — index into the class axis.
- `statistic: str` — one of `"hss"`, `"csi"`, `"pod"`.
- `threshold: float | None` — when set, binarize `y_pred` at that threshold before
  accumulating (`tf.where(y_pred >= threshold, 1.0, 0.0)`), matching the existing
  `heidke_skill_score` behavior.
- State: four scalar weight variables (true positives, false positives, false negatives,
  true negatives), created with `add_weight(..., initializer="zeros")`.
- `update_state(y_true, y_pred, sample_weight=None)`: cast both to float32, slice the
  class axis to `class_index` (use `tf.gather(..., axis=-1)` or equivalent so it works for
  both 4D and 5D inputs), apply the threshold if set, then accumulate the four counts by
  summing over **all** remaining axes. Do not apply class weighting — this metric is
  already restricted to one class.
- `result()`: compute the statistic from the accumulated counts, using
  `tf.math.divide_no_nan` throughout.
  - HSS: `2 * divide_no_nan((a*d) - (b*c), ((a+c)*(c+d)) + ((a+b)*(b+d)))` where
    `a, b, c, d` are TP, FP, FN, TN — identical algebra to
    `heidke_skill_score` (`src/fronts/layers/metrics.py:222-227`).
  - CSI: `divide_no_nan(tp, tp + fp + fn)`.
  - POD: `divide_no_nan(tp, tp + fn)`.
- `reset_state()`: zero all four variables.
- `get_config()`: round-trip `class_index`, `statistic`, `threshold`, `name`, `dtype` so
  the metric survives model serialization.

Add a module-level factory that builds the full set for a run:

```python
def per_front_type_metrics(front_type_class_index, hard_threshold=0.5):
    """Build the per-front-type metric set attached to the finest model output."""
```

It returns a list containing, for each front type `ft` in `front_type_class_index`:

| Metric name | statistic | threshold |
|---|---|---|
| `hss_{ft}` | `hss` | `None` |
| `hss_hard_{ft}` | `hss` | `0.5` |
| `csi_{ft}` | `csi` | `0.5` |
| `pod_{ft}` | `pod` | `0.5` |

CSI and POD use the hard threshold because the probabilistic forms are hard to interpret
and the thresholded values are what the offline eval reports. Metric names must be unique
across the whole returned list.

### Tests

Extend `tests/layers/test_metrics.py`, reusing the existing fixture conventions
(`N_BATCH=2, N_H=8, N_W=8, N_CLASSES=6`, hand-built one-hot arrays):

- Per-class HSS/CSI/POD on a hand-constructed prediction match values computed by hand
  from the contingency table.
- A perfect prediction gives HSS 1.0, CSI 1.0, POD 1.0 for every front type.
- **The bias test:** two batches where the first contains zero pixels of class `CF` and
  the second contains `CF` pixels predicted perfectly. Accumulated across both batches the
  metric must report CSI/POD/HSS of 1.0 for `CF` — not 0.5, which is what a batch-mean
  implementation would give. This is the regression test for the whole design choice.
- Accumulation over two batches equals a single call on the concatenated batch.
- `reset_state()` returns the metric to its initial value.
- The metric works on a 5D `(batch, level, lat, lon, class)` input.
- `get_config()` round-trips through `from_config()`.
- `per_front_type_metrics` returns 4 metrics per front type with unique names matching
  the table above.

---

## Task 3: Per-front-type loss reporting and `_compile` wiring

**Goal:** report the configured training loss broken down per front type, and attach the
whole per-front-type metric set to the finest output only.

Depends on Tasks 1 and 2.

### Per-front-type loss metrics

In `src/fronts/train.py`, add a helper that builds one loss-valued metric per class by
calling the existing `_build_loss(...)` with a one-hot `loss_class_weights` vector, and
wrapping each in `tf.keras.metrics.MeanMetricWrapper` (the existing `hss_hard` at
`train.py:380-383` is the precedent for wrapping to get a stable `.name`).

Naming: `loss_{ft}` for each front type, plus `loss_{BACKGROUND_CLASS_KEY}` (i.e.
`loss_none`) for class 0.

**Include background** so the six values form a complete decomposition of the total loss.
Whether they sum exactly to the total depends on how each loss normalizes its class
weights — `fractions_skill_score` normalizes (`relative_cw = cw / sum(cw)`) while
`neighborhood_brier_score` does not. Determine the actual behavior by reading the code,
then make the reported per-class values sum to the total for the configured loss, and pin
that with a test. If exact additivity is not achievable for a given loss without changing
the loss itself, document why in the docstring and assert the weaker property that holds
(the Global Constraint forbids changing the loss).

### `TrainConfig` flag

Add to `TrainConfig` in `src/fronts/train.py`:

```python
# Defaulted (contrary to the usual no-defaults rule for dataclasses) so the 17 existing
# YAML configs keep parsing: dacite raises on a missing required field.
per_front_type_metrics: bool = True
```

Document it in the `TrainConfig` docstring's Attributes section, matching the existing
style.

### `_compile` changes

`src/fronts/train.py:392-396` currently reads:

```python
metrics=[[hss_fn, hss_hard_fn]] * n_out,
```

Change it so the per-front-type metrics attach to `model.outputs[0]` only:

```python
metrics=[[hss_fn, hss_hard_fn, *per_class]] + [[hss_fn, hss_hard_fn]] * (n_out - 1),
```

where `per_class` is empty when `train_cfg.per_front_type_metrics` is False. Handle
`n_out == 1` (deep supervision disabled) correctly.

The aggregate `hss_fn` / `hss_hard_fn` on every output must be untouched, as must the
`loss=loss_fn` argument.

### Tests

Extend `tests/test_train.py`, using the existing `_build_small_unet` helper:

- `_compile` with `per_front_type_metrics=True` attaches the per-class metrics to output 0
  and leaves every other output with exactly `[hss, hss_hard]`.
- `per_front_type_metrics=False` reproduces the current metric layout exactly.
- All compiled metric names are unique.
- `n_out == 1` works.
- The per-front-type loss values sum to the total loss (or the documented weaker property)
  for both `neighborhood_brier_score` and `fractions_skill_score`.
- The compiled loss and the optimized scalar are unchanged by the flag — assert that
  `model.compiled_loss` / the loss value on a fixed input is identical with the flag on
  and off. This is the Global Constraint's regression test.

---

## Task 4: W&B key naming in `MetricsConsolidationCallback`

**Goal:** rename the flat per-class metric keys into slash-delimited W&B keys so the run
page groups by front type.

Depends on Tasks 1-3 for the key names.

In `src/fronts/callbacks.py`, extend `MetricsConsolidationCallback` with a renaming step
that runs **after** the existing consolidation (so it sees `hss_CF`, not
`sup1_softmax_hss_CF`) and before the callback returns.

Rename rule: for a key whose name ends in `_{token}` where `token` is a key of
`FRONT_TYPE_CLASS_INDEX` or equals `BACKGROUND_CLASS_KEY`, rewrite it as
`front/{token}/{remainder}`, preserving any `val_` prefix on the remainder:

| Before | After |
|---|---|
| `hss_CF` | `front/CF/hss` |
| `hss_hard_CF` | `front/CF/hss_hard` |
| `csi_DL` | `front/DL/csi` |
| `pod_OF` | `front/OF/pod` |
| `loss_CF` | `front/CF/loss` |
| `loss_none` | `front/none/loss` |
| `val_hss_CF` | `front/CF/val_hss` |
| `val_loss_none` | `front/none/val_loss` |

Aggregate keys (`hss`, `hss_hard`, `loss`, `val_loss`, ...) must be left alone — none of
them ends in a front-type token.

`WandbMetricsLogger` prefixes epoch metrics with `epoch/`, so the final W&B keys become
`epoch/front/CF/hss` and W&B groups them into one collapsible section per front type.

### Tests

Extend `tests/test_callbacks.py`, following the existing
`TestMetricsConsolidationCallback` patterns (which construct `logs` dicts directly):

- Each row of the rename table above.
- Aggregate keys pass through untouched.
- A metric name that merely *contains* a front-type token but does not end with one is not
  renamed.
- The existing `sup{N}` averaging behavior is unchanged (the existing tests must still
  pass).
- Per-class keys present on only `sup1` survive consolidation with their value intact
  (a one-member group averages to itself).

---

## Task 5: Persist per-epoch metrics to CSV

**Goal:** write every epoch's metrics to a local CSV so they can be plotted later
independently of W&B.

Depends on Task 4 for ordering.

### `CallbacksConfig` field

Add to `CallbacksConfig` in `src/fronts/callbacks.py`:

```python
# Defaulted (contrary to the usual no-defaults rule for dataclasses) so the 17 existing
# YAML configs keep parsing: dacite raises on a missing required field.
metrics_csv_path: str | None = None
```

When `None`, derive `metrics_epoch.csv` in the same directory as
`model_checkpoint_path`. When `model_checkpoint_path` is also `None`, skip the CSV logger
entirely rather than guessing a location. Document both behaviors in the docstring's
Attributes section.

### Callback wiring

In `_build_run_callbacks` (`src/fronts/train.py:476-547`), add
`tf.keras.callbacks.CSVLogger(path, append=True)` positioned **after**
`MetricsConsolidationCallback` so it records consolidated, renamed keys, and before
`WandbMetricsLogger`. `append=True` so a resumed run extends rather than truncates.

Create the parent directory if it does not exist.

Extend the ordering docstring/comment already present at `train.py:531-533` to cover the
new constraint, in the same style.

### Tests

Extend `tests/test_train.py`'s `TestBuildRunCallbacks`:

- The CSV logger is present and positioned after `MetricsConsolidationCallback`.
- An explicit `metrics_csv_path` is honored.
- `metrics_csv_path=None` with a checkpoint path derives `metrics_epoch.csv` beside it.
- Both `None` omits the CSV logger entirely.
- `append=True` is set.
- Use `tmp_path` for filesystem assertions.

---

## Definition of Done

- `pixi run test` passes in full.
- `ruff` formatting clean, `pylint` clean on changed files.
- Training a model produces W&B keys `epoch/front/{CF,WF,SF,OF,DL}/{hss,hss_hard,csi,pod,loss}`
  and `epoch/front/none/loss`, alongside the unchanged aggregate `epoch/hss`,
  `epoch/hss_hard`, `epoch/loss`.
- The same values land in `metrics_epoch.csv` next to the checkpoint.
- The optimized loss is provably unchanged.
