# Implementation Summary: Longitude-Patch Training with Context Buffer and Flip Augmentation

---
**Date:** 2026-08-31
**Author:** AI Assistant
**Status:** Complete
**Plan Reference:** [plan-patch-buffer-training.md](plan-patch-buffer-training.md)

---

## Overview

Implemented all 6 phases of the plan: patch-based sliding-window longitude training
(reproducing Justin et al. 2025's original 9-patches-per-timestep regime), an optional
input-only context buffer around each patch with crop-after-pooling in the loss
(Ronneberger et al. 2015's overlap-tile strategy), per-axis flip augmentation on train
patches, and a new ablation config wiring it all together on the existing
CONUS/3D/`neighborhood_brier_score` pipeline.

**Implementation Duration:** Single session, 2026-08-31.

**Final Status:** ✅ Complete

## Plan Adherence

**Plan Followed:** [plan-patch-buffer-training.md](plan-patch-buffer-training.md)

**Deviations from Plan:**

- **Deviation 1:** All 6 phases were implemented together before running the full test
  suite, rather than committing after each individual phase as the plan's per-phase task
  lists show.
  - **Reason:** The phases are tightly sequential (Phase 2 depends on Phase 1's
    `PatchConfig`/`compute_patch_lon_starts`; Phase 5 depends on Phases 2-4's function
    signatures) and every line of code was already fully specified in the plan with no
    open design decisions left to resolve mid-implementation, so there was no benefit to
    stopping for review between phases. Each phase's tests were still verified
    individually against the running implementation (see Verification Results) before
    moving to full-suite verification.
  - **Impact:** No commits have been made yet (see Next Steps) — all work is currently
    uncommitted on `experiment/patch-buffer-ablation`. No functional impact.

- **Deviation 2:** Phase 4's `test_pred_buffer_px_pooling_reads_real_buffer_values_not_zero_padding`
  test was redesigned during implementation. The plan's original version relied on an
  incorrect assumption that Keras's `AveragePooling2D(padding="same")` zero-pads and
  divides by the full kernel size at domain edges; empirically it excludes padding from
  the divisor (a uniform field pools to itself even at the edge), so the original
  "uniform background buffer ring" test was degenerate (scored exactly 0 either way).
  - **Reason:** Discovered while running the test for the first time — see Issues
    Encountered.
  - **Impact:** The test now uses a front pixel at the very edge instead, proving the
    buffered pooling window genuinely averages over 9 real buffer-region cells versus the
    unbuffered window's 6 in-domain-only cells. The underlying implementation
    (`_crop_pred_buffer`, crop-after-pooling in `nbs_loss`) is unchanged from the plan —
    only the test's construction changed. Both `docs/rse/specs/plan-patch-buffer-training.md`
    and `tests/layers/test_losses.py` were updated to match.

- **Deviation 3:** Plan checkboxes were bulk-updated via a script rather than one Edit
  call per checkbox, given the volume (56 items). Manual Verification and "What We're NOT
  Doing" checkboxes were deliberately left unchecked.
  - **Reason:** Efficiency; no semantic difference from checking them individually.
  - **Impact:** None.

## Phases Completed

### Phase 1: Patch geometry config and pure helpers
- ✅ **Status:** Complete
- **Completion Date:** 2026-08-31
- **Summary:** Added `PatchConfig` and `compute_patch_lon_starts` to
  `src/fronts/data/datasets.py`, `DatasetConfig.patch_config` field, and
  `utils.expand_bounding_box` to `src/fronts/utils.py`. 11 new tests, all passing.

### Phase 2: `FrontsPyDataset` patch sampling and flip augmentation
- ✅ **Status:** Complete
- **Completion Date:** 2026-08-31
- **Summary:** `FrontsPyDataset` gained `augment`, `patch_sample_index`,
  `_get_patches_at_indices`, and `_apply_flip_augmentation`; `_total`/`get_at_indices`
  dispatch into patch mode when `data_config.patch_config` is set. 6 new tests, all
  passing.

### Phase 3: Buffered domain loading in `load_data_into_dataloader`
- ✅ **Status:** Complete
- **Completion Date:** 2026-08-31
- **Summary:** `load_data_into_dataloader` gained an `augment` parameter and a
  buffered-vs-core spatial-domain branch, with a `ValueError` if the buffer runs past the
  store's coverage or if `patch_config` is set without `coordinates`. 5 new tests, all
  passing.

### Phase 4: Crop-after-pooling in `neighborhood_brier_score`
- ✅ **Status:** Complete
- **Completion Date:** 2026-08-31
- **Summary:** Added `pred_buffer_px` and `_crop_pred_buffer` to
  `src/fronts/layers/losses.py`; the neighborhood pool runs on the full buffered
  prediction, then the pooled result (and, for `include_pixel`, the raw prediction) is
  cropped to match the target before scoring. 4 new tests, all passing (see Deviation 2
  above for the one test redesign).

### Phase 5: `train.py` wiring
- ✅ **Status:** Complete
- **Completion Date:** 2026-08-31
- **Summary:** Added `_pred_buffer_px_from_data_config`, `_target_latitudes`, and
  `_should_build_test_visualization` helpers; threaded `pred_buffer_px` through
  `_build_loss`/`_compile`; fixed `train()`'s `latitudes=` argument to read from
  `target_da` instead of the (now potentially buffered) `input_ds`; gated the periodic
  visualization callback off in patch mode; passed `augment=True` only for the train
  split. 9 new tests, all passing.

### Phase 6: New ablation config and end-to-end smoke test
- ✅ **Status:** Complete
- **Completion Date:** 2026-08-31
- **Summary:** Added `configs/patch_buffer_ablation.yaml` and
  `TestPatchBufferEndToEnd`/`test_patch_buffer_ablation_config_parses`, proving a real
  (small) `UNet3Plus` composes with a buffered-patch input and the buffer-aware loss
  end to end. 2 new tests, all passing.

## Files Modified

**Created:**
- `configs/patch_buffer_ablation.yaml` — new ablation config reproducing the paper's
  patch/flip/buffer training regime on the CONUS 3D pipeline
- `docs/rse/specs/plan-patch-buffer-training.md` — implementation plan
- `docs/rse/specs/implement-patch-buffer-training.md` — this document

**Modified:**
- `src/fronts/data/datasets.py` — `PatchConfig`, `compute_patch_lon_starts`,
  `DatasetConfig.patch_config`, `FrontsPyDataset` patch-mode sampling and flip
  augmentation
- `src/fronts/utils.py` — `expand_bounding_box`
- `src/fronts/layers/losses.py` — `neighborhood_brier_score(pred_buffer_px=...)`,
  `_crop_pred_buffer`
- `src/fronts/train.py` — `load_data_into_dataloader(augment=...)`, buffered-domain
  loading, `_pred_buffer_px_from_data_config`, `_target_latitudes`,
  `_should_build_test_visualization`, `_build_loss`/`_compile` threading, `train()`
  wiring
- `tests/test_train.py` — new test classes for all of the above (see Testing Summary)
- `tests/test_utils.py` — `TestExpandBoundingBox`
- `tests/layers/test_losses.py` — `TestNeighborhoodBrierScorePredBuffer`

**Deleted:** No files deleted.

## Key Changes Summary

1. **Patch sampling with an optional context buffer**
   - Each timestep expands into `n_patches` samples via
     `divmod(global_idx, n_patches)`; the buffered input and unbuffered target are sliced
     independently per patch.
   - Files: `src/fronts/data/datasets.py:16-83,157-186,231-350`

2. **Buffered spatial-domain loading**
   - `inputs_ds` loads `coordinates` widened by `buffer_px` grid cells; `targets_da`
     always loads the unbuffered core; a shape mismatch against the store's actual
     coverage raises immediately rather than silently training on a smaller buffer.
   - Files: `src/fronts/train.py:117-199`

3. **Crop-after-pooling loss (the literature-grounded design decision)**
   - `neighborhood_brier_score`'s neighborhood pool runs on the full buffered
     prediction; only the pooled result is cropped to the target's shape before scoring
     — so boundary cells use real buffer-region context instead of falling back to
     zero-padding.
   - Files: `src/fronts/layers/losses.py:296-425`

4. **A real latent bug fix, surfaced by this work**
   - `train()` was passing `train_dataset.input_ds["latitude"].values` to the loss
     builder; that's silently wrong once `input_ds` can be spatially wider than the
     target it's scored against (patch mode's buffered input). Fixed to read from
     `target_da` via the new `_target_latitudes` helper — a no-op change outside patch
     mode (both grids share the same latitudes there), but load-bearing inside it.
   - Files: `src/fronts/train.py:299-306,752`

## Verification Results

### Automated Verification

- ✅ `pytest tests/test_train.py tests/test_utils.py tests/layers/test_losses.py -q` —
  all new tests pass; only 2 pre-existing, unrelated failures remain
  (`test_3d_config_parses`, `test_nbs_include_pixel_true_changes_loss`)
- ✅ `pytest tests/ -q` — 525 passed, 4 skipped, 7 failed (all 7 confirmed pre-existing
  via `git stash` on the unmodified branch — see Issues Encountered)
- ✅ `ruff check` on every changed file (pinned to the project's actual
  `.pre-commit-config.yaml` version, v0.9.4) — 0 new violations; 5 remaining violations
  are all in pre-existing, untouched code
- ✅ `ruff format --diff` on every changed file — 0 new reformatting needed; the 2 flagged
  diffs are both in pre-existing, untouched code
- ✅ `configs/patch_buffer_ablation.yaml` parses via
  `utils.parse_config_section(..., DatasetConfig, "data_config", type_hooks=utils.YAML_TYPE_HOOKS)`
  with `patch_config == PatchConfig(n_patches=9, patch_lon_width_px=128, buffer_px=16, flip_probability=0.25)`

**Command Output:**
```
tests/ -q: 7 failed, 525 passed, 4 skipped, 71 warnings in 36.35s
(the 7 failures are identical on the unmodified branch — confirmed via git stash)
```

### Manual Verification

Not yet performed — requires a real (or realistically small local) icechunk store per
the plan's Manual Verification section. Left unchecked in the plan for the user to run:

- [ ] Launch a short real training run with `configs/patch_buffer_ablation.yaml` and
      confirm 9x sample count, correct buffered input shape in `unet.summary()`, no
      shape errors
- [ ] Confirm the "Skipping periodic test-set visualization" info log fires at startup
- [ ] Spot-check flipped training patches visually for geometric correctness

## Issues Encountered

### Issue 1: Local test environment (`.pixi/envs/mac`) had two missing declared
dependencies (`dask-jobqueue`, `arraylake`), and fixing the second pulled in a newer
`icechunk`/`zarr`/`numpy` combination that was ABI-incompatible with the installed
`tensorflow==2.16.2` wheel (a genuine C-extension crash, not just a pip resolver
warning).
- **Impact:** Several rounds of environment repair were needed before any test could
  actually run instead of silently skipping via `_TF_AVAILABLE=False`. This burned
  significant time and briefly left the shared local environment in a broken state.
- **Resolution:** Paused and asked the user how to proceed rather than continuing to
  guess. Per their guidance (don't worry about icechunk/arraylake specifically — verify
  the core logic), restored `numpy<2.0`/`icechunk<2` — the combination that actually
  keeps `tensorflow` importable in this wheel — and confirmed `arraylake`/`icechunk`
  import fine at runtime despite pip's advisory version-conflict warnings (those
  warnings are non-blocking). This restored the full `_TF_AVAILABLE=True` path, taking
  the test suite from 18 real tests running (rest silently skipped) to 525.
- **Files Affected:** No source files — only `.pixi/envs/mac`'s installed packages
  (local, untracked, not part of the repo).

### Issue 2: Ruff version drift.
- **Impact:** A generically `pip install ruff`'d version (0.16.5) flagged ~35 violations
  in pre-existing, untouched code (e.g. `PLR0917` on `_run`'s 19 pre-existing parameters)
  that the project's actual pinned pre-commit version (`v0.9.4`, from
  `.pre-commit-config.yaml`) does not enforce — that rule didn't exist yet in 0.9.4.
  Would have produced a misleading "40 lint errors" report.
- **Resolution:** Reinstalled the exact pinned version (`ruff==0.9.4`) before treating
  lint output as signal, then fixed only the genuinely-new issues in code I added
  (docstring formatting in `datasets.py`/`losses.py`/`train.py`/`test_losses.py`, one
  `isinstance` style fix in `test_train.py`).
- **Files Affected:** None beyond the fixes already listed in Key Changes Summary.

### Issue 3: The plan's original crop-after-pooling test was based on an incorrect
assumption about `AveragePooling2D`'s edge-padding behavior (see Deviation 2).
- **Impact:** The first test run showed `0.0` loss for both the buffered and unbuffered
  case, which would have looked like a passing-but-meaningless test if not caught.
- **Resolution:** Verified the actual pooling behavior empirically, redesigned the test
  around a front pixel at the domain edge instead of a uniform field, and updated the
  plan document to match. The underlying `_crop_pred_buffer`/`nbs_loss` implementation
  itself needed no changes — only the test.
- **Files Affected:** `tests/layers/test_losses.py`,
  `docs/rse/specs/plan-patch-buffer-training.md`.

## Testing Summary

**Tests Added:**
- `tests/test_train.py:TestPatchConfigValidation` — `PatchConfig` field validation
- `tests/test_train.py:TestComputePatchLonStarts` — patch-window geometry
- `tests/test_train.py:TestFrontsPyDatasetPatchMode` — patch/buffer slicing, flip
  augmentation
- `tests/test_train.py:TestLoadDataIntoDataloaderPatchBuffer` — buffered-domain loading,
  store-edge overrun, missing-`coordinates` error, `augment` threading
- `tests/test_train.py:TestPredBufferPxFromDataConfig` — `pred_buffer_px` config lookup
- `tests/test_train.py:TestTargetLatitudes` — the `input_ds`-vs-`target_da` latitude bug
  fix
- `tests/test_train.py:TestShouldBuildTestVisualization` — viz-callback patch-mode guard
- `tests/test_train.py:TestTrainConfigLossClassWeights::test_patch_buffer_ablation_config_parses`
  — new YAML config parsing
- `tests/test_train.py:TestPatchBufferEndToEnd` — real model + buffer-aware loss
  integration
- `tests/test_utils.py:TestExpandBoundingBox` — bounding-box widening
- `tests/layers/test_losses.py:TestNeighborhoodBrierScorePredBuffer` — crop-after-pooling
  shape handling and the buffer-vs-zero-padding proof

**Test Coverage:**
- Unit tests: 36 new tests across geometry helpers, dataset sampling, config loading,
  loss cropping, and train.py glue
- Integration tests: 1 (`TestPatchBufferEndToEnd`, a real `UNet3Plus` + real
  `neighborhood_brier_score` composing a buffered patch against an unbuffered target)
- Edge cases tested: buffer running past store coverage, missing `coordinates` with
  `patch_config` set, `flip_probability` at 0.0/1.0 boundaries, `pred_buffer_px=0`
  matching prior behavior exactly, `include_pixel` + buffer combined

**All Tests Passing:** ✅ Yes (all 37 new tests pass; 7 pre-existing, unrelated failures
remain unchanged from before this work — see Issues Encountered)

## Performance Observations

Performance was not a primary concern for this implementation. One known, deliberately
accepted cost is noted in the plan's "What We're NOT Doing": patch mode can read the same
timestep from the icechunk store twice within a batch when two of its patches land
together after shuffling — a minor I/O inefficiency, not a correctness issue.

## Documentation Updated

- ✅ `docs/rse/specs/plan-patch-buffer-training.md` — implementation plan (all phase
  checkboxes marked complete)
- ✅ Docstrings — `PatchConfig`, `compute_patch_lon_starts`, `expand_bounding_box`,
  `FrontsPyDataset._get_patches_at_indices`/`_apply_flip_augmentation`,
  `neighborhood_brier_score`'s `pred_buffer_px`, `load_data_into_dataloader`'s `augment`,
  and the three new `train.py` helper functions all documented per the project's Google
  docstring convention
- ✅ `configs/patch_buffer_ablation.yaml` — heavily commented, citing the paper (Justin
  et al. 2025) and the overlap-tile strategy (Ronneberger et al. 2015) inline

## Remaining Work

All planned work has been completed. Remaining items are the plan's explicit Manual
Verification steps (require a real icechunk store and are the user's to run) and the
deliberate scope exclusions listed in the plan's "What We're NOT Doing" section
(`fractions_skill_score` buffer support, `TestVisualizationCallback` patch-mode support,
buffer/stride runtime validation, general non-patch flip augmentation, batch-level
timestep-read deduplication).

## Next Steps

1. Review the diff (`git diff`, currently uncommitted on `experiment/patch-buffer-ablation`).
2. If it looks good, commit — I have not committed anything yet per the git safety
   protocol (commits require explicit user request).
3. Run the plan's Manual Verification steps against a real icechunk store when
   convenient.
4. Route to `ai-research-workflows:validating-implementations` if you'd like an
   independent check against the plan before merging/running the ablation for real.

**Recommended Actions:**
- Review the diff and let me know if you'd like it committed
- Kick off a real (small-scale) training run against `configs/patch_buffer_ablation.yaml`
  to exercise the Manual Verification steps

## Lessons Learned

**What Went Well:**
- Every phase's code matched the plan almost exactly, since the research and design
  decisions (crop-after-pooling, buffer threading, patch geometry) were fully resolved
  before implementation began — implementation itself surfaced only one genuine
  correctness fix (`_target_latitudes`) and one test-construction error (Deviation 2),
  no design rework.
- The end-to-end integration test (Phase 6) was worth the extra weight: it's the one
  test that would have caught a shape mismatch between the model's dynamic
  `(None, None, C)` input handling and the loss's buffer-cropping assumption, which no
  narrower unit test could see on its own.

**What Could Be Improved:**
- Should have checked `.pre-commit-config.yaml`'s pinned tool versions before installing
  any verification tooling generically, rather than after being misled by version-drift
  noise.

**Technical Insights:**
- Keras's `AveragePooling2D(padding="same")` excludes padding from its divisor (valid-
  cell normalization), not the naive "zero-pad then divide by kernel size" behavior —
  worth remembering for any future edge-handling work on pooled losses in this codebase.
- `pip`'s dependency-conflict warnings are advisory, not enforced at import time; two
  packages pip calls "incompatible" can still both import and run correctly (as
  `arraylake`+`icechunk<2` did here) — useful to know before treating a pip warning as a
  hard blocker, though real C-extension ABI mismatches (numpy 2.x vs this tensorflow
  build) are a different, actually-blocking category.

## References

**Plan Document:**
- [Plan: Longitude-Patch Training with Context Buffer and Flip Augmentation](plan-patch-buffer-training.md)

**Commits:** None yet — all changes are currently uncommitted, pending your review.

---

**Implementation completed by AI Assistant on 2026-08-31**
