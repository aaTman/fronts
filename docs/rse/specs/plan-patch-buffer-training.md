# Implementation Plan: Longitude-Patch Training with Context Buffer and Flip Augmentation

---
**Date:** 2026-08-31
**Author:** AI Assistant
**Status:** Draft
**Related Documents:** none (scoped directly from a research conversation; see References)

---

## Overview

Justin et al. (2025)'s original FrontFinder training regime extracts nine 128×128×5×10
(longitude × latitude × level × variable) patches per timestep, evenly spaced along
longitude, with each patch independently having a 25% chance of being flipped along each
horizontal dimension (43.75% of patches get at least one flip). This codebase currently
trains on one full-domain grid per timestep instead — `fronts/model_1702/adapter.py:9-12`
documents that departure as deliberate (a fully-convolutional net benefits from more
context than 128×128 tiles give it). This plan reproduces the paper's patch/flip regime as
an opt-in ablation path, alongside a context buffer around each patch: extra input pixels
on every side that give the model real spatial context near patch edges without being
scored by the loss.

**Goal:** A new `configs/patch_buffer_ablation.yaml` that trains the existing
`UNet3Plus`/`neighborhood_brier_score` pipeline on nine evenly-spaced, buffered,
flip-augmented longitude patches per CONUS timestep, with the loss's neighborhood pooling
run on the full buffered prediction and only *then* cropped to match the unbuffered core
target — the overlap-tile strategy from Ronneberger et al. (2015)'s U-Net paper, adapted to
this repo's pool-then-score loss instead of valid convolutions.

**Motivation:** Directly ablate the "full-domain single pass beats patch tiling" design
choice already recorded in `adapter.py`, using the exact patch/augmentation parameters the
original paper reports, plus the buffer this project wants to add on top.

## Current State Analysis

**Existing Implementation:**
- `fronts/data/datasets.py:76-186` (`FrontsPyDataset`) — one training sample per timestep;
  `get_at_indices` does a single `.isel(time=idxs)` against the full (already spatially
  cropped) input/target arrays. No per-sample spatial windowing or augmentation exists.
- `fronts/data/datasets.py:16-73` (`DatasetConfig`) — `coordinates: BoundingBox | None`
  applies one static spatial crop to the whole dataset before any batching.
- `fronts/train.py:117-227` (`load_data_into_dataloader`) — opens the icechunk stores,
  applies `coordinates` once via `utils.select_spatial_domain`, intersects/filters
  timesteps, splits by year, and constructs one `FrontsPyDataset`.
- `fronts/train.py:718` — passes `train_dataset.input_ds["latitude"].values` as the
  `latitudes` argument to the loss builder; today `input_ds` and `target_da` share the same
  spatial grid so this is currently correct, but it silently breaks once `input_ds` can be
  spatially wider than `target_da` (patch mode's buffered input).
- `configs/schooner_train_conus_3d.yaml` — already crops to
  `coordinates: [25.0, 56.75, 228.0, 299.75]`, which at the store's native 0.25° resolution
  is exactly 128 latitude × 288 longitude points — the paper's legacy 288×128 training
  image size (confirmed by `adapter.py:9-12`'s own comment).
- `fronts/layers/losses.py:296-392` (`neighborhood_brier_score`) — pools `y_true` and
  `y_pred` with matching shapes (`O_n = isotropic_pool(y_true)`, `M_n = isotropic_pool(y_pred)`,
  `_brier(O_n, M_n)`); assumes `y_pred` and `y_true` are the same spatial size.
- `fronts/utils.py:154-176` (`select_spatial_domain`) — already handles wrap-crossing
  longitude via `PeriodicBoundaryIndex` (`utils.py:56-136`), so widening a longitude crop
  past 360° "just works" with no new wraparound logic needed.
- `fronts/model.py` (`UNet3Plus.build`) — `Input(shape=self.input_shape)` where
  `train.py:679` derives `input_shape = (None, None, *train_inputs_da.shape[3:])` — spatial
  dims are already fully dynamic (`None, None`), so the model itself needs **no** change to
  accept a spatially larger (buffered) input than its target.
- `fronts/layers/modules.py:664-696` (`deep_supervision_side_output`) — for
  `output_level == 1` (the finest decoder node, level 1), no additional upsampling is
  applied (`upsample_size_1 = None`, `upsample_size_2 = None`), so every supervision head's
  output resolution exactly matches the model's input resolution (`"same"` padding
  throughout preserves spatial size at every stage).
- `fronts/train.py:526-570` (`_build_test_visualization_callback`) and
  `fronts/callbacks.py:286-396` (`TestVisualizationCallback`) — build one active-day
  prediction map and per-office-region performance diagrams from whole-timestep
  input/target pairs and whole-domain `lats`/`lons`; not adaptable to per-patch tiling
  without materially more work than this ablation needs.

**Current Behavior:** One full CONUS (or full-domain) grid per timestep is fed to the
model per training sample; no spatial data augmentation exists.

**Current Limitations:**
- No mechanism to draw multiple spatial sub-samples per timestep.
- No mechanism to give the model extra input-only spatial context beyond what gets scored.
- No training-time flip augmentation.

## Desired End State

**New Behavior:** With `data_config.patch_config` set, `load_data_into_dataloader` loads
inputs over a *buffered* spatial domain (core `coordinates` widened by `buffer_px` grid
cells on every side) and targets over the *unbuffered* core `coordinates` box only. Each
training sample is one of nine evenly-spaced 128-px-wide longitude windows (full 128-px
core latitude height, no latitude tiling) drawn from a given timestep; train-split patches
are independently flipped along latitude and along longitude with probability
`flip_probability` each. The model's raw output stays at the buffered spatial size;
`neighborhood_brier_score(pred_buffer_px=...)` pools that full buffered prediction, then
crops the *pooled* result down to the core size before scoring against the (never
buffered) target — so the loss's own neighborhood averaging benefits from genuine
buffer-region context instead of the zero-padding it would otherwise fall back to at
domain edges.

**Success Looks Like:**
- `configs/patch_buffer_ablation.yaml` parses cleanly and its `data_config.patch_config`
  produces 9 samples per timestep.
- A tiny end-to-end run (real `UNet3Plus` + real `neighborhood_brier_score`) accepts a
  buffered patch and an unbuffered core target with no shape errors and produces a finite
  loss.
- `pytest tests/ tests/layers/test_losses.py` passes, including new patch/buffer/flip tests.

## What We're NOT Doing

- [ ] **`fractions_skill_score` buffer support.** No config in this repo uses it for
  anything relevant here (both `sooner_ablations.yaml` and `schooner_train_conus_3d.yaml`
  use `neighborhood_brier_score`); adding generic N-D buffer cropping to its
  variable-rank `AveragePooling{1,2,3}D` pooling is real extra complexity with no current
  caller.
- [ ] **Latitude tiling.** The paper only tiles along longitude; CONUS's `coordinates` box
  is already exactly one 128-px latitude window, so there is nothing to tile.
- [ ] **Periodic-longitude buffer wraparound edge cases beyond what
  `PeriodicBoundaryIndex` already provides.** `select_spatial_domain` already handles
  wrap-crossing slices; no new wraparound logic is added.
- [ ] **`TestVisualizationCallback` support for patch-mode training.** Its active-day map
  and per-office-region performance diagrams assume one whole-domain timestep per sample
  and whole-domain `lats`/`lons`; adapting them to per-patch tiling and a buffered input
  shape is separate scope. `train()` skips building this callback when `patch_config` is
  set (`_should_build_test_visualization`), and the new ablation config sets
  `test_viz_every_n_epochs: null` explicitly so this is visible in the config itself, not
  just inferred from a runtime guard.
- [ ] **Runtime enforcement that `buffer_px` is compatible with the model's total
  downsampling stride.** `128 + 2*buffer_px` (and, if the core weren't already square,
  the latitude analog) must stay divisible by the product of `model_config.pool_size`
  across `model_config.levels - 1` pooling stages, or Keras's `Concatenate` layers raise a
  shape-mismatch error at model-build time — the same failure mode that already exists
  today for any other spatially-incompatible input size, unrelated to this change. This is
  documented in the new config's comments and exercised by an end-to-end smoke test
  (Phase 6) rather than re-implemented as a new, `model_config`-aware validation inside
  `PatchConfig`, which knows nothing about model architecture today and shouldn't need to.
- [ ] **General (non-patch-mode) flip augmentation.** The paper's augmentation is part of
  its patch regime specifically; `flip_probability` lives on `PatchConfig`, not as an
  independent `DatasetConfig` option for full-domain training.
- [ ] **Deduplicating repeated timestep reads within a batch.** In patch mode, two patches
  from the same timestep can land in one shuffled batch, causing that timestep to be read
  from the icechunk store twice. This is a performance nit, not a correctness bug (each
  read returns identical, correct data); not addressed here.

## Implementation Approach

**Technical Strategy:** Keep the existing "read once per batch, no dask graph" contract
of `FrontsPyDataset` intact. Patch mode maps each *global sample index* to
`(time_idx, patch_idx) = divmod(index, n_patches)`, so `_total`/`_order`/`__getitem__`'s
existing shuffle-by-permutation logic needs no changes — only `get_at_indices` grows a
patch-mode branch. The spatial buffer is threaded as plain grid-pixel widths end to end
(`PatchConfig.buffer_px` → widened `BoundingBox` at load time → sliced input array in
`FrontsPyDataset` → `pred_buffer_px` passed into the loss), never converted to physical
units except once, when computing the widened `BoundingBox` from the core domain's own
grid spacing.

**Key Architectural Decisions:**

1. **Decision:** Crop the loss's *pooled* prediction to match the target, rather than
   masking the loss or adding a `Cropping` layer to the model.
   - **Rationale:** This is the literature-established approach — Ronneberger et al.
     (2015)'s U-Net overlap-tile strategy: "the segmentation map only contains the pixels
     for which the full context is available in the input image... missing input data is
     extrapolated by mirroring" (https://arxiv.org/abs/1505.04597). Cropping *after*
     pooling (not before) additionally means the neighborhood pooling used by
     `fractions_skill_score`/`neighborhood_brier_score` genuinely reads real buffer-region
     pixel values for boundary cells, rather than falling back to the zero-padding it would
     use if handed an already-cropped (core-only) prediction.
   - **Trade-offs:** The loss functions gain a `pred_buffer_px` parameter and must accept
     `y_pred` at a different spatial size than `y_true` — a real change to their contract,
     but a narrow, well-tested one.
   - **Alternatives considered:** A `Cropping2D`/`Cropping3D` Keras layer on the model
     output would crop *before* pooling, discarding exactly the buffer context the
     neighborhood-pooling loss needs for its own boundary-cell accuracy. Masking the loss
     (scoring the full buffered output against a buffered-but-masked target) was considered
     and rejected: it requires buffering the target too (defeating the "targets are never
     buffered" design) or fabricating target values in the mask region.

2. **Decision:** `PatchConfig` lives on `DatasetConfig`, not `TrainConfig` or `ModelConfig`.
   - **Rationale:** Patch geometry and buffering are properties of how training samples are
     drawn from the data, matching where `coordinates`/`front_dilation`/`batch_size`
     already live.
   - **Trade-offs:** The loss builder (`train.py`'s `_compile`) has to read
     `data_cfg.patch_config.buffer_px` to build `pred_buffer_px` for the loss, adding one
     small cross-config read; simpler than duplicating `buffer_px` into `TrainConfig`.
   - **Alternatives considered:** A separate top-level `patch_config` YAML section (like
     `train_config`/`data_config`) was rejected — patch geometry has no independent meaning
     without `DatasetConfig.coordinates`, so nesting it there keeps that dependency explicit
     (`load_data_into_dataloader` raises if `patch_config` is set without `coordinates`).

3. **Decision:** Compute the buffered `BoundingBox` from the *core* box's own resolution
   at load time, then validate the loaded buffered array's shape against the expected
   `core + 2*buffer_px`, raising `ValueError` if the store doesn't have enough margin.
   - **Rationale:** `select_spatial_domain`'s `.sel()` silently returns fewer points than
     requested when a slice runs past the store's actual coverage (no error) — validating
     shape explicitly turns a silent train-on-wrong-shape bug into an immediate, clear
     failure.
   - **Trade-offs:** One extra `select_spatial_domain` call (on the core box) purely to
     read off `n_lat_core`/`n_lon_core`/`resolution_deg` before the buffered load.
   - **Alternatives considered:** Trusting the caller to pick a store with enough margin
     was rejected — a wrong `buffer_px` for a given store's coverage should fail loudly at
     data-loading time, not silently train on a smaller-than-intended buffer.

**Patterns to Follow:**
- Sample-index permutation/shuffling — see `FrontsPyDataset._order`/`on_epoch_end` at
  `fronts/data/datasets.py:126-148` (unchanged; patch mode reuses it against the expanded
  `_total`).
- Row-grouped precomputation for a small, bounded set of distinct values — see
  `losses._lat_dependent_pool`'s per-width row grouping at `fronts/layers/losses.py:260-293`
  (same spirit as precomputing `patch_lon_starts` once per `FrontsPyDataset`, since
  `n_patches` is always small).
- Config-driven spatial cropping via `utils.select_spatial_domain` — see
  `train.py:175-178`; the buffered load reuses this unchanged, just with a wider box.
- Nested-dataclass YAML config sections — see how `data_config.inputs_icechunk_config`
  (`utils.IcechunkStorageConfig`) already nests inside `DatasetConfig` and parses via plain
  `dacite` recursion with no extra type hook needed (`fronts/utils.py:369-393`); the new
  `patch_config: PatchConfig | None` field follows the same pattern.

## Implementation Phases

### Phase 1: Patch geometry config and pure helpers

**Objective:** Add `PatchConfig`, `DatasetConfig.patch_config`, `compute_patch_lon_starts`,
and `utils.expand_bounding_box` — no I/O, no `FrontsPyDataset` changes yet.

**Tasks:**
- [x] **Write the failing tests** for `PatchConfig` validation and `compute_patch_lon_starts`.
  - File: `tests/test_train.py` (new test classes, placed after the existing
    `class TestFrontsPyDatasetVolume:` block and before `def _build_small_unet`, so they sit
    alongside the other `fronts.data.datasets` coverage already in this file)

    ```python
    class TestPatchConfigValidation:
        def test_n_patches_must_be_positive(self):
            with pytest.raises(ValueError, match="n_patches"):
                PatchConfig(n_patches=0, patch_lon_width_px=4)

        def test_patch_lon_width_px_must_be_positive(self):
            with pytest.raises(ValueError, match="patch_lon_width_px"):
                PatchConfig(n_patches=1, patch_lon_width_px=0)

        def test_buffer_px_must_be_non_negative(self):
            with pytest.raises(ValueError, match="buffer_px"):
                PatchConfig(n_patches=1, patch_lon_width_px=4, buffer_px=-1)

        def test_flip_probability_must_be_in_unit_interval(self):
            with pytest.raises(ValueError, match="flip_probability"):
                PatchConfig(n_patches=1, patch_lon_width_px=4, flip_probability=1.5)

        def test_defaults(self):
            cfg = PatchConfig(n_patches=9, patch_lon_width_px=128)
            assert cfg.buffer_px == 0
            assert cfg.flip_probability == 0.0


    class TestComputePatchLonStarts:
        def test_nine_patches_evenly_spaced_across_288(self):
            starts = compute_patch_lon_starts(n_lon_core=288, patch_width=128, n_patches=9)
            np.testing.assert_array_equal(starts, [0, 20, 40, 60, 80, 100, 120, 140, 160])

        def test_single_patch_starts_at_zero(self):
            starts = compute_patch_lon_starts(n_lon_core=20, patch_width=8, n_patches=1)
            np.testing.assert_array_equal(starts, [0])

        def test_patch_wider_than_core_raises(self):
            with pytest.raises(ValueError, match="exceeds"):
                compute_patch_lon_starts(n_lon_core=10, patch_width=12, n_patches=2)

        def test_last_patch_ends_exactly_at_core_width(self):
            starts = compute_patch_lon_starts(n_lon_core=12, patch_width=4, n_patches=3)
            assert starts[-1] + 4 == 12
    ```
  - Add `PatchConfig, compute_patch_lon_starts` to the existing
    `from fronts.data.datasets import DatasetConfig, FrontsPyDataset` import at
    `tests/test_train.py:14`.

- [x] **Run it, watch it fail:**
  `pixi run pytest tests/test_train.py -k "PatchConfig or ComputePatchLonStarts" -v`
  → expect FAIL (`ImportError: cannot import name 'PatchConfig'`)

- [x] **Implement the minimal code.**
  - File: `src/fronts/data/datasets.py` — insert before `class DatasetConfig:` (currently
    line 16)

    ```python
    def compute_patch_lon_starts(n_lon_core: int, patch_width: int, n_patches: int) -> np.ndarray:
        """Evenly spaced starting pixel offsets for sliding longitude windows.

        Reproduces "nine pairs of images evenly spaced along the longitude" (Justin et al.
        2025): ``n_patches`` windows of ``patch_width`` pixels tiled across a core domain of
        ``n_lon_core`` pixels, first window flush with the west edge, last flush with the
        east edge, evenly spaced in between.

        Args:
            n_lon_core: Width of the (unbuffered) core longitude domain, in grid pixels.
            patch_width: Width of each patch's core region, in grid pixels.
            n_patches: Number of patch positions.

        Returns:
            Integer array of shape (n_patches,), each a 0-indexed starting pixel offset
            into the core domain.

        Raises:
            ValueError: If patch_width exceeds n_lon_core, or n_patches < 1.
        """
        if patch_width > n_lon_core:
            raise ValueError(f"patch_lon_width_px ({patch_width}) exceeds the core domain width ({n_lon_core}).")
        if n_patches < 1:
            raise ValueError(f"n_patches must be >= 1, got {n_patches}")
        if n_patches == 1:
            return np.array([0], dtype=int)
        return np.round(np.linspace(0, n_lon_core - patch_width, n_patches)).astype(int)


    @dataclasses.dataclass
    class PatchConfig:
        """Sliding-window longitude patch extraction with an optional input-only context buffer.

        Reproduces Justin et al. (2025)'s original training regime: nine 128x128 patches per
        timestep, evenly spaced along longitude, each independently flipped along latitude
        and/or longitude with some probability. Requires ``DatasetConfig.coordinates`` to be
        set (defines the core domain patches are tiled from); ``load_data_into_dataloader``
        raises if ``patch_config`` is set without it.

        Attributes:
            n_patches: Number of evenly-spaced longitude patch positions per timestep.
            patch_lon_width_px: Width of each patch's core (unbuffered, loss-supervised)
                region along longitude, in grid pixels. Every patch's latitude extent is the
                full height of ``DatasetConfig.coordinates`` — no latitude tiling, since
                CONUS's 128-point latitude range already equals the paper's patch height.
            buffer_px: Extra context pixels appended on every side (north, south, east, west)
                of each patch's core region, for the *input* only — never the target. 0
                disables buffering. ``patch_lon_width_px + 2 * buffer_px`` (and the core
                latitude height + 2 * buffer_px) must stay divisible by the model's total
                downsampling stride (product of ``model_config.pool_size`` across
                ``model_config.levels - 1`` pooling stages) or the model fails to build —
                see the "What We're NOT Doing" note on stride validation in
                docs/rse/specs/plan-patch-buffer-training.md.
            flip_probability: Independent per-axis probability of flipping a training patch
                along latitude and along longitude. 0.25 reproduces the paper's rate (a
                1 - (1 - p)^2 = 43.75% chance of at least one flip at p=0.25). Applied only
                to the train split.
        """

        n_patches: int
        patch_lon_width_px: int
        buffer_px: int = 0
        flip_probability: float = 0.0

        def __post_init__(self) -> None:
            """Validate patch geometry and augmentation parameters."""
            if self.n_patches < 1:
                raise ValueError(f"n_patches must be >= 1, got {self.n_patches}")
            if self.patch_lon_width_px < 1:
                raise ValueError(f"patch_lon_width_px must be >= 1, got {self.patch_lon_width_px}")
            if self.buffer_px < 0:
                raise ValueError(f"buffer_px must be >= 0, got {self.buffer_px}")
            if not 0.0 <= self.flip_probability <= 1.0:
                raise ValueError(f"flip_probability must be in [0, 1], got {self.flip_probability}")
    ```
  - Add a `patch_config: PatchConfig | None = None` field at the end of `DatasetConfig`
    (`src/fronts/data/datasets.py:73`, right after `pressure_levels`).

- [x] **Write the failing test** for `expand_bounding_box`.
  - File: `tests/test_utils.py` (new class, after `class TestSelectSpatialDomain:`)

    ```python
    class TestExpandBoundingBox:
        def test_widens_all_four_sides_by_buffer_times_resolution(self):
            bb = utils.BoundingBox(lat_min=20.0, lat_max=40.0, lon_min=120.0, lon_max=150.0)
            expanded = utils.expand_bounding_box(bb, buffer_px=2, resolution_deg=10.0)
            assert expanded == utils.BoundingBox(lat_min=0.0, lat_max=60.0, lon_min=100.0, lon_max=170.0)

        def test_zero_buffer_is_a_no_op(self):
            bb = utils.BoundingBox(lat_min=20.0, lat_max=40.0, lon_min=120.0, lon_max=150.0)
            assert utils.expand_bounding_box(bb, buffer_px=0, resolution_deg=10.0) == bb
    ```

- [x] **Run it, watch it fail:**
  `pixi run pytest tests/test_utils.py::TestExpandBoundingBox -v`
  → expect FAIL (`AttributeError: module 'fronts.utils' has no attribute 'expand_bounding_box'`)

- [x] **Implement the minimal code.**
  - File: `src/fronts/utils.py` — insert after `select_spatial_domain` (currently ending at
    line 176), before `select_pressure_levels`

    ```python
    def expand_bounding_box(bb: BoundingBox, buffer_px: int, resolution_deg: float) -> BoundingBox:
        """Widen a BoundingBox by ``buffer_px`` grid cells on every side.

        Args:
            bb: Core bounding box.
            buffer_px: Number of grid cells to add on each side (lat and lon).
            resolution_deg: Grid spacing in degrees, used to convert pixels to degrees.

        Returns:
            A new BoundingBox widened by ``buffer_px * resolution_deg`` degrees on every side.
        """
        margin = buffer_px * resolution_deg
        return BoundingBox(
            lat_min=bb.lat_min - margin,
            lat_max=bb.lat_max + margin,
            lon_min=bb.lon_min - margin,
            lon_max=bb.lon_max + margin,
        )
    ```

- [x] **Run it, watch it pass:**
  `pixi run pytest tests/test_train.py -k "PatchConfig or ComputePatchLonStarts" tests/test_utils.py::TestExpandBoundingBox -v`
  → expect PASS
- [x] **Commit:** `git commit -m "feat: add PatchConfig, compute_patch_lon_starts, expand_bounding_box"`

**Dependencies:** none.

**Verification:**
- [x] `pixi run pytest tests/test_train.py -k "PatchConfig or ComputePatchLonStarts" tests/test_utils.py::TestExpandBoundingBox -v` → all PASS

### Phase 2: `FrontsPyDataset` patch sampling and flip augmentation

**Objective:** `FrontsPyDataset` draws patch samples (with buffer) instead of whole-timestep
samples when `data_config.patch_config` is set, and applies per-axis flip augmentation when
`augment=True`.

**Tasks:**
- [x] **Write the failing tests.**
  - File: `tests/test_train.py` (new class, after `class TestFrontsPyDataset:`)

    ```python
    class TestFrontsPyDatasetPatchMode:
        """PatchConfig support: sliding longitude windows with an optional input-only buffer."""

        _N_TIME = 2
        _N_LAT_CORE = 6
        _N_LON_CORE = 12
        _BUFFER = 2
        _PATCH_WIDTH = 4
        _N_PATCHES = 3  # starts = [0, 4, 8] for a 12-wide core and a 4-wide patch

        def _make_ds(self, flip_probability=0.0, augment=False):
            n_lat_buf = self._N_LAT_CORE + 2 * self._BUFFER
            n_lon_buf = self._N_LON_CORE + 2 * self._BUFFER
            # Every pixel gets a unique value (lat_idx * 1000 + lon_idx) so mis-slicing is caught.
            buffered_vals = (np.arange(n_lat_buf)[:, None] * 1000 + np.arange(n_lon_buf)[None, :]).astype(np.float32)
            input_ds = xr.Dataset(
                {
                    "temperature": xr.DataArray(
                        np.broadcast_to(buffered_vals, (self._N_TIME, n_lat_buf, n_lon_buf)).copy(),
                        dims=["time", "latitude", "longitude"],
                        coords={"time": np.arange(self._N_TIME)},
                    )
                }
            )
            core_vals = (np.arange(self._N_LAT_CORE)[:, None] + np.arange(self._N_LON_CORE)[None, :]) % 2
            target_da = xr.DataArray(
                np.broadcast_to(core_vals, (self._N_TIME, self._N_LAT_CORE, self._N_LON_CORE)).astype(np.int32).copy(),
                dims=["time", "latitude", "longitude"],
                coords={"time": np.arange(self._N_TIME)},
            )
            dummy_store = IcechunkStorageConfig(store_path="unused", branch_name="main")
            patch_config = PatchConfig(
                n_patches=self._N_PATCHES,
                patch_lon_width_px=self._PATCH_WIDTH,
                buffer_px=self._BUFFER,
                flip_probability=flip_probability,
            )
            data_config = DatasetConfig(
                inputs_icechunk_config=dummy_store,
                targets_icechunk_config=dummy_store,
                variables=["temperature"],
                test_years=[],
                val_years=[],
                patch_config=patch_config,
            )
            return FrontsPyDataset(input_ds, target_da, data_config, batch_size=1, augment=augment, seed=0)

        def test_total_samples_equals_time_times_patches(self):
            ds = self._make_ds()
            assert ds.n_samples == self._N_TIME * self._N_PATCHES

        def test_input_patch_width_includes_buffer_on_both_sides(self):
            ds = self._make_ds()
            x, _ = ds.get_at_indices(np.array([0]))
            assert x.shape == (1, self._N_LAT_CORE + 2 * self._BUFFER, self._PATCH_WIDTH + 2 * self._BUFFER, 1)

        def test_target_patch_has_no_buffer(self):
            ds = self._make_ds()
            _, y = ds.get_at_indices(np.array([0]))
            assert y.shape[1:3] == (self._N_LAT_CORE, self._PATCH_WIDTH)

        def test_input_patch_matches_manual_slice_with_buffer(self):
            ds = self._make_ds()
            starts = compute_patch_lon_starts(self._N_LON_CORE, self._PATCH_WIDTH, self._N_PATCHES)
            n_lat_buf = self._N_LAT_CORE + 2 * self._BUFFER
            for global_idx in range(ds.n_samples):
                x, _ = ds.get_at_indices(np.array([global_idx]))
                _, patch_idx = divmod(global_idx, self._N_PATCHES)
                start = starts[patch_idx]
                expected_lon = np.arange(start, start + self._PATCH_WIDTH + 2 * self._BUFFER)
                expected = (np.arange(n_lat_buf)[:, None] * 1000 + expected_lon[None, :]).astype(np.float32)
                np.testing.assert_allclose(x[0, :, :, 0], expected)

        def test_flip_probability_one_always_flips_both_axes_when_augmenting(self):
            ds_flip = self._make_ds(flip_probability=1.0, augment=True)
            ds_noflip = self._make_ds(flip_probability=0.0, augment=True)
            x_flip, y_flip = ds_flip.get_at_indices(np.array([0]))
            x_noflip, y_noflip = ds_noflip.get_at_indices(np.array([0]))
            np.testing.assert_allclose(x_flip[0], x_noflip[0][::-1, ::-1, :])
            np.testing.assert_allclose(y_flip[0], y_noflip[0][::-1, ::-1, :])

        def test_augment_false_ignores_flip_probability(self):
            ds = self._make_ds(flip_probability=1.0, augment=False)
            x, _ = ds.get_at_indices(np.array([0]))
            starts = compute_patch_lon_starts(self._N_LON_CORE, self._PATCH_WIDTH, self._N_PATCHES)
            n_lat_buf = self._N_LAT_CORE + 2 * self._BUFFER
            expected_lon = np.arange(starts[0], starts[0] + self._PATCH_WIDTH + 2 * self._BUFFER)
            expected = (np.arange(n_lat_buf)[:, None] * 1000 + expected_lon[None, :]).astype(np.float32)
            np.testing.assert_allclose(x[0, :, :, 0], expected)
    ```

- [x] **Run it, watch it fail:**
  `pixi run pytest tests/test_train.py::TestFrontsPyDatasetPatchMode -v`
  → expect FAIL (`TypeError: FrontsPyDataset.__init__() got an unexpected keyword argument 'augment'`)

- [x] **Implement the minimal code.**
  - File: `src/fronts/data/datasets.py` — `FrontsPyDataset.__init__`
    (currently lines 104-128): add an `augment: bool = False` parameter, stored as
    `self.augment = augment`, and precompute patch geometry when patch mode is active:

    ```python
    def __init__(
        self,
        input_ds: xr.Dataset,
        target_da: xr.DataArray,
        data_config: DatasetConfig,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        workers: int = 1,
        max_queue_size: int = 10,
        drop_remainder: bool = False,
        augment: bool = False,
    ):
        super().__init__(workers=workers, max_queue_size=max_queue_size)
        if input_ds.sizes["time"] != target_da.sizes["time"]:
            raise ValueError(
                f"Input and target time lengths differ: {input_ds.sizes['time']} vs {target_da.sizes['time']}"
            )
        self.input_ds = input_ds.copy()
        self.target_da = target_da.copy()
        self.data_config = data_config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder
        self.augment = augment
        self._n_patches = data_config.patch_config.n_patches if data_config.patch_config is not None else 1
        if data_config.patch_config is not None:
            self._patch_lon_starts = compute_patch_lon_starts(
                n_lon_core=target_da.sizes["longitude"],
                patch_width=data_config.patch_config.patch_lon_width_px,
                n_patches=data_config.patch_config.n_patches,
            )
        self._rng = np.random.default_rng(seed)
        self._order = self._rng.permutation(self._total) if shuffle else np.arange(self._total)
    ```
  - Update `_total` (currently lines 130-132) to reflect the expanded sample count:

    ```python
    @property
    def _total(self) -> int:
        return self.input_ds.sizes["time"] * self._n_patches
    ```
  - Add `patch_sample_index` (used by Phase 5's visualization-skip logic and directly
    testable) right after `n_samples` (currently lines 134-137):

    ```python
    def patch_sample_index(self, time_idx: int, patch_idx: int = 0) -> int:
        """Map a timestep index to a global sample index, picking one patch within it.

        In patch mode, ``get_at_indices`` expects global sample indices
        (``time_idx * n_patches + patch_idx``), not raw timestep indices. Non-patch mode
        returns ``time_idx`` unchanged, since ``n_patches`` is 1.
        """
        return time_idx * self._n_patches + patch_idx
    ```
  - Split `get_at_indices` (currently lines 150-176) into a dispatcher plus the new
    patch-mode path:

    ```python
    def get_at_indices(self, idxs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns the (input, target) arrays at arbitrary global sample indices.

        Unlike ``__getitem__``, ``idxs`` need not be batch-sized or in ``_order``'s
        epoch sequence — used by callers that need specific samples directly (e.g. a
        test-set visualization callback selecting one active day or a random subsample).
        In patch mode, ``idxs`` are global sample indices (see ``patch_sample_index``),
        not raw timestep indices.
        """
        if self.data_config.patch_config is not None:
            return self._get_patches_at_indices(idxs)
        x_xarray = self.input_ds.isel(time=idxs)
        y_da = self.target_da.isel(time=idxs)

        if self.data_config.volume_inputs:
            x = inputs.inputs_ds_to_volume_dataarray(x_xarray, self.data_config.variables).values
        else:
            x = inputs.inputs_ds_to_dataarray(x_xarray, self.data_config.variables).values

        y_da = targets.one_hot_encode_to_dataarray(targets.remap_fronts(y_da))
        if self.data_config.front_dilation > 0:
            y_da = targets.dilate_fronts(y_da, self.data_config.front_dilation)

        y = y_da.values
        return x, y

    def _get_patches_at_indices(self, idxs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Patch-mode ``get_at_indices``: slices a buffered input window and unbuffered
        target window per sample, then applies flip augmentation if ``self.augment``."""
        pc = self.data_config.patch_config
        time_idxs = idxs // pc.n_patches
        patch_idxs = idxs % pc.n_patches

        x_xarray = self.input_ds.isel(time=time_idxs)
        y_da = self.target_da.isel(time=time_idxs)

        if self.data_config.volume_inputs:
            x_full = inputs.inputs_ds_to_volume_dataarray(x_xarray, self.data_config.variables).values
        else:
            x_full = inputs.inputs_ds_to_dataarray(x_xarray, self.data_config.variables).values

        y_da = targets.one_hot_encode_to_dataarray(targets.remap_fronts(y_da))
        if self.data_config.front_dilation > 0:
            y_da = targets.dilate_fronts(y_da, self.data_config.front_dilation)
        y_full = y_da.values

        width = pc.patch_lon_width_px
        buf = pc.buffer_px
        starts = self._patch_lon_starts
        x = np.stack(
            [x_full[i, :, starts[p] : starts[p] + width + 2 * buf, ...] for i, p in enumerate(patch_idxs)], axis=0
        )
        y = np.stack(
            [y_full[i, :, starts[p] : starts[p] + width, :] for i, p in enumerate(patch_idxs)], axis=0
        )

        if self.augment and pc.flip_probability > 0:
            x, y = self._apply_flip_augmentation(x, y, pc.flip_probability)
        return x, y

    def _apply_flip_augmentation(
        self, x: np.ndarray, y: np.ndarray, flip_probability: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Independently flips each sample along latitude and/or longitude with probability
        ``flip_probability`` per axis — Justin et al. (2025)'s augmentation."""
        n = x.shape[0]
        flip_lat = self._rng.random(n) < flip_probability
        flip_lon = self._rng.random(n) < flip_probability
        for i in range(n):
            if flip_lat[i]:
                x[i] = x[i, ::-1, ...]
                y[i] = y[i, ::-1, ...]
            if flip_lon[i]:
                x[i] = x[i, :, ::-1, ...]
                y[i] = y[i, :, ::-1, ...]
        return x.copy(), y.copy()
    ```
  - Update the module import at the top of `src/fronts/data/datasets.py` (currently no
    change needed — `np`, `xr`, `tf`, `inputs`, `targets` are already imported).

- [x] **Run it, watch it pass:**
  `pixi run pytest tests/test_train.py::TestFrontsPyDatasetPatchMode tests/test_train.py::TestFrontsPyDataset -v`
  → expect PASS (including the pre-existing non-patch-mode tests, unaffected)
- [x] **Commit:** `git commit -m "feat: patch-mode sampling and flip augmentation in FrontsPyDataset"`

**Dependencies:** Phase 1 (`PatchConfig`, `compute_patch_lon_starts`).

**Verification:**
- [x] `pixi run pytest tests/test_train.py -k "FrontsPyDataset" -v` → all PASS

### Phase 3: Buffered domain loading in `load_data_into_dataloader`

**Objective:** When `patch_config` is set, `inputs_ds` loads the core domain widened by
`buffer_px` grid cells; `targets_da` stays at the unbuffered core; a mismatch between the
requested buffer and the store's actual coverage raises a clear error; `augment` threads
through to `FrontsPyDataset`.

**Tasks:**
- [x] **Write the failing tests.**
  - File: `tests/test_train.py` (new class, after `class TestLoadDataIntoDataloaderPressureLevels:`)

    ```python
    class TestLoadDataIntoDataloaderPatchBuffer:
        """patch_config.buffer_px must widen only inputs_ds's loaded domain, never targets_da's."""

        _TIMES = pd.date_range("2020-01-01", periods=4, freq="6h")
        _LAT = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
        _LON = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0])

        def _write_store(self, tmp_path, name: str, var_name: str) -> IcechunkStorageConfig:
            storage_config = IcechunkStorageConfig(store_path=str(tmp_path / name), branch_name="main")
            ds = xr.Dataset(
                {
                    var_name: xr.DataArray(
                        np.zeros((len(self._TIMES), len(self._LAT), len(self._LON)), dtype=np.float32),
                        dims=["time", "latitude", "longitude"],
                        coords={"time": self._TIMES, "latitude": self._LAT, "longitude": self._LON},
                    )
                }
            )
            write_or_append_icechunk_store(storage_config, ds)
            return storage_config

        def _data_config(self, tmp_path, coordinates, buffer_px):
            return DatasetConfig(
                inputs_icechunk_config=self._write_store(tmp_path, "inputs", "temperature"),
                targets_icechunk_config=self._write_store(tmp_path, "targets", "identifier"),
                variables=["temperature"],
                test_years=[2020],
                val_years=[],
                coordinates=coordinates,
                patch_config=PatchConfig(n_patches=2, patch_lon_width_px=2, buffer_px=buffer_px)
                if buffer_px is not None
                else None,
            )

        def test_inputs_widened_by_buffer_targets_stay_core(self, tmp_path):
            from fronts.utils import BoundingBox

            data_config = self._data_config(
                tmp_path, BoundingBox(lat_min=20.0, lat_max=40.0, lon_min=120.0, lon_max=150.0), buffer_px=1
            )
            test_dataset = load_data_into_dataloader(data_config, split="test", seed=0)
            assert test_dataset.input_ds.sizes["latitude"] == 5
            assert test_dataset.input_ds.sizes["longitude"] == 6
            assert test_dataset.target_da.sizes["latitude"] == 3
            assert test_dataset.target_da.sizes["longitude"] == 4

        def test_zero_buffer_matches_unbuffered_domain(self, tmp_path):
            from fronts.utils import BoundingBox

            data_config = self._data_config(
                tmp_path, BoundingBox(lat_min=20.0, lat_max=40.0, lon_min=120.0, lon_max=150.0), buffer_px=0
            )
            test_dataset = load_data_into_dataloader(data_config, split="test", seed=0)
            assert test_dataset.input_ds.sizes["latitude"] == 3
            assert test_dataset.input_ds.sizes["longitude"] == 4

        def test_buffer_past_store_edge_raises(self, tmp_path):
            from fronts.utils import BoundingBox

            data_config = self._data_config(
                tmp_path, BoundingBox(lat_min=0.0, lat_max=40.0, lon_min=120.0, lon_max=150.0), buffer_px=1
            )
            with pytest.raises(ValueError, match="buffer_px"):
                load_data_into_dataloader(data_config, split="test", seed=0)

        def test_patch_config_without_coordinates_raises(self, tmp_path):
            data_config = DatasetConfig(
                inputs_icechunk_config=self._write_store(tmp_path, "inputs", "temperature"),
                targets_icechunk_config=self._write_store(tmp_path, "targets", "identifier"),
                variables=["temperature"],
                test_years=[2020],
                val_years=[],
                coordinates=None,
                patch_config=PatchConfig(n_patches=2, patch_lon_width_px=2, buffer_px=0),
            )
            with pytest.raises(ValueError, match="coordinates"):
                load_data_into_dataloader(data_config, split="test", seed=0)

        def test_augment_flag_threaded_to_dataset(self, tmp_path):
            from fronts.utils import BoundingBox

            data_config = self._data_config(
                tmp_path, BoundingBox(lat_min=20.0, lat_max=40.0, lon_min=120.0, lon_max=150.0), buffer_px=0
            )
            test_dataset = load_data_into_dataloader(data_config, split="test", seed=0, augment=True)
            assert test_dataset.augment is True
    ```

- [x] **Run it, watch it fail:**
  `pixi run pytest tests/test_train.py::TestLoadDataIntoDataloaderPatchBuffer -v`
  → expect FAIL (`TypeError: load_data_into_dataloader() got an unexpected keyword argument 'augment'`,
  and the widen/raise tests fail since no such behavior exists yet)

- [x] **Implement the minimal code.**
  - File: `src/fronts/train.py` — `load_data_into_dataloader` signature (currently lines
    117-123): add `augment: bool = False`.
  - File: `src/fronts/train.py` — the `coordinates` block (currently lines 175-178),
    replaced with:

    ```python
    if data_config.patch_config is not None and data_config.coordinates is None:
        raise ValueError("data_config.patch_config requires data_config.coordinates to be set.")

    if data_config.coordinates is not None:
        logger.info("Restricting to spatial domain: %s", data_config.coordinates)
        if data_config.patch_config is not None and data_config.patch_config.buffer_px > 0:
            core_inputs_ds = utils.select_spatial_domain(inputs_ds, data_config.coordinates)
            resolution_deg = float(np.median(np.abs(np.diff(core_inputs_ds["latitude"].values))))
            buffer_px = data_config.patch_config.buffer_px
            buffered_bbox = utils.expand_bounding_box(data_config.coordinates, buffer_px, resolution_deg)
            inputs_ds = utils.select_spatial_domain(inputs_ds, buffered_bbox)
            expected_lat = core_inputs_ds.sizes["latitude"] + 2 * buffer_px
            expected_lon = core_inputs_ds.sizes["longitude"] + 2 * buffer_px
            if inputs_ds.sizes["latitude"] != expected_lat or inputs_ds.sizes["longitude"] != expected_lon:
                raise ValueError(
                    f"patch_config.buffer_px={buffer_px} extends past the input store's available "
                    f"domain: expected buffered shape (lat={expected_lat}, lon={expected_lon}), got "
                    f"(lat={inputs_ds.sizes['latitude']}, lon={inputs_ds.sizes['longitude']}). Widen "
                    "the icechunk store's coverage or reduce buffer_px."
                )
        else:
            inputs_ds = utils.select_spatial_domain(inputs_ds, data_config.coordinates)
        targets_da = utils.select_spatial_domain(targets_da, data_config.coordinates)
    ```
  - File: `src/fronts/train.py` — the `FrontsPyDataset(...)` construction at the end of
    `load_data_into_dataloader` (currently lines 217-227): add `augment=augment,`.

- [x] **Run it, watch it pass:**
  `pixi run pytest tests/test_train.py::TestLoadDataIntoDataloaderPatchBuffer -v`
  → expect PASS
- [x] **Commit:** `git commit -m "feat: buffered spatial domain loading for patch-mode training"`

**Dependencies:** Phase 1, Phase 2.

**Verification:**
- [x] `pixi run pytest tests/test_train.py -k "LoadDataIntoDataloader" -v` → all PASS

### Phase 4: Crop-after-pooling in `neighborhood_brier_score`

**Objective:** `neighborhood_brier_score` accepts a `pred_buffer_px` parameter: pool the
full (buffered) prediction, crop the pooled result to match the (unbuffered) target's
shape, then score — using genuine buffer-region values rather than falling back to
zero-padding.

**Tasks:**
- [x] **Write the failing tests.**
  - File: `tests/layers/test_losses.py` (new class, after the `TestNeighborhoodBrierScore`
    class's existing tests)

    ```python
    class TestNeighborhoodBrierScorePredBuffer:
        def test_pred_buffer_px_zero_matches_default_behavior(self):
            y_true = _with_front_row(_one_hot_background(), row=4)
            y_pred = _with_front_row(_one_hot_background(), row=5)
            default = losses.neighborhood_brier_score(latitudes=EQUATOR_LATITUDES, tolerance_km=25.0)
            explicit_zero = losses.neighborhood_brier_score(
                latitudes=EQUATOR_LATITUDES, tolerance_km=25.0, pred_buffer_px=0
            )
            np.testing.assert_allclose(default(y_true, y_pred).numpy(), explicit_zero(y_true, y_pred).numpy())

        def test_pred_buffer_px_accepts_larger_prediction_and_returns_finite_scalar(self):
            y_true = _one_hot_background()
            buffer_px = 2
            y_pred = np.zeros((N_BATCH, N_H + 2 * buffer_px, N_W + 2 * buffer_px, N_CLASSES), dtype=np.float32)
            y_pred[..., 0] = 1.0
            loss_fn = losses.neighborhood_brier_score(
                latitudes=EQUATOR_LATITUDES, tolerance_km=25.0, pred_buffer_px=buffer_px
            )
            result = loss_fn(y_true, y_pred).numpy()
            assert result.shape == (N_BATCH,)
            assert np.all(np.isfinite(result))

        def test_pred_buffer_px_pooling_reads_real_buffer_values_not_zero_padding(self):
            """Pooling the buffered prediction must use the declared buffer-ring content at
            cells whose pooling window reaches into the buffer -- otherwise pred_buffer_px
            would be a no-op and defeat the point of supplying buffer context.

            Keras's AveragePooling2D(padding="same") already excludes padding from its
            divisor (a uniform field pools to itself even at the domain edge -- verified
            empirically), so "zero-padding dilutes the edge" is not the right mental model
            here; edge cells are always valid-cell-normalized, over however many cells the
            window actually has access to. That's exactly the lever this test uses: with a
            front pixel at the very edge (row 0, col 4) and a core prediction that matches
            the truth exactly, the *unbuffered* loss is exactly zero (O_n and M_n both
            valid-cell-average the identical 6 in-domain neighbors). Declaring a buffer ring
            that continues the same front pattern one row further north changes the
            buffered M_n's neighbor count from 6 to 9 real cells -- a different (and here,
            nonzero) average -- proving the crop happens after a pooling call that genuinely
            saw the buffer pixels, not before it.
            """
            y_true = _one_hot_background().copy()
            y_true[:, 0, 4, 0] = 0.0
            y_true[:, 0, 4, 1] = 1.0  # a single front pixel at the very edge (row 0, col 4)
            core_pred = y_true.copy()
            buffer_px = 1
            buffered_pred = np.zeros(
                (N_BATCH, N_H + 2 * buffer_px, N_W + 2 * buffer_px, N_CLASSES), dtype=np.float32
            )
            buffered_pred[..., 0] = 1.0
            buffered_pred[:, buffer_px:-buffer_px, buffer_px:-buffer_px, :] = core_pred
            # North buffer row: the front continues one row further north, aligned under
            # core col 4 (buffered col index 4 + buffer_px).
            buffered_pred[:, 0, :, 0] = 1.0
            buffered_pred[:, 0, :, 1] = 0.0
            buffered_pred[:, 0, 4 + buffer_px, 0] = 0.0
            buffered_pred[:, 0, 4 + buffer_px, 1] = 1.0

            cw = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]  # isolate the front class
            unbuffered_loss = losses.neighborhood_brier_score(
                latitudes=EQUATOR_LATITUDES, tolerance_km=25.0, class_weights=cw
            )
            buffered_loss = losses.neighborhood_brier_score(
                latitudes=EQUATOR_LATITUDES, tolerance_km=25.0, class_weights=cw, pred_buffer_px=buffer_px
            )

            unbuffered_value = unbuffered_loss(y_true, core_pred).numpy().mean()
            buffered_value = buffered_loss(y_true, buffered_pred).numpy().mean()

            assert unbuffered_value == pytest.approx(0.0, abs=1e-7), (
                "sanity check: an unbuffered perfect-core prediction must score exactly zero "
                "since O_n and M_n valid-cell-average the identical 6 in-domain neighbors"
            )
            assert buffered_value > 1e-7, (
                "the buffered prediction differs from the unbuffered one only in its buffer "
                "ring, so a nonzero loss here proves the pooling step actually read those "
                "buffer pixels (averaging over 9 cells) rather than stopping at the domain "
                "edge (averaging over 6)"
            )

        def test_pred_buffer_px_pixel_term_uses_cropped_raw_prediction(self):
            """include_pixel's un-pooled term must compare against the cropped raw
            prediction, not the full buffered one -- otherwise shapes wouldn't broadcast."""
            y_true = _with_front_row(_one_hot_background(), row=4)
            buffer_px = 1
            core_pred = _one_hot_background()  # wrong everywhere in the core (a miss)
            buffered_pred = np.zeros(
                (N_BATCH, N_H + 2 * buffer_px, N_W + 2 * buffer_px, N_CLASSES), dtype=np.float32
            )
            buffered_pred[..., 0] = 1.0
            buffered_pred[:, buffer_px:-buffer_px, buffer_px:-buffer_px, :] = core_pred
            loss_fn = losses.neighborhood_brier_score(
                latitudes=EQUATOR_LATITUDES,
                tolerance_km=25.0,
                pred_buffer_px=buffer_px,
                include_pixel=True,
                pixel_weight=1.0,
            )
            result = loss_fn(y_true, buffered_pred).numpy()
            assert np.all(np.isfinite(result)) and result.mean() > 0.0
    ```

- [x] **Run it, watch it fail:**
  `pixi run pytest tests/layers/test_losses.py::TestNeighborhoodBrierScorePredBuffer -v`
  → expect FAIL (`TypeError: neighborhood_brier_score() got an unexpected keyword argument 'pred_buffer_px'`)

- [x] **Implement the minimal code.**
  - File: `src/fronts/layers/losses.py` — `neighborhood_brier_score` signature (currently
    lines 296-307): add `pred_buffer_px: int = 0` and document it in the docstring's Args
    (after `include_pixel`):

    ```python
    def neighborhood_brier_score(
        latitudes: np.ndarray | Sequence[float],
        resolution_deg: float | None = None,
        tolerance_km: float = 25.0,
        include_pixel: bool = False,
        pixel_weight: float = 0.1,
        pred_buffer_px: int = 0,
        class_weights: list[int | float] | None = None,
        periodic_lon: bool = False,
        max_half_x: int = 128,
        max_distinct_widths: int | None = 8,
        lat_dependent_pool: bool = False,
    ) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    ```
    Docstring addition, right after the `include_pixel` Args entry:
    ```
            pred_buffer_px: If > 0, y_pred is expected to carry this many extra pixels of
                context on every spatial side beyond y_true's shape (e.g. from a patch
                trained with an input-only buffer — see fronts.data.datasets.PatchConfig).
                The neighborhood pool runs on the full buffered y_pred first; only the
                *pooled* result is then cropped by pred_buffer_px on every side before
                scoring against y_true, so boundary cells use real buffer-region context
                instead of falling back to zero-padding (the overlap-tile strategy from
                Ronneberger et al. 2015, https://arxiv.org/abs/1505.04597). 0 (default)
                requires y_pred and y_true to share the same shape, matching prior behavior.
    ```
  - Add a small crop helper right before `neighborhood_brier_score` (after
    `_lat_dependent_pool`, currently ending at line 293):

    ```python
    def _crop_pred_buffer(field: tf.Tensor, buffer_px: int) -> tf.Tensor:
        """Crops ``buffer_px`` pixels off every side of the latitude/longitude axes."""
        if buffer_px == 0:
            return field
        return field[:, buffer_px:-buffer_px, buffer_px:-buffer_px, :]
    ```
  - Update `nbs_loss`'s body (currently lines 362-390) to crop `M_n` after pooling and crop
    the raw prediction before the pixel term:

    ```python
    @tf.function
    def nbs_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the neighborhood Brier loss for a batch.

        Args:
            y_true: One-hot encoded tensor containing labels.
            y_pred: Tensor containing model predictions. When pred_buffer_px > 0, this is
                pred_buffer_px pixels wider than y_true on every spatial side.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        spatial_axes = list(range(1, len(y_true.shape)))

        def _brier(obs: tf.Tensor, mod: tf.Tensor) -> tf.Tensor:
            se = tf.math.square(obs - mod)
            if cw is not None:
                se *= cw
            return tf.reduce_mean(se, axis=spatial_axes)

        if lat_dependent_pool:
            O_n = _lat_dependent_pool(y_true, plan["half_y"], plan["half_x"], periodic_lon)
            M_n = _lat_dependent_pool(y_pred, plan["half_y"], plan["half_x"], periodic_lon)
        else:
            O_n = isotropic_pool(y_true)
            M_n = isotropic_pool(y_pred)
        M_n = _crop_pred_buffer(M_n, pred_buffer_px)
        total = _brier(O_n, M_n)
        if include_pixel:
            y_pred_core = _crop_pred_buffer(y_pred, pred_buffer_px)
            total += float(pixel_weight) * _brier(y_true, y_pred_core)
        return total / weight_sum

    return nbs_loss
    ```
    (`spatial_axes` now derives from `y_true.shape` instead of `y_pred.shape` — both have
    the same rank regardless of buffering, so this is a no-op for existing callers and
    correct once `y_pred`'s spatial size can legitimately differ from `y_true`'s.)

- [x] **Run it, watch it pass:**
  `pixi run pytest tests/layers/test_losses.py -v`
  → expect PASS (all existing + new tests)
- [x] **Commit:** `git commit -m "feat: crop-after-pooling pred_buffer_px support in neighborhood_brier_score"`

**Dependencies:** none (independent of Phases 1-3; can be built/tested in parallel).

**Verification:**
- [x] `pixi run pytest tests/layers/test_losses.py -v` → all PASS

### Phase 5: `train.py` wiring

**Objective:** Thread `pred_buffer_px` from `data_cfg.patch_config` into the compiled loss,
fix the `latitudes` source bug the buffered `input_ds` would otherwise trigger, skip the
(unsupported) periodic visualization callback in patch mode, and pass `augment=True` only
for the train split.

**Tasks:**
- [x] **Write the failing tests.**
  - File: `tests/test_train.py` (new classes, after `class TestBuildLoss:`)

    ```python
    class TestPredBufferPxFromDataConfig:
        def test_no_patch_config_returns_zero(self, data_config):
            assert _pred_buffer_px_from_data_config(data_config) == 0

        def test_patch_config_returns_its_buffer_px(self, data_config):
            import dataclasses as dc

            cfg = dc.replace(data_config, patch_config=PatchConfig(n_patches=9, patch_lon_width_px=128, buffer_px=16))
            assert _pred_buffer_px_from_data_config(cfg) == 16


    class TestTargetLatitudes:
        def test_returns_target_da_latitudes_not_input_ds(self, data_config):
            n_time, n_lat_core, n_lon = 2, 3, 4
            input_ds = xr.Dataset(
                {
                    "temperature": xr.DataArray(
                        np.zeros((n_time, n_lat_core + 2, n_lon), dtype=np.float32),
                        dims=["time", "latitude", "longitude"],
                        coords={"time": np.arange(n_time), "latitude": np.arange(n_lat_core + 2) + 100.0},
                    )
                }
            )
            target_da = xr.DataArray(
                np.zeros((n_time, n_lat_core, n_lon), dtype=np.int32),
                dims=["time", "latitude", "longitude"],
                coords={"time": np.arange(n_time), "latitude": np.arange(n_lat_core) + 5.0},
            )
            ds = FrontsPyDataset(input_ds, target_da, data_config, batch_size=1)
            np.testing.assert_array_equal(_target_latitudes(ds), np.arange(n_lat_core) + 5.0)


    class TestShouldBuildTestVisualization:
        def test_true_when_wandb_and_cadence_set_and_no_patch_config(self):
            assert _should_build_test_visualization(None, "fronts", 1) is True

        def test_false_without_wandb_project(self):
            assert _should_build_test_visualization(None, None, 1) is False

        def test_false_without_cadence(self):
            assert _should_build_test_visualization(None, "fronts", None) is False

        def test_false_with_patch_config(self):
            pc = PatchConfig(n_patches=9, patch_lon_width_px=128, buffer_px=0)
            assert _should_build_test_visualization(pc, "fronts", 1) is False
    ```
  - Add `_pred_buffer_px_from_data_config, _target_latitudes, _should_build_test_visualization`
    to the `from fronts.train import (...)` block at `tests/test_train.py:17-23`, and add
    `PatchConfig` to the `from fronts.data.datasets import DatasetConfig, FrontsPyDataset`
    import (already extended in Phase 1).

- [x] **Run it, watch it fail:**
  `pixi run pytest tests/test_train.py -k "PredBufferPxFromDataConfig or TargetLatitudes or ShouldBuildTestVisualization" -v`
  → expect FAIL (`ImportError`)

- [x] **Implement the minimal code.**
  - File: `src/fronts/train.py` — add three small helpers after `_build_loss` (currently
    ending at line 296), before `_load_pretrained_weights`:

    ```python
    def _pred_buffer_px_from_data_config(data_cfg: datasets.DatasetConfig) -> int:
        """Return the loss's prediction context margin: patch_config.buffer_px, or 0 without patch mode."""
        return data_cfg.patch_config.buffer_px if data_cfg.patch_config is not None else 0


    def _target_latitudes(dataset: datasets.FrontsPyDataset) -> np.ndarray:
        """Return the *target* grid's latitudes, for sizing loss/metric pooling windows.

        Must come from target_da, not input_ds: in patch mode input_ds carries the buffered
        (wider) domain, while every loss/metric window must be sized to the unbuffered core
        the model is actually scored against.
        """
        return dataset.target_da["latitude"].values


    def _should_build_test_visualization(
        patch_config: datasets.PatchConfig | None,
        wandb_project: str | None,
        test_viz_every_n_epochs: int | None,
    ) -> bool:
        """Whether train() should build the periodic test-set visualization callback.

        Patch-mode training isn't supported yet: the callback's active-day map and
        per-office-region performance diagrams assume one input/target pair per
        whole-domain timestep, not per-longitude-patch tiling with a buffered input shape.
        """
        return bool(wandb_project) and test_viz_every_n_epochs is not None and patch_config is None
    ```
  - File: `src/fronts/train.py` — `_compile` (currently lines 349-382): add
    `pred_buffer_px: int = 0` parameter, threaded into `_build_loss`:

    ```python
    def _compile(
        model: tf.keras.Model,
        learning_rate: float,
        metric_class_weights: list[float] | None,
        train_cfg: "TrainConfig",
        latitudes: np.ndarray,
        gradient_clip_norm: float | None = None,
        pred_buffer_px: int = 0,
    ) -> int:
        n_out = len(model.outputs)
        loss_fn = _build_loss(
            loss_name=train_cfg.loss_name,
            loss_class_weights=train_cfg.loss_class_weights,
            latitudes=latitudes,
            fss_mask_size=train_cfg.fss_mask_size,
            nbs_tolerance_km=train_cfg.nbs_tolerance_km,
            nbs_periodic_lon=train_cfg.nbs_periodic_lon,
            nbs_lat_dependent_pool=train_cfg.nbs_lat_dependent_pool,
            nbs_include_pixel=train_cfg.nbs_include_pixel,
            nbs_pixel_weight=train_cfg.nbs_pixel_weight,
            nbs_pred_buffer_px=pred_buffer_px,
        )
        ...  # unchanged below
    ```
  - File: `src/fronts/train.py` — `_build_loss` (currently lines 248-296): add
    `nbs_pred_buffer_px: int = 0` parameter and thread it into the
    `neighborhood_brier_score(...)` call:

    ```python
    def _build_loss(
        loss_name: Literal["fractions_skill_score", "neighborhood_brier_score"],
        loss_class_weights: list[float] | None,
        latitudes: np.ndarray,
        fss_mask_size: tuple[int, ...],
        nbs_tolerance_km: float,
        nbs_periodic_lon: bool,
        nbs_lat_dependent_pool: bool,
        nbs_include_pixel: bool = False,
        nbs_pixel_weight: float = 0.1,
        nbs_pred_buffer_px: int = 0,
    ):
        if loss_name == "fractions_skill_score":
            return losses.fractions_skill_score(mask_size=fss_mask_size, class_weights=loss_class_weights)
        if loss_name == "neighborhood_brier_score":
            return losses.neighborhood_brier_score(
                latitudes=latitudes,
                tolerance_km=nbs_tolerance_km,
                class_weights=loss_class_weights,
                periodic_lon=nbs_periodic_lon,
                lat_dependent_pool=nbs_lat_dependent_pool,
                include_pixel=nbs_include_pixel,
                pixel_weight=nbs_pixel_weight,
                pred_buffer_px=nbs_pred_buffer_px,
            )
        raise ValueError(
            f"Unrecognized loss_name {loss_name!r}; expected 'fractions_skill_score' or 'neighborhood_brier_score'."
        )
    ```
  - File: `src/fronts/train.py` — `train()`: change the `_compile(...)` call
    (currently lines 713-720) to pass `latitudes=_target_latitudes(train_dataset)` instead
    of `train_dataset.input_ds["latitude"].values`, and add
    `pred_buffer_px=_pred_buffer_px_from_data_config(data_cfg)`:

    ```python
    _compile(
        unet,
        train_cfg.learning_rate,
        metric_class_weights=data_cfg.class_weights,
        train_cfg=train_cfg,
        latitudes=_target_latitudes(train_dataset),
        gradient_clip_norm=train_cfg.gradient_clip_norm,
        pred_buffer_px=_pred_buffer_px_from_data_config(data_cfg),
    )
    ```
  - File: `src/fronts/train.py` — `train()`: change the two `load_data_into_dataloader`
    calls (currently lines 628-631) to pass `augment` only for the train split:

    ```python
    train_dataset = load_data_into_dataloader(
        data_cfg, split="train", seed=train_cfg.seed, shuffle=train_cfg.shuffle, drop_remainder=True, augment=True
    )
    val_dataset = load_data_into_dataloader(data_cfg, split="val", seed=train_cfg.seed)
    ```
  - File: `src/fronts/train.py` — `train()`: replace the visualization-callback block
    (currently lines 768-777) with the guarded version:

    ```python
    extra_callbacks = []
    if _should_build_test_visualization(data_cfg.patch_config, wandb_project, callbacks_cfg.test_viz_every_n_epochs):
        try:
            extra_callbacks.append(_build_test_visualization_callback(data_cfg, callbacks_cfg, train_cfg.seed))
        except ValueError:
            logger.warning(
                "Skipping periodic test-set visualization: could not build the callback "
                "(see preceding error). Training will continue without it.",
                exc_info=True,
            )
    elif wandb_project and callbacks_cfg.test_viz_every_n_epochs and data_cfg.patch_config is not None:
        logger.info("Skipping periodic test-set visualization: not yet supported for patch-mode training.")
    ```

- [x] **Run it, watch it pass:**
  `pixi run pytest tests/test_train.py -v`
  → expect PASS (all existing + new tests)
- [x] **Commit:** `git commit -m "feat: wire patch buffer through train.py's loss build and viz-callback guard"`

**Dependencies:** Phase 2, Phase 3, Phase 4.

**Verification:**
- [x] `pixi run pytest tests/test_train.py -v` → all PASS

### Phase 6: New ablation config and end-to-end smoke test

**Objective:** `configs/patch_buffer_ablation.yaml` reproduces the paper's parameters on
top of `schooner_train_conus_3d.yaml`'s existing CONUS/3D setup; an end-to-end test proves
a real (small) `UNet3Plus` composes with the buffered-patch shape and the buffer-aware loss.

**Tasks:**
- [x] **Write the failing tests.**
  - File: `tests/test_train.py` (new tests, in `class TestTrainConfigLossClassWeights:`
    after `test_3d_config_parses`, and a new top-level class after
    `class TestLoadPretrainedWeights:`)

    ```python
        def test_patch_buffer_ablation_config_parses(self, train_config_cls):
            from fronts import utils

            yaml_data = utils.load_yaml("configs/patch_buffer_ablation.yaml")
            data_cfg = utils.parse_config_section(
                yaml_data, DatasetConfig, "data_config", type_hooks=utils.YAML_TYPE_HOOKS
            )
            model_cfg = utils.parse_config_section(yaml_data, ModelConfig, "model_config")
            train_cfg = utils.parse_config_section(yaml_data, train_config_cls, "train_config")
            callbacks_cfg = utils.parse_config_section(yaml_data, CallbacksConfig, "callbacks_config")

            assert data_cfg.patch_config == PatchConfig(
                n_patches=9, patch_lon_width_px=128, buffer_px=16, flip_probability=0.25
            )
            assert data_cfg.coordinates == utils.BoundingBox(
                lat_min=25.0, lat_max=56.75, lon_min=228.0, lon_max=299.75
            )
            assert data_cfg.volume_inputs is True
            assert list(model_cfg.pool_size) == [2, 2, 1]
            assert callbacks_cfg.test_viz_every_n_epochs is None
    ```
    (Add `from fronts.callbacks import CallbacksConfig` to the try-import block at
    `tests/test_train.py:11-24` if not already present via another import.)

    ```python
    class TestPatchBufferEndToEnd:
        """Buffered patches must compose through the real model into the buffer-aware loss."""

        def test_buffered_patch_input_scores_against_unbuffered_core_target(self):
            core = 16
            buffer_px = 4
            buffered = core + 2 * buffer_px  # 24; still divisible by the 2-stage (levels=3) stride of 4
            model = _build_small_unet(levels=3, deep_supervision=False)  # input_shape=(None, None, 4)

            rng = np.random.default_rng(3)
            x = rng.standard_normal((2, buffered, buffered, 4)).astype(np.float32)
            y_true = tf.one_hot(rng.integers(0, 6, size=(2, core, core)), 6).numpy().astype(np.float32)

            y_pred = model(x, training=False)
            if isinstance(y_pred, (list, tuple)):
                y_pred = y_pred[0]
            assert y_pred.shape == (2, buffered, buffered, 6)

            loss_fn = losses.neighborhood_brier_score(
                latitudes=np.linspace(25.0, 30.0, core), tolerance_km=25.0, pred_buffer_px=buffer_px
            )
            result = loss_fn(y_true, y_pred).numpy()
            assert result.shape == (2,)
            assert np.all(np.isfinite(result))
    ```
  - Add `from fronts.layers import losses` to the try-import block at
    `tests/test_train.py:11-24`.

- [x] **Run it, watch it fail:**
  `pixi run pytest tests/test_train.py -k "patch_buffer_ablation_config_parses or PatchBufferEndToEnd" -v`
  → expect FAIL (config file doesn't exist yet: `FileNotFoundError`)

- [x] **Implement the minimal code.**
  - File: `configs/patch_buffer_ablation.yaml` (new)

    ```yaml
    run_name: patch_buffer_ablation_3d

    # Reproduces the paper's (Justin et al. 2025) original patch-based training regime: nine
    # 128x128x5x10 (lon x lat x level x variable) patches, evenly spaced along longitude,
    # drawn from the same 128 lat x 288 lon CONUS domain schooner_train_conus_3d.yaml already
    # trains on full-domain -- plus an input-only context buffer around every patch
    # (Ronneberger et al. 2015's overlap-tile strategy: the buffer is fed to the model as
    # input context but cropped from the *pooled* prediction, after the loss's own
    # neighborhood pooling runs, before scoring against the unbuffered target -- see
    # fronts/layers/losses.py's pred_buffer_px) and the paper's per-axis 25% horizontal flip
    # augmentation (0.25 chance each for latitude and longitude, independently, on train
    # patches only -> 43.75% of patches get at least one flip). Ablates the full-domain-
    # single-pass training choice documented in fronts/model_1702/adapter.py against the
    # paper's original patch-tiled regime.

    train_config:
      epochs: 5000
      seed: 42
      learning_rate: 0.0001
      shuffle: true
      loss_class_weights: null  # null supervises all classes incl. background, matching the AIES FrontFinder model
      gradient_clip_norm: 1.0
      loss_name: neighborhood_brier_score
      nbs_tolerance_km: 25.0

    data_config:
      inputs_icechunk_config:
        store_path: "/ourdisk/hpc/ai2es/tman/data/fronts/train/"
        branch_name: "main"
        group_name: "era5"
      targets_icechunk_config:
        store_path: "/ourdisk/hpc/ai2es/tman/restructured_front_data/icechunk"
        branch_name: "main"
        virtual_chunk_local_path: "/ourdisk/hpc/ai2es/tman/restructured_front_data/netcdf/"
      batch_size: 32
      test_years: [2019]
      val_years: [2018]
      max_queue_size: 16
      class_weights: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      front_dilation: 1
      time_resolution: "3h"  # CONUS fronts are available 3-hourly
      coordinates: [25.0, 56.75, 228.0, 299.75]  # [lat_min, lat_max, lon_min, lon_max] -- 128 lat x 288 lon core
      volume_inputs: true  # batches shaped (batch, lat, lon, level, variable) for Conv3D
      patch_config:
        n_patches: 9  # "nine pairs of images evenly spaced along the longitude"
        patch_lon_width_px: 128  # patch core width; latitude uses the full 128px coordinates height (no lat tiling)
        buffer_px: 16  # 128 + 2*16 = 160 = 20 * 8, divisible by the model's total downsampling stride below (8)
        flip_probability: 0.25  # independent per-axis flip probability on train patches only
      variables: [
        "geopotential",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "specific_humidity",
        "vertical_velocity",
        "potential_vorticity",
        "equivalent_potential_temperature",
        "virtual_temperature",
        "dewpoint_temperature",
        "relative_humidity",
        "potential_temperature",
        "mean_sea_level_pressure",
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_dewpoint_temperature",
        "geopotential_at_surface",
        ]
      norm_stats_cache_dir: "/ourdisk/hpc/ai2es/tman/data/fronts/norm_stats_cache"
      normalization_method: minmax
      num_pydataset_workers: 16

    model_config:
      n_classes: 6
      levels: 4
      filter_num: [16, 32, 64, 128]
      pool_size: [2, 2, 1]  # horizontal-only downsampling; 3 pooling stages -> total stride 8
      upsample_size: [2, 2, 1]
      kernel_size: 5
      squeeze_axes: 3  # collapse the level axis at each supervision head via a [1,1,n_levels] valid conv
      first_encoder_connections: true
      deep_supervision: true
      batch_normalization: true
      activation: "gelu"
      output_activation: "softmax"
      modules_per_node: 2

    callbacks_config:
      monitor: "val_loss"
      patience: 3
      learning_rate_decay_factor: 0.2
      learning_rate_minimum: 1e-6
      min_delta: 0.0
      early_stopping_patience: 12
      model_checkpoint_path: "/ourdisk/hpc/ai2es/tman/models/${run_name}/"
      test_viz_every_n_epochs: null  # periodic visualization isn't supported for patch-mode training yet
      test_viz_sample_size: 200

    wandb_config:
      project_name: "fronts"
      run_name: "${run_name}"
      log_freq: 10
    ```

- [x] **Run it, watch it pass:**
  `pixi run pytest tests/test_train.py -k "patch_buffer_ablation_config_parses or PatchBufferEndToEnd" -v`
  → expect PASS
- [x] **Commit:** `git commit -m "feat: add patch_buffer_ablation.yaml and end-to-end patch/buffer test"`

**Dependencies:** Phase 1, Phase 4, Phase 5.

**Verification:**
- [x] `pixi run pytest tests/ -v` → all PASS (full suite)

## Success Criteria

### Automated Verification

- [x] `pixi run pytest tests/test_train.py -v` passes, including every new test class
      listed above.
- [x] `pixi run pytest tests/layers/test_losses.py -v` passes, including
      `TestNeighborhoodBrierScorePredBuffer`.
- [x] `pixi run pytest tests/test_utils.py -v` passes, including `TestExpandBoundingBox`.
- [x] `pixi run pytest tests/ -v` passes in full (no regressions elsewhere).
- [x] `configs/patch_buffer_ablation.yaml` exists and
      `utils.parse_config_section(utils.load_yaml("configs/patch_buffer_ablation.yaml"), DatasetConfig, "data_config", type_hooks=utils.YAML_TYPE_HOOKS)`
      succeeds with `patch_config == PatchConfig(n_patches=9, patch_lon_width_px=128, buffer_px=16, flip_probability=0.25)`.
- [x] Pre-commit hooks (ruff, etc.) pass on every changed file:
      `pixi run pre-commit run --files src/fronts/data/datasets.py src/fronts/utils.py src/fronts/layers/losses.py src/fronts/train.py configs/patch_buffer_ablation.yaml tests/test_train.py tests/test_utils.py tests/layers/test_losses.py`.

### Manual Verification

- [ ] Launch a short real training run (a handful of steps, e.g.
      `train_config.epochs` overridden low and pointed at a small local icechunk store) with
      `configs/patch_buffer_ablation.yaml` and confirm: it logs 9x the timestep count as
      the number of training samples, `unet.summary()` shows the buffered input shape, and
      training proceeds without a shape error.
- [ ] Watch W&B (or stdout logs) for the "Skipping periodic test-set visualization" info
      log at startup, confirming the patch-mode guard fires as intended.
- [ ] Spot-check a handful of flipped training patches visually (e.g. via
      `fronts.plot.plot`) against their un-augmented counterparts to confirm the flip
      augmentation looks geometrically correct (no transpose/axis-swap bugs).

## Testing Strategy

Unit tests are written test-first within each phase (see Implementation Phases above).

**Unit Test Coverage (summary, written in-phase):**
- [x] `PatchConfig` validation and `compute_patch_lon_starts` geometry (Phase 1).
- [x] `expand_bounding_box` (Phase 1).
- [x] `FrontsPyDataset` patch-mode sample count, buffered-input slicing, unbuffered-target
      slicing, and flip augmentation determinism at `flip_probability` in {0.0, 1.0}
      (Phase 2).
- [x] `load_data_into_dataloader` buffered-vs-core domain loading, store-edge overrun
      error, missing-`coordinates` error, and `augment` threading (Phase 3).
- [x] `neighborhood_brier_score(pred_buffer_px=...)` shape handling, buffer-vs-zero-pad
      pooling behavior, and `include_pixel` interaction (Phase 4).
- [x] `_pred_buffer_px_from_data_config`, `_target_latitudes`,
      `_should_build_test_visualization` pure-function unit tests (Phase 5).
- [x] New ablation config parsing and full model+loss end-to-end shape/finiteness
      (Phase 6).

**Integration Tests:**
- [x] `TestPatchBufferEndToEnd` (Phase 6) is the integration point proving the model,
      buffered patch input, and buffer-aware loss compose correctly together — the one test
      that would catch a shape mismatch none of the narrower unit tests could see on their
      own.

**Manual Testing:** see Manual Verification above.

**Test Data Requirements:** All new automated tests use small synthetic in-memory
`xarray` fixtures or `tmp_path`-backed icechunk stores via the existing
`write_or_append_icechunk_store` test helper — no real ERA5/fronts data is required to run
the test suite. The Manual Verification steps require access to a real (or realistically
small local) icechunk store, same as any other training config in this repo.

### Reproducibility & Correctness (research code)

- [x] `configs/patch_buffer_ablation.yaml` pins `train_config.seed`, `data_config`'s store
      paths/branches, and every hyperparameter needed to rerun the ablation, matching the
      existing convention in `configs/schooner_train_conus_3d.yaml` and
      `configs/sooner_ablations.yaml`.
  Full reproducibility capture (icechunk snapshot IDs, git commit, environment) is already
  handled generically by `train.py`'s existing `_collect_run_metadata`
  (`src/fronts/train.py:573-605`) and W&B run config for every training run, patch-mode
  included — no new work needed here.
- [x] Numerical correctness criterion: `TestNeighborhoodBrierScorePredBuffer::test_pred_buffer_px_pooling_reads_real_buffer_values_not_zero_padding`
      is the analytic-case check for the loss's crop-after-pooling contract — an exactly
      zero unbuffered baseline vs. a provably nonzero buffered result, both hand-derived
      above rather than asserted against an opaque reference value.
- [x] The result reproduces in a clean environment: all new tests are self-contained
      (synthetic fixtures, `tmp_path` icechunk stores), so `pixi run pytest tests/` passing
      in CI is sufficient evidence of a clean-environment reproduction of the mechanism;
      the ablation's actual trained-model result still depends on the real, shared icechunk
      stores referenced in the config, same as every other training run in this repo.

## References

**Files Analyzed:**
- `src/fronts/data/datasets.py`
- `src/fronts/data/inputs.py`
- `src/fronts/data/targets.py`
- `src/fronts/train.py`
- `src/fronts/model.py`
- `src/fronts/layers/losses.py`
- `src/fronts/layers/modules.py`
- `src/fronts/callbacks.py`
- `src/fronts/utils.py`
- `src/fronts/model_1702/adapter.py`
- `configs/schooner_train_conus_3d.yaml`
- `configs/sooner_ablations.yaml`
- `tests/test_train.py`
- `tests/test_utils.py`
- `tests/layers/test_losses.py`
- `tests/conftest.py`

**External Documentation:**
- Ronneberger, Fischer & Brox (2015), *U-Net: Convolutional Networks for Biomedical Image
  Segmentation* — https://arxiv.org/abs/1505.04597 (overlap-tile strategy: source for the
  crop-after-pooling design decision)
- Roberts & Lean (2008) — https://doi.org/10.1175/2007MWR2123.1 (fractions skill score,
  already cited in `losses.py`)
- Stein & Stoop (2024) — https://doi.org/10.1175/MWR-D-22-0235.1 (already cited in
  `losses.py`)

**Codebase citation convention:** this repo already refers to the source paper internally
as "Justin et al. (2025)" (see `fronts/data/targets.py:10`); this plan follows that
existing convention rather than introducing a new one.

---

## Review History

### Version 1.0 — 2026-08-31
- Initial plan created from a research conversation covering patch-based training
  reproduction and buffer/crop-after-pooling design (Ronneberger et al. 2015 citation).
