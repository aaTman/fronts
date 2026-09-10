"""Regression test for the patch-vs-baseline per-epoch model-voxel budget.

The whole point of `experiment/patch-buffer-ablation` is that patch-mode training should
be **at least as fast per epoch** as `configs/schooner_train_3d.yaml`'s whole-domain
baseline. Today it is not: `configs/patch_buffer_ablation.yaml` visits every one of its 30
patches per timestep per epoch (no `patches_per_epoch` subsampling yet), padded with a
single, symmetric `buffer_px: 16` applied to *both* latitude and longitude. That combines
into a **5.50x** per-epoch voxel count relative to the baseline (see the table below),
which is exactly why patch-mode training currently runs roughly 5x slower per epoch.

| factor                    | today's value | where it comes from                          |
|----------------------------|---------------|-----------------------------------------------|
| core-coverage redundancy   | 4.00x         | 30 patches x 128px core over a 960px domain    |
| longitude buffer overhead  | 1.25x         | 128 -> 160px (16px each side)                 |
| latitude buffer overhead   | 1.10x         | 320 -> 352px (16px each side)                 |
| **combined**                | **5.50x**     | product of the three factors above             |

The intended fix (tracked in a shared contract the other agents are implementing against)
is two-part: (1) a per-axis buffer, since every patch already spans the full latitude
height of the domain -- there is no artificial tile cut along latitude, so
`buffer_lat_px` should default to 0 and only `buffer_lon_px` (which guards a real
longitude tile cut, Ronneberger et al. 2015's overlap-tile context) stays nonzero; and
(2) a `patches_per_epoch` knob that visits only k of the n_patches positions per timestep
per epoch, redrawn each epoch, so an epoch covers roughly one pass over the domain instead
of `n_patches` passes. With the target `n_patches=30, patches_per_epoch=6,
buffer_lon_px=16, buffer_lat_px=0`, the three factors become 0.80x * 1.25x * 1.00x =
**1.00x** -- parity with the baseline.

This module computes that arithmetic straight from the YAML configs (never from
hardcoded copies of their numbers) so the budget can never silently regress again.
`test_patch_epoch_voxel_budget_at_or_below_baseline` is the load-bearing assertion; the
rest exist to make a future failure self-explaining rather than a bare number mismatch.

Deliberately excluded from the cost model: variable count (18 for the baseline, 8 for the
patch config). In this U-Net only the first convolution's input channels scale with
variable count -- every other layer's cost is set by spatial/level resolution, not
channel count -- so variable count is a second-order effect on the true per-epoch voxel
cost. Excluding it also makes this test strictly conservative: it never lets the patch
config "win" the comparison merely by training on fewer variables.

This module intentionally imports nothing from `fronts` (not even `fronts.utils`, whose
`load_yaml` pulls in `icechunk`/`xarray`/`zarr`): it reads the two YAML files directly, so
it runs fast, needs no TensorFlow, and stays independent of the in-flight API changes the
other agents are making to `datasets.py`, `train.py`, etc.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_CONFIG_PATH = REPO_ROOT / "configs" / "schooner_train_3d.yaml"
PATCH_CONFIG_PATH = REPO_ROOT / "configs" / "patch_buffer_ablation.yaml"

# The icechunk store's native grid resolution; both configs' `coordinates` bounding boxes
# are expressed in degrees at this spacing.
GRID_RESOLUTION_DEG = 0.25

# This job's 4-GPU tf.distribute.MirroredStrategy split (see fronts.train._validate_batch_
# size_for_strategy, which hard-fails model.fit with the same check).
NUM_GPUS = 4

# CONTRACT.md's target patch_config, which multiplies out to the 1.00x parity ratio.
TARGET_PATCHES_PER_EPOCH = 6
TARGET_BUFFER_LON_PX = 16
TARGET_BUFFER_LAT_PX = 0


def _load_config(path: Path) -> dict[str, Any]:
    """Loads a YAML training config as a plain dict.

    Deliberately does not use `fronts.utils.load_yaml`: that helper also interpolates
    `${run_name}`-style placeholders, which none of the fields this test reads need, and
    pulls in `fronts.utils`'s heavier imports (icechunk, xarray, zarr). Plain
    `yaml.safe_load` is sufficient and keeps this test's import surface minimal.

    Args:
        path: Path to the YAML config file.

    Returns:
        The parsed YAML document as a dict.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def _n_px(lo: float, hi: float) -> int:
    """Number of 0.25deg grid points spanning a closed interval [lo, hi].

    Args:
        lo: Lower bound in degrees (inclusive).
        hi: Upper bound in degrees (inclusive).

    Returns:
        The number of grid points, e.g. `_n_px(0.25, 80)` -> 320.
    """
    return round((hi - lo) / GRID_RESOLUTION_DEG) + 1


def _required(mapping: dict[str, Any], key: str, source: str) -> Any:
    """Fetches `mapping[key]`, failing the test cleanly (not a raw `KeyError`) if absent.

    Args:
        mapping: The dict to read from (a YAML config section).
        key: The required key.
        source: A human-readable label for the config section, used in the failure
            message (e.g. "configs/patch_buffer_ablation.yaml: data_config.patch_config").

    Returns:
        `mapping[key]`.
    """
    if key not in mapping or mapping[key] is None:
        pytest.fail(f"{source}.{key} is required but missing from the config.")
    return mapping[key]


@pytest.fixture(scope="module")
def baseline_data() -> dict[str, Any]:
    """`data_config` section of the whole-domain baseline config."""
    return _load_config(BASELINE_CONFIG_PATH)["data_config"]


@pytest.fixture(scope="module")
def patch_full() -> dict[str, Any]:
    """Full parsed contents of the patch-ablation config (all top-level sections)."""
    return _load_config(PATCH_CONFIG_PATH)


@pytest.fixture(scope="module")
def patch_data(patch_full: dict[str, Any]) -> dict[str, Any]:
    """`data_config` section of the patch-ablation config."""
    return patch_full["data_config"]


@pytest.fixture(scope="module")
def patch_model(patch_full: dict[str, Any]) -> dict[str, Any]:
    """`model_config` section of the patch-ablation config."""
    return patch_full["model_config"]


@pytest.fixture(scope="module")
def patch_patch(patch_data: dict[str, Any]) -> dict[str, Any]:
    """`data_config.patch_config` section of the patch-ablation config."""
    return _required(patch_data, "patch_config", "configs/patch_buffer_ablation.yaml: data_config")


@pytest.fixture(scope="module")
def domain(patch_data: dict[str, Any]) -> tuple[int, int]:
    """(n_lat, n_lon) of the full training domain, in grid pixels at 0.25deg.

    Derived from `patch_buffer_ablation.yaml`'s `data_config.coordinates`
    ([lat_min, lat_max, lon_min, lon_max]) rather than hardcoded, so a future change to
    the domain is picked up automatically.
    """
    coords = _required(patch_data, "coordinates", "configs/patch_buffer_ablation.yaml: data_config")
    if len(coords) != 4:
        pytest.fail(
            "configs/patch_buffer_ablation.yaml: data_config.coordinates must be "
            f"[lat_min, lat_max, lon_min, lon_max]; got {coords!r}."
        )
    lat_min, lat_max, lon_min, lon_max = coords
    return _n_px(lat_min, lat_max), _n_px(lon_min, lon_max)


@pytest.fixture(scope="module")
def n_levels(baseline_data: dict[str, Any], patch_data: dict[str, Any]) -> tuple[int, int]:
    """(baseline_n_levels, patch_n_levels): vertical-level count each config trains on.

    `n_levels` is `len(pressure_levels)` when `volume_inputs` is true (the model carries
    the level axis through the network via Conv3D/MaxPooling3D), else 1 (levels are
    flattened into the channel axis for a 2D model). This factor multiplies both sides of
    the cost model identically, so it cancels out of the headline *ratio* test entirely --
    it only affects the absolute voxel counts a failure message prints.
    """

    def _levels_or_none(data_cfg: dict[str, Any]) -> list[int] | None:
        if not data_cfg.get("volume_inputs", False):
            return [None]  # sentinel list of length 1 -> n_levels == 1
        return data_cfg.get("pressure_levels")

    patch_levels = _levels_or_none(patch_data)
    if patch_levels is None:
        pytest.fail(
            "configs/patch_buffer_ablation.yaml: data_config.volume_inputs is true but "
            "pressure_levels is not set; the cost model needs an explicit level count."
        )
    n_patch_levels = len(patch_levels)

    baseline_levels = _levels_or_none(baseline_data)
    if baseline_levels is not None:
        n_baseline_levels = len(baseline_levels)
    else:
        # schooner_train_3d.yaml's own header comment documents that omitting
        # pressure_levels selects the icechunk store's native levels, and that "the
        # store's 6 native levels (1000-300 hPa) are used" -- exactly the 6 levels
        # patch_buffer_ablation.yaml lists explicitly. Falling back to that count (rather
        # than hardcoding 6) keeps the comparison traceable to the configs themselves.
        n_baseline_levels = n_patch_levels
    return n_baseline_levels, n_patch_levels


@pytest.fixture(scope="module")
def patch_geometry(patch_patch: dict[str, Any]) -> SimpleNamespace:
    """Patch geometry knobs, with documented fallbacks for keys not yet renamed/added.

    `patches_per_epoch` absent means "visit all n_patches" (the dataclass default
    documents this: `patches_per_epoch: int | None = None` means None -> all n_patches) --
    i.e. today's actual, wasteful behavior, not a missing-key error.

    `buffer_lat_px`/`buffer_lon_px` absent falls back to the old, pre-rename symmetric
    `buffer_px` key (applied to both axes) if present, else 0. This means the cost
    computed here reflects the *actual* padding cost of the config as it stands today
    (single 16px buffer on both axes) rather than silently reporting 0 padding, which
    would understate how bad the current per-epoch voxel budget really is. Once
    `buffer_px` is renamed/split per the contract, this fallback becomes a no-op and the
    new keys are read directly.
    """
    n_patches = _required(patch_patch, "n_patches", "configs/patch_buffer_ablation.yaml: patch_config")
    patch_lon_width_px = _required(
        patch_patch, "patch_lon_width_px", "configs/patch_buffer_ablation.yaml: patch_config"
    )

    patches_per_epoch = patch_patch.get("patches_per_epoch")
    if patches_per_epoch is None:
        patches_per_epoch = n_patches

    def _buffer_px(axis: str) -> int:
        new_key = f"buffer_{axis}_px"
        if patch_patch.get(new_key) is not None:
            return patch_patch[new_key]
        if patch_patch.get("buffer_px") is not None:
            return patch_patch["buffer_px"]
        return 0

    return SimpleNamespace(
        n_patches=n_patches,
        patch_lon_width_px=patch_lon_width_px,
        patches_per_epoch=patches_per_epoch,
        buffer_lat_px=_buffer_px("lat"),
        buffer_lon_px=_buffer_px("lon"),
    )


@pytest.fixture(scope="module")
def costs(domain: tuple[int, int], n_levels: tuple[int, int], patch_geometry: SimpleNamespace) -> SimpleNamespace:
    """Per-training-timestep, per-epoch model-input-voxel costs for both configs.

    whole domain (baseline) = n_lat * n_lon * n_levels
    patch mode              = patches_per_epoch * (n_lat + 2*buffer_lat_px)
                                                 * (patch_lon_width_px + 2*buffer_lon_px)
                                                 * n_levels
    """
    n_lat, n_lon = domain
    n_baseline_levels, n_patch_levels = n_levels
    g = patch_geometry

    baseline_voxels = n_lat * n_lon * n_baseline_levels

    buffered_lat = n_lat + 2 * g.buffer_lat_px
    buffered_lon = g.patch_lon_width_px + 2 * g.buffer_lon_px
    patch_voxels = g.patches_per_epoch * buffered_lat * buffered_lon * n_patch_levels

    return SimpleNamespace(
        baseline_voxels=baseline_voxels,
        patch_voxels=patch_voxels,
        ratio=patch_voxels / baseline_voxels,
        core_redundancy=g.patches_per_epoch * g.patch_lon_width_px / n_lon,
        lon_buffer_factor=buffered_lon / g.patch_lon_width_px,
        lat_buffer_factor=buffered_lat / n_lat,
        buffered_lat=buffered_lat,
        buffered_lon=buffered_lon,
    )


class TestPatchEpochVoxelBudget:
    """Encodes "patch-mode must be >= as fast per epoch as the whole-domain baseline".

    See the module docstring for the full cost-model derivation and today's 5.50x number.
    """

    def test_baseline_full_domain_matches_patch_config_coordinates(
        self, baseline_data: dict[str, Any], domain: tuple[int, int]
    ) -> None:
        """The baseline's implicit full domain must equal the patch config's domain.

        `schooner_train_3d.yaml` has no `data_config.coordinates` at all, which means
        "train on the full store domain" (see `fronts.train`: `coordinates` is only
        applied when set). `patch_buffer_ablation.yaml` sets `coordinates` explicitly to
        what its own comment documents as that same full domain. The rest of this test
        module borrows the patch config's coordinates as a stand-in for "the baseline's
        domain" -- this test makes that substitution an explicit, checked claim instead
        of a silent assumption.
        """
        baseline_coords = baseline_data.get("coordinates")
        assert baseline_coords is None, (
            "configs/schooner_train_3d.yaml now sets data_config.coordinates explicitly "
            f"({baseline_coords!r}) instead of training on the full store domain. This "
            "test module assumed the baseline's coordinates were absent and borrowed "
            "patch_buffer_ablation.yaml's coordinates as a stand-in for 'full domain' -- "
            "that assumption is now wrong. Update the domain fixture to read the "
            "baseline's own coordinates directly instead of reusing the patch config's."
        )
        n_lat, n_lon = domain
        assert (n_lat, n_lon) == (320, 960), (
            f"Expected the full domain (from patch_buffer_ablation.yaml's coordinates) to "
            f"be 320 lat x 960 lon at {GRID_RESOLUTION_DEG}deg resolution; got "
            f"{n_lat} x {n_lon}. If the icechunk store's domain has changed, the "
            "baseline's implicit full-domain size has changed with it, and every voxel "
            "count in this module needs re-deriving from the new domain."
        )

    def test_patch_epoch_voxel_budget_at_or_below_baseline(self, costs: SimpleNamespace) -> None:
        """THE headline test: patch-mode must cost <= the baseline per epoch (5% slack).

        This is the executable form of the success criterion: "equivalent or greater
        speed with the patching approach vs the feat/2.0.0 schooner_train_3d.yaml
        approach." Per-epoch wall-clock time is dominated by the number of model input
        voxels processed, so this compares voxel counts per training timestep per epoch.
        """
        tolerance = 1.05
        assert costs.patch_voxels <= costs.baseline_voxels * tolerance, (
            "\nPatch-mode epoch is MORE EXPENSIVE than the whole-domain baseline:\n"
            f"  patch-mode voxels/timestep : {costs.patch_voxels:,}\n"
            f"  baseline voxels/timestep   : {costs.baseline_voxels:,}\n"
            f"  ratio (patch / baseline)   : {costs.ratio:.2f}x  (must be <= {tolerance:.2f}x)\n"
            "\n"
            "Contributing factors (these three multiply together to the ratio above):\n"
            "  core-coverage redundancy   = patches_per_epoch * patch_lon_width_px / n_lon "
            f"= {costs.core_redundancy:.2f}x\n"
            "  longitude buffer overhead  = (patch_lon_width_px + 2*buffer_lon_px) / patch_lon_width_px "
            f"= {costs.lon_buffer_factor:.2f}x\n"
            "  latitude buffer overhead   = (n_lat + 2*buffer_lat_px) / n_lat "
            f"= {costs.lat_buffer_factor:.2f}x\n"
            "\n"
            "Whichever factor above is inflated is the config knob to fix. CONTRACT.md's "
            "target (patches_per_epoch=6, buffer_lon_px=16, buffer_lat_px=0) multiplies "
            "out to 0.80x * 1.25x * 1.00x = 1.00x."
        )

    def test_batch_size_divides_by_patches_per_epoch(
        self, patch_data: dict[str, Any], patch_geometry: SimpleNamespace
    ) -> None:
        """`batch_size % patches_per_epoch == 0` -> every batch reads whole timesteps.

        When this holds, every batch drawn from FrontsPyDataset's `_order` covers exactly
        `batch_size // patches_per_epoch` distinct timesteps and, across a whole epoch,
        each timestep is materialized exactly once (CONTRACT.md's "Read-amplification
        invariant"). When it doesn't, some batches straddle a partial timestep's worth of
        patches, forcing that timestep to be read again in the next batch.
        """
        batch_size = _required(patch_data, "batch_size", "configs/patch_buffer_ablation.yaml: data_config")
        k = patch_geometry.patches_per_epoch
        assert batch_size % k == 0, (
            f"batch_size ({batch_size}) is not a multiple of patches_per_epoch ({k}); "
            f"{batch_size} / {k} does not divide evenly, so at least one batch per epoch "
            "straddles a partial timestep and forces a cross-batch re-read of that "
            "timestep's inputs. Set data_config.batch_size and "
            "data_config.patch_config.patches_per_epoch to values where one divides the "
            "other exactly (target: batch_size=24, patches_per_epoch=6 -> 24/6=4 whole "
            "timesteps per batch, zero re-reads)."
        )

    def test_batch_size_divisible_by_num_gpus(self, patch_data: dict[str, Any]) -> None:
        """`batch_size % 4 == 0` -> an even per-replica split across the 4-GPU strategy.

        `fronts.train._validate_batch_size_for_strategy` hard-fails `model.fit` with
        exactly this check: an uneven split crashes deep-supervision gradient aggregation
        (an AddN shape mismatch) or the cuDNN backward pass on the odd-sized replica --
        both well after a full model build and data load, so it is worth catching here.
        """
        batch_size = _required(patch_data, "batch_size", "configs/patch_buffer_ablation.yaml: data_config")
        assert batch_size % NUM_GPUS == 0, (
            f"batch_size ({batch_size}) is not a multiple of NUM_GPUS ({NUM_GPUS}); "
            "fronts.train._validate_batch_size_for_strategy will hard-fail model.fit "
            "with this same check once training actually starts, after a full model "
            "build and data load. Pick a batch_size that is a multiple of 4."
        )

    def test_buffered_patch_dims_divisible_by_model_stride(
        self, patch_geometry: SimpleNamespace, domain: tuple[int, int], patch_model: dict[str, Any]
    ) -> None:
        """The buffered patch height and width must each divide the model's stride.

        `model_config.pool_size` (e.g. `[2, 2, 1]`) is applied uniformly at each of
        `model_config.levels - 1` pooling stages (see `fronts.model`: a single pool_size
        tuple, not one per level), so the total horizontal stride is
        `pool_size[0] ** (levels - 1)` for latitude and `pool_size[1] ** (levels - 1)` for
        longitude. A MaxPooling/UpSampling round trip needs each spatial dim to be an
        exact multiple of that stride, or shapes mismatch when skip connections are
        concatenated back in.
        """
        pool_size = _required(patch_model, "pool_size", "configs/patch_buffer_ablation.yaml: model_config")
        levels = _required(patch_model, "levels", "configs/patch_buffer_ablation.yaml: model_config")
        n_pool_stages = levels - 1
        stride_lat = pool_size[0] ** n_pool_stages
        stride_lon = pool_size[1] ** n_pool_stages

        n_lat, _ = domain
        buffered_lat = n_lat + 2 * patch_geometry.buffer_lat_px
        buffered_lon = patch_geometry.patch_lon_width_px + 2 * patch_geometry.buffer_lon_px

        assert buffered_lat % stride_lat == 0, (
            f"buffered patch height (n_lat + 2*buffer_lat_px = {n_lat} + "
            f"2*{patch_geometry.buffer_lat_px} = {buffered_lat}) is not divisible by the "
            f"model's latitude stride ({stride_lat} = pool_size[0]**{n_pool_stages} = "
            f"{pool_size[0]}**{n_pool_stages}). Choose buffer_lat_px so the buffered "
            f"height is a multiple of {stride_lat}."
        )
        assert buffered_lon % stride_lon == 0, (
            f"buffered patch width (patch_lon_width_px + 2*buffer_lon_px = "
            f"{patch_geometry.patch_lon_width_px} + 2*{patch_geometry.buffer_lon_px} = "
            f"{buffered_lon}) is not divisible by the model's longitude stride "
            f"({stride_lon} = pool_size[1]**{n_pool_stages} = {pool_size[1]}**{n_pool_stages}). "
            f"Choose buffer_lon_px so the buffered width is a multiple of {stride_lon}."
        )

    def test_core_coverage_redundancy_matches_target(
        self, costs: SimpleNamespace, patch_geometry: SimpleNamespace, domain: tuple[int, int]
    ) -> None:
        """Characterization test: core-coverage redundancy vs. CONTRACT.md's target.

        Target: `patches_per_epoch=6` over a 960px domain with a 128px core ->
        `6 * 128 / 960 = 0.80x`, i.e. roughly one pass over the domain per epoch,
        directly comparable to the baseline's one-pass epoch.
        """
        _, n_lon = domain
        target = TARGET_PATCHES_PER_EPOCH * patch_geometry.patch_lon_width_px / n_lon
        assert costs.core_redundancy == pytest.approx(target), (
            f"core-coverage redundancy is {costs.core_redundancy:.2f}x "
            f"(patches_per_epoch={patch_geometry.patches_per_epoch} * "
            f"patch_lon_width_px={patch_geometry.patch_lon_width_px} / n_lon={n_lon}); "
            f"target is {target:.2f}x with patches_per_epoch={TARGET_PATCHES_PER_EPOCH} "
            "(CONTRACT.md: 'k=6 an epoch covers 6*128/960 = 0.8 of the domain, i.e. "
            "roughly one pass'). Set data_config.patch_config.patches_per_epoch: "
            f"{TARGET_PATCHES_PER_EPOCH} in configs/patch_buffer_ablation.yaml."
        )

    def test_longitude_buffer_overhead_matches_target(self, costs: SimpleNamespace) -> None:
        """Characterization test: longitude buffer overhead vs. CONTRACT.md's target.

        Target: 128px core -> 160px buffered (16px context each side) = 1.25x. This
        factor is *intended* to stay nonzero: longitude is where patches actually tile
        the domain, so Ronneberger et al. 2015's overlap-tile context buffer is doing
        real work here.
        """
        assert costs.lon_buffer_factor == pytest.approx(1.25), (
            f"longitude buffer overhead is {costs.lon_buffer_factor:.2f}x; target is "
            "1.25x (128 -> 160px, 16px of context each side, per CONTRACT.md). Set "
            f"data_config.patch_config.buffer_lon_px: {TARGET_BUFFER_LON_PX}."
        )

    def test_latitude_buffer_overhead_matches_target(self, costs: SimpleNamespace) -> None:
        """Characterization test: latitude buffer overhead vs. CONTRACT.md's target.

        Target: 1.00x, i.e. no latitude buffer at all. Every patch already spans the
        full latitude height of the domain -- there is no artificial tile cut along
        latitude for an overlap-tile buffer to compensate for. Padding it anyway (as
        today's config does) reflects fabricated data off the domain's own north/south
        edges, which is pure waste: the baseline model sees those same edges unpadded.
        """
        assert costs.lat_buffer_factor == pytest.approx(1.0), (
            f"latitude buffer overhead is {costs.lat_buffer_factor:.2f}x; target is "
            "1.00x (buffer_lat_px=0 -- CONTRACT.md: 'every patch already spans the full "
            "latitude height of the domain, so there is no artificial tile cut along "
            "latitude'). Set data_config.patch_config.buffer_lat_px: "
            f"{TARGET_BUFFER_LAT_PX}."
        )
