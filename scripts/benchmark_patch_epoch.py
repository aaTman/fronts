r"""Estimate patch-mode training's per-epoch cost relative to whole-domain training.

Patch-based training (``patch_config`` set in ``data_config``) is roughly 5x slower per
epoch than whole-domain training on the same store, because it recomputes overlapping
longitude tiles and pads every tile with input-only context on both axes. The root cause
decomposes into three multiplicative factors (see
``docs/rse/specs/plan-patch-buffer-training.md``):

    core-coverage redundancy = patches_per_epoch * patch_lon_width_px / n_lon
    longitude buffer overhead = (patch_lon_width_px + 2 * buffer_lon_px) / patch_lon_width_px
    latitude buffer overhead  = (n_lat + 2 * buffer_lat_px) / n_lat

This script checks that decomposition, and the fix for it, WITHOUT touching a GPU or an
icechunk store:

    Mode 1 (default): parses two training configs and reports the analytic voxel-count
    comparison above. Instant, no TensorFlow import, safe to run anywhere.

    Mode 2 (``--measure``): builds the real ``fronts.model.UNet3Plus`` for each config and
    times actual forward+backward steps on synthetic data, as a rough relative-cost proxy
    (this machine has no GPU, so absolute numbers do not transfer to the cluster).

Usage:
    python scripts/benchmark_patch_epoch.py \\
        --config configs/patch_buffer_ablation.yaml \\
        --baseline-config configs/schooner_train_3d.yaml

    python scripts/benchmark_patch_epoch.py \\
        --config configs/patch_buffer_ablation.yaml \\
        --baseline-config configs/schooner_train_3d.yaml \\
        --measure --scale-factor 8 --batch-size-override 2
"""

import argparse
import logging
import math
import statistics
import time
from typing import Any

import numpy as np

from fronts import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_handler)

# The store this repo trains against is documented (see schooner_train_3d.yaml's header
# comment) to hold exactly 6 native pressure levels (1000-300 hPa). A config with
# volume_inputs=true and no explicit pressure_levels list (e.g. the whole-domain 3D
# baseline) loads every native level, so this is the only way to know its level count
# without opening the icechunk store -- which Mode 1 deliberately never does.
NATIVE_STORE_LEVELS = 6

# Full spatial extent of the ERA5/fronts icechunk stores at 0.25 deg resolution (see
# diagnose_read_throughput.py and datasets.PatchConfig's docstring). A config with no
# data_config.coordinates trains on this whole domain rather than a cropped one.
FULL_DOMAIN_N_LAT = 320
FULL_DOMAIN_N_LON = 960
GRID_RESOLUTION_DEG = 0.25

# Assumed number of training timesteps per epoch, used only to turn the (config-
# independent) per-timestep voxel ratio into an illustrative absolute batches/epoch and
# epoch-time figure. Both configs here hold out the same years (test=2019, val=2018) at
# the same 6-hourly resolution, so this cancels out of every *ratio* reported below
# regardless of its exact value -- it only affects the standalone "batches per epoch" and
# "estimated epoch time" lines.
ASSUMED_TRAIN_TIMESTEPS = 4000


def _n_lat_lon(data_cfg: dict[str, Any]) -> tuple[int, int, bool]:
    """Derives the training domain's grid size in pixels from a data_config dict.

    Args:
        data_cfg: The raw ``data_config`` YAML section.

    Returns:
        A (n_lat, n_lon, used_fallback) tuple. ``used_fallback`` is True when
        ``coordinates`` was absent and the full store domain was assumed instead.
    """
    coordinates = data_cfg.get("coordinates")
    if coordinates is None:
        return FULL_DOMAIN_N_LAT, FULL_DOMAIN_N_LON, True
    lat_min, lat_max, lon_min, lon_max = coordinates
    n_lat = round((lat_max - lat_min) / GRID_RESOLUTION_DEG) + 1
    n_lon = round((lon_max - lon_min) / GRID_RESOLUTION_DEG) + 1
    return n_lat, n_lon, False


def _n_levels(data_cfg: dict[str, Any]) -> tuple[int, bool]:
    """Derives the per-sample vertical level count from a data_config dict.

    Args:
        data_cfg: The raw ``data_config`` YAML section.

    Returns:
        A (n_levels, used_fallback) tuple. ``used_fallback`` is True when
        ``volume_inputs`` is set but ``pressure_levels`` was absent, so the store's
        native level count was assumed instead.
    """
    if not data_cfg.get("volume_inputs", False):
        return 1, False
    pressure_levels = data_cfg.get("pressure_levels")
    if pressure_levels is None:
        return NATIVE_STORE_LEVELS, True
    return len(pressure_levels), False


def _parse_patch_config(patch_cfg: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalizes a raw ``patch_config`` dict across the old and new key spellings.

    The old spelling (pre-migration) has a single isotropic ``buffer_px`` and no
    ``patches_per_epoch`` (every one of ``n_patches`` positions is visited every epoch).
    The new spelling splits the buffer per-axis (``buffer_lon_px``/``buffer_lat_px``,
    latitude defaulting to 0) and adds ``patches_per_epoch`` to subsample positions per
    epoch. This repo migrates key spellings while this script's contract stays fixed, so
    both must parse.

    Args:
        patch_cfg: The raw ``data_config.patch_config`` YAML section, or None.

    Returns:
        None if ``patch_cfg`` is None, else a dict with normalized keys ``n_patches``,
        ``patch_lon_width_px``, ``buffer_lon_px``, ``buffer_lat_px``, ``patches_per_epoch``,
        and ``spelling`` (``"old"`` or ``"new"``, for display).
    """
    if patch_cfg is None:
        return None
    n_patches = int(patch_cfg["n_patches"])
    patch_lon_width_px = int(patch_cfg["patch_lon_width_px"])
    if "buffer_lon_px" in patch_cfg or "buffer_lat_px" in patch_cfg:
        spelling = "new (buffer_lon_px/buffer_lat_px)"
        buffer_lon_px = int(patch_cfg.get("buffer_lon_px", 0))
        buffer_lat_px = int(patch_cfg.get("buffer_lat_px", 0))
    else:
        spelling = "old (buffer_px, isotropic)"
        buffer_px = int(patch_cfg.get("buffer_px", 0))
        buffer_lon_px = buffer_px
        buffer_lat_px = buffer_px
    patches_per_epoch = int(patch_cfg.get("patches_per_epoch") or n_patches)
    return {
        "n_patches": n_patches,
        "patch_lon_width_px": patch_lon_width_px,
        "buffer_lon_px": buffer_lon_px,
        "buffer_lat_px": buffer_lat_px,
        "patches_per_epoch": patches_per_epoch,
        "spelling": spelling,
    }


def _simulate_read_amplification(
    n_time: int, n_patches: int, patches_per_epoch: int, batch_size: int, seed: int = 0
) -> float:
    """Simulates ``FrontsPyDataset._build_order``'s block-shuffled order to measure re-reads.

    Reproduces the block-alignment scheme exactly: blocks of
    ``batch_size // gcd(batch_size, patches_per_epoch)`` contiguous timesteps are kept
    together and only block visitation order is shuffled, with a ragged trailing block
    placed last. This mirrors the datasets.py implementation but is reimplemented locally
    (rather than imported) so Mode 1 never imports ``fronts.data.datasets``, which pulls in
    TensorFlow at module scope.

    Args:
        n_time: Number of timesteps to simulate an epoch over.
        n_patches: Total patch positions per timestep (index-space stride); 1 outside
            patch mode.
        patches_per_epoch: Patch positions actually drawn per timestep per epoch; equal
            to n_patches outside patch mode or when unset.
        batch_size: Timesteps-worth of samples per training batch.
        seed: RNG seed for block shuffling (does not affect the result asymptotically,
            only which specific blocks land in the ragged remainder).

    Returns:
        Distinct-timestep materializations per epoch, divided by n_time. Exactly 1.0 when
        batch_size % patches_per_epoch == 0.
    """
    rng = np.random.default_rng(seed)
    block_timesteps = batch_size // math.gcd(batch_size, patches_per_epoch)
    n_full_blocks = n_time // block_timesteps
    full_block_starts = np.arange(n_full_blocks) * block_timesteps
    shuffled_starts = full_block_starts[rng.permutation(n_full_blocks)]

    # One timestep index per sample slot (not per global patch index -- read cost is per
    # timestep, regardless of which/how many patches of it a batch draws), in the same
    # block-shuffled-but-internally-ordered sequence _build_order emits.
    time_idxs = np.repeat(
        np.concatenate([np.arange(s, s + block_timesteps) for s in shuffled_starts])
        if n_full_blocks
        else np.array([], dtype=int),
        patches_per_epoch,
    )
    remainder_start = n_full_blocks * block_timesteps
    if remainder_start < n_time:
        remainder = np.repeat(np.arange(remainder_start, n_time), patches_per_epoch)
        time_idxs = np.concatenate([time_idxs, remainder])

    n_batches = math.ceil(len(time_idxs) / batch_size)
    distinct_total = sum(len(np.unique(time_idxs[b * batch_size : (b + 1) * batch_size])) for b in range(n_batches))
    return distinct_total / n_time


class ConfigGeometry:
    """Derived per-timestep training geometry for one config, used by both CLI modes."""

    def __init__(self, path: str, label: str) -> None:
        """Loads and derives geometry for one training config.

        Args:
            path: Path to the training config YAML.
            label: Human-readable label for this config in printed output.
        """
        self.path = path
        self.label = label
        yaml_data = utils.load_yaml(path)
        self.data_cfg = yaml_data.get("data_config", {})
        self.model_cfg = yaml_data.get("model_config", {})
        self.batch_size = int(self.data_cfg.get("batch_size", 4))

        self.n_lat, self.n_lon, self.used_domain_fallback = _n_lat_lon(self.data_cfg)
        self.n_levels, self.used_levels_fallback = _n_levels(self.data_cfg)
        self.patch = _parse_patch_config(self.data_cfg.get("patch_config"))

        if self.patch is None:
            self.samples_per_timestep = 1
            self.sample_lat_px = self.n_lat
            self.sample_lon_px = self.n_lon
            self.core_coverage = 1.0
            self.lon_buffer_factor = 1.0
            self.lat_buffer_factor = 1.0
        else:
            self.samples_per_timestep = self.patch["patches_per_epoch"]
            self.sample_lat_px = self.n_lat + 2 * self.patch["buffer_lat_px"]
            self.sample_lon_px = self.patch["patch_lon_width_px"] + 2 * self.patch["buffer_lon_px"]
            self.core_coverage = self.patch["patches_per_epoch"] * self.patch["patch_lon_width_px"] / self.n_lon
            self.lon_buffer_factor = self.sample_lon_px / self.patch["patch_lon_width_px"]
            self.lat_buffer_factor = self.sample_lat_px / self.n_lat

        # "Voxels" here means spatial-x-vertical grid cells (lat * lon * levels), the axes
        # patch/buffer geometry actually inflates. The variable/channel axis is deliberately
        # excluded: it is an architecture/input-selection choice orthogonal to patch
        # redundancy, and the two configs this script is meant to compare (the patch
        # ablation vs. its whole-domain baseline) use different variable lists for
        # unrelated reasons -- folding it in would corrupt the redundancy comparison.
        self.input_voxels_per_timestep = (
            self.samples_per_timestep * self.sample_lat_px * self.sample_lon_px * self.n_levels
        )

        # Scored voxels are the unbuffered, per-patch (or whole-domain) core region that
        # the loss actually supervises -- levels excluded, since targets have no vertical
        # axis; kept purely spatial so compute_efficiency isolates buffer waste from the
        # architecture's vertical extent.
        core_lon_px = self.patch["patch_lon_width_px"] if self.patch is not None else self.n_lon
        self.scored_spatial_voxels_per_timestep = self.samples_per_timestep * self.n_lat * core_lon_px
        computed_spatial_voxels_per_timestep = self.samples_per_timestep * self.sample_lat_px * self.sample_lon_px
        self.compute_efficiency = self.scored_spatial_voxels_per_timestep / computed_spatial_voxels_per_timestep

        self.total_samples = self.samples_per_timestep * ASSUMED_TRAIN_TIMESTEPS
        self.batches_per_epoch = math.ceil(self.total_samples / self.batch_size)
        n_patches_for_sim = self.patch["n_patches"] if self.patch is not None else 1
        self.read_amplification = _simulate_read_amplification(
            n_time=ASSUMED_TRAIN_TIMESTEPS,
            n_patches=n_patches_for_sim,
            patches_per_epoch=self.samples_per_timestep,
            batch_size=self.batch_size,
        )

    def print_report(self) -> None:
        """Prints this config's derived geometry, aligned for terminal reading."""
        print(f"\n=== {self.label}: {self.path} ===")
        if self.used_domain_fallback:
            print(
                f"  {'domain':28s}: no data_config.coordinates -- assuming full store domain "
                f"{FULL_DOMAIN_N_LAT} lat x {FULL_DOMAIN_N_LON} lon"
            )
        print(f"  {'n_lat x n_lon (core)':28s}: {self.n_lat} x {self.n_lon}")
        if self.used_levels_fallback:
            print(
                f"  {'n_levels':28s}: {self.n_levels} (volume_inputs=true, no pressure_levels listed -- "
                f"assuming all {NATIVE_STORE_LEVELS} native store levels)"
            )
        else:
            print(f"  {'n_levels':28s}: {self.n_levels}")
        if self.patch is not None:
            print(f"  {'patch_config key spelling':28s}: {self.patch['spelling']}")
            print(
                f"  {'patches_per_epoch / n_patches':28s}: {self.patch['patches_per_epoch']} / "
                f"{self.patch['n_patches']}"
            )
        print(f"  {'samples per timestep':28s}: {self.samples_per_timestep}")
        print(
            f"  {'model input shape / sample':28s}: (lat={self.sample_lat_px}, lon={self.sample_lon_px}, "
            f"levels={self.n_levels})"
        )
        print(f"  {'input voxels / timestep / ep':28s}: {self.input_voxels_per_timestep:,}")
        if self.patch is not None:
            print(f"  {'  core-coverage redundancy':28s}: {self.core_coverage:.3f}x")
            print(f"  {'  longitude buffer overhead':28s}: {self.lon_buffer_factor:.3f}x")
            print(f"  {'  latitude buffer overhead':28s}: {self.lat_buffer_factor:.3f}x")
        print(f"  {'scored spatial voxels / ts':28s}: {self.scored_spatial_voxels_per_timestep:,}")
        print(f"  {'compute efficiency':28s}: {self.compute_efficiency:.3%} (scored / computed, spatial only)")
        print(f"  {'batch_size':28s}: {self.batch_size}")
        print(
            f"  {'batches / epoch (assumed':28s}   {ASSUMED_TRAIN_TIMESTEPS} train timesteps): "
            f"{self.batches_per_epoch:,}"
        )
        print(
            f"  {'read amplification':28s}: {self.read_amplification:.3f}x distinct-timestep "
            "materializations per timestep"
        )


def run_analytic_mode(config_path: str, baseline_config_path: str) -> None:
    """Runs Mode 1: parse both configs and print the analytic per-epoch cost comparison.

    Args:
        config_path: Path to the patch-mode (or candidate) training config.
        baseline_config_path: Path to the whole-domain baseline training config.
    """
    patch_geom = ConfigGeometry(config_path, "Candidate config")
    baseline_geom = ConfigGeometry(baseline_config_path, "Baseline config")

    patch_geom.print_report()
    baseline_geom.print_report()

    # The headline ratio is per-timestep, not per-epoch: ASSUMED_TRAIN_TIMESTEPS cancels
    # out of it entirely as long as both configs train over the same number of timesteps,
    # which holds here (both hold out test_years=[2019]/val_years=[2018] at the same
    # "6h" time_resolution). Only the standalone "batches / epoch" and any absolute
    # epoch-time figure above depend on that assumed count.
    ratio = patch_geom.input_voxels_per_timestep / baseline_geom.input_voxels_per_timestep
    threshold = 1.05
    verdict = "PASS" if ratio <= threshold else "FAIL"

    print("\n=== Headline ===")
    print(
        f"  Candidate ({patch_geom.path}) computes {ratio:.3f}x the per-timestep input voxels of "
        f"baseline ({baseline_geom.path})."
    )
    print(f"  Success criterion: per-epoch cost at or below {threshold:.2f}x baseline -> {verdict}")
    if patch_geom.patch is not None:
        implied = patch_geom.core_coverage * patch_geom.lon_buffer_factor * patch_geom.lat_buffer_factor
        print(
            f"  (sanity check: core-coverage {patch_geom.core_coverage:.3f}x * lon-buffer "
            f"{patch_geom.lon_buffer_factor:.3f}x * lat-buffer {patch_geom.lat_buffer_factor:.3f}x = "
            f"{implied:.3f}x, matching the headline ratio up to the two configs' level-count difference)"
        )


def _build_timing_model(geom: ConfigGeometry, scale_factor: int) -> tuple[Any, tuple[int, int, int, int]]:
    """Builds the real UNet3Plus for one config at a (possibly scaled-down) spatial shape.

    Args:
        geom: This config's derived geometry (from Mode 1's parsing).
        scale_factor: Common divisor shrinking both spatial dims, so a CPU run finishes in
            reasonable time. The ratio between configs, not the absolute shape, is what
            stays meaningful after scaling.

    Returns:
        A (built_model, input_shape) tuple, where input_shape is
        (lat, lon, levels, n_variables) -- Mode 2 only supports volume_inputs configs,
        matching the two configs this script ships to compare.

    Raises:
        ValueError: If data_config.volume_inputs is not true (Mode 2's synthetic-data
            path only implements the 3D/volume input shape used by both shipped configs).
    """
    # Lazy import: keeps Mode 1 (this module's default path) free of a TensorFlow import.
    from fronts import model

    if not geom.data_cfg.get("volume_inputs", False):
        raise ValueError(
            f"{geom.path}: --measure only supports volume_inputs=true configs "
            "(the two configs this script ships to compare both set it)."
        )

    pool_size = geom.model_cfg.get("pool_size", [2, 2, 1])
    levels = int(geom.model_cfg.get("levels", 4))
    stride_lat = int(pool_size[0]) ** (levels - 1)
    stride_lon = int(pool_size[1]) ** (levels - 1)

    def _scaled(dim: int, stride: int) -> int:
        scaled = max(stride, (dim // scale_factor // stride) * stride)
        return scaled

    lat_px = _scaled(geom.sample_lat_px, stride_lat)
    lon_px = _scaled(geom.sample_lon_px, stride_lon)
    n_variables = len(geom.data_cfg.get("variables", []))
    input_shape = (lat_px, lon_px, geom.n_levels, n_variables)

    unet = model.UNet3Plus(
        input_shape=input_shape,
        num_classes=int(geom.model_cfg.get("n_classes", 6)),
        levels=levels,
        filter_num=geom.model_cfg.get("filter_num", [16, 32, 64, 128]),
        pool_size=pool_size,
        upsample_size=geom.model_cfg.get("upsample_size", pool_size),
        kernel_size=int(geom.model_cfg.get("kernel_size", 3)),
        squeeze_axes=geom.model_cfg.get("squeeze_axes"),
        first_encoder_connections=bool(geom.model_cfg.get("first_encoder_connections", False)),
        deep_supervision=bool(geom.model_cfg.get("deep_supervision", False)),
        batch_normalization=bool(geom.model_cfg.get("batch_normalization", True)),
        activation=geom.model_cfg.get("activation", "gelu"),
        output_activation=geom.model_cfg.get("output_activation", "softmax"),
        modules_per_node=int(geom.model_cfg.get("modules_per_node", 2)),
        normalization_method=geom.data_cfg.get("normalization_method", "minmax"),
        normalization_stat_a=np.zeros((geom.n_levels, n_variables), dtype=np.float32),
        normalization_stat_b=np.ones((geom.n_levels, n_variables), dtype=np.float32),
    ).build()
    return unet, input_shape


def _time_train_steps(unet: Any, batch_size: int, input_shape: tuple[int, ...], n_steps: int) -> list[float]:
    """Times ``n_steps`` synthetic forward+backward passes, discarding one warmup step.

    Args:
        unet: A built ``tf.keras.Model`` (possibly with multiple deep-supervision outputs).
        batch_size: Number of synthetic samples per step.
        input_shape: Per-sample input shape (lat, lon, levels, n_variables).
        n_steps: Number of *timed* steps to run (one extra warmup step runs first and is
            discarded, since it also pays for tf.function tracing).

    Returns:
        Wall-clock seconds for each of the n_steps timed steps (not including warmup).
    """
    # Lazy import: keeps Mode 1 (this module's default path) free of a TensorFlow import.
    import tensorflow as tf

    rng = np.random.default_rng(0)
    x = tf.constant(rng.standard_normal((batch_size, *input_shape)), dtype=tf.float32)
    outputs = unet.outputs if isinstance(unet.outputs, (list, tuple)) else [unet.outputs]
    y_list = [tf.constant(rng.standard_normal((batch_size, *out.shape[1:])), dtype=tf.float32) for out in outputs]
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    @tf.function
    def _step(x_batch: tf.Tensor, y_batch: list[tf.Tensor]) -> tf.Tensor:
        with tf.GradientTape() as tape:
            preds = unet(x_batch, training=True)
            preds = preds if isinstance(preds, (list, tuple)) else [preds]
            loss = tf.add_n([tf.reduce_mean(tf.square(p - y)) for p, y in zip(preds, y_batch, strict=True)])
        grads = tape.gradient(loss, unet.trainable_variables)
        optimizer.apply_gradients(zip(grads, unet.trainable_variables, strict=True))
        return loss

    _step(x, y_list)  # warmup: pays for tf.function tracing, excluded from timing

    step_times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        _step(x, y_list)
        step_times.append(time.perf_counter() - t0)
    return step_times


def run_measured_mode(
    config_path: str,
    baseline_config_path: str,
    scale_factor: int,
    n_steps: int,
    batch_size_override: int | None,
) -> None:
    """Runs Mode 2: build both real models and time actual forward+backward steps.

    Args:
        config_path: Path to the patch-mode (or candidate) training config.
        baseline_config_path: Path to the whole-domain baseline training config.
        scale_factor: Common divisor shrinking both configs' spatial dims for a
            CPU-feasible run. Only the ratio between the two configs stays meaningful.
        n_steps: Timed steps per config (after one discarded warmup step).
        batch_size_override: If set, overrides both configs' batch_size, e.g. to keep the
            whole-domain config's host RAM use bounded on a GPU-less dev machine.
    """
    print(
        "\n*** CAVEAT: this machine has no GPU. These are CPU forward+backward times, a rough "
        "relative-cost proxy only -- NOT an absolute prediction of cluster epoch time. Only the "
        "RATIO between the two configs below is meaningful. ***"
    )

    results = {}
    for path, label in [(config_path, "Candidate"), (baseline_config_path, "Baseline")]:
        geom = ConfigGeometry(path, label)
        batch_size = batch_size_override or geom.batch_size
        unet, input_shape = _build_timing_model(geom, scale_factor)
        print(
            f"\n=== {label}: {path} ===\n"
            f"  scaled input shape (lat, lon, levels, variables): {input_shape}\n"
            f"  batch_size: {batch_size} (samples_per_timestep={geom.samples_per_timestep})\n"
            f"  running {n_steps} timed steps ({1} warmup step discarded)..."
        )
        step_times = _time_train_steps(unet, batch_size, input_shape, n_steps)
        median_step_s = statistics.median(step_times)
        per_timestep_s = median_step_s * (geom.samples_per_timestep / batch_size)
        est_epoch_s = per_timestep_s * ASSUMED_TRAIN_TIMESTEPS
        print(
            f"  step times (s): {[f'{t:.3f}' for t in step_times]}\n"
            f"  median step time: {median_step_s:.3f} s\n"
            f"  implied time / training timestep: {per_timestep_s:.4f} s\n"
            f"  illustrative epoch time @ {ASSUMED_TRAIN_TIMESTEPS} timesteps: {est_epoch_s:.1f} s"
        )
        results[label] = per_timestep_s

    ratio = results["Candidate"] / results["Baseline"]
    print(f"\n=== Measured headline ===\n  Candidate / Baseline per-timestep step time: {ratio:.3f}x")


def main() -> None:
    """Parses CLI args and dispatches to Mode 1 (analytic) or Mode 2 (--measure)."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare patch-mode vs. whole-domain training's per-epoch compute cost, without "
            "needing a GPU or the icechunk data stores for the default analytic mode."
        )
    )
    parser.add_argument("--config", required=True, help="Candidate (typically patch-mode) training config YAML.")
    parser.add_argument(
        "--baseline-config", required=True, help="Whole-domain baseline training config YAML to compare against."
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="Also build the real models and time synthetic forward+backward steps on this machine's CPU.",
    )
    parser.add_argument(
        "--scale-factor",
        type=int,
        default=4,
        help="--measure only: common divisor shrinking both configs' spatial dims so a CPU run finishes "
        "quickly. Only the ratio between configs stays meaningful after scaling (default: 4).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="--measure only: number of timed train steps per config, after one discarded warmup step (default: 5).",
    )
    parser.add_argument(
        "--batch-size-override",
        type=int,
        default=None,
        help="--measure only: overrides both configs' batch_size, e.g. to keep the whole-domain config's "
        "host RAM use bounded on a GPU-less machine.",
    )
    args = parser.parse_args()

    run_analytic_mode(args.config, args.baseline_config)
    if args.measure:
        run_measured_mode(args.config, args.baseline_config, args.scale_factor, args.steps, args.batch_size_override)


if __name__ == "__main__":
    main()
