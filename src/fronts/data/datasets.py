import dataclasses
import logging
import math
import time

import numpy as np
import tensorflow as tf
import xarray as xr

from fronts import utils
from fronts.data import inputs, targets

logger = logging.getLogger(__name__)


def compute_patch_lon_starts(n_lon_core: int, patch_width: int, n_patches: int) -> np.ndarray:
    """Evenly spaced starting pixel offsets for sliding longitude windows.

    Reproduces "nine pairs of images evenly spaced along the longitude" (Justin et al.
    2025): ``n_patches`` windows of ``patch_width`` pixels tiled across a core domain of
    ``n_lon_core`` pixels, first window flush with the west edge, last flush with the east
    edge, evenly spaced in between.

    Args:
        n_lon_core: Width of the (unbuffered) core longitude domain, in grid pixels.
        patch_width: Width of each patch's core region, in grid pixels.
        n_patches: Number of patch positions.

    Returns:
        Integer array of shape (n_patches,), each a 0-indexed starting pixel offset into
        the core domain.

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
            ``model_config.levels - 1`` pooling stages) or the model fails to build — see
            the "What We're NOT Doing" note on stride validation in
            docs/rse/specs/plan-patch-buffer-training.md.
        flip_probability: Independent per-axis probability of flipping a training patch
            along latitude and along longitude. 0.25 reproduces the paper's rate (a
            1 - (1 - p)^2 = 43.75% chance of at least one flip at p=0.25). Applied only to
            the train split.
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


@dataclasses.dataclass
class DatasetConfig:
    """Configuration for loading and splitting input and fronts data.

    Attributes:
        inputs_icechunk_config: Icechunk store config for ERA5 input data.
        targets_icechunk_config: Icechunk store config for fronts data.
        variables: ERA5 variable names to load as input channels.
        test_years: Calendar years to hold out as the sequestered test set (never seen
            during training or validation).
        val_years: Calendar years to hold out for validation. Must not overlap test_years.
            All years not in test_years or val_years are used for training.
        batch_size: Number of timesteps per training batch.
        class_weights: Per-class loss weights. None means equal weighting.
        front_dilation: Number of binary dilation iterations applied to each non-background
            front class. 0 means no dilation.
        time_resolution: Optional pandas offset string (e.g. ``"6h"``) used to subsample
            the loaded timesteps. Only timestamps whose hour is already aligned to this
            interval are kept (e.g. ``"6h"`` retains 00, 06, 12, 18 UTC). ``None`` keeps
            all available timesteps.
        norm_stats_cache_dir: Optional directory for caching normalization
            statistics, keyed by store snapshot, channels, and train indices.
            None recomputes the statistics on every run.
        normalization_method: "standardization" normalizes inputs by z-score
            (mean/variance); "minmax" rescales inputs to their min/max range. See
            ``fronts.data.inputs.compute_norm_stats`` and ``fronts.model.UNet3Plus``.
        max_queue_size: Maximum number of prefetched batches kept in RAM ahead of the
            training loop (passed to ``tf.keras.utils.PyDataset(max_queue_size=...)``).
        max_pydataset_workers: Maximum number of threads used by ``tf.keras.utils.PyDataset`` to
            load batches in parallel. None uses the number of CPUs allocated to the job.
        coordinates: Optional spatial bounding box to restrict inputs and targets to before
            batching (e.g. a CONUS crop). None trains on the full domain loaded from the
            icechunk stores.
        volume_inputs: If True, batches keep the vertical structure as a separate axis —
            shape (batch, latitude, longitude, level, variable) for a 3D Conv3D model —
            instead of flattening level and variable into one channel axis for a 2D model.
        pressure_levels: Pressure levels (hPa) to select from the icechunk store's
            ``level`` dimension. None keeps every level already present in the store.
            Must be a subset of the levels the store was generated with (see
            ``fronts.data.generate.ERA5DataLoaderConfig.pressure_levels``).
        patch_config: Optional sliding-window longitude patch extraction (with optional
            context buffer and flip augmentation) reproducing Justin et al. (2025)'s
            original training regime. None trains on the full ``coordinates``-cropped
            domain per timestep, as today. Requires ``coordinates`` to be set.
    """

    inputs_icechunk_config: utils.IcechunkStorageConfig
    targets_icechunk_config: utils.IcechunkStorageConfig
    variables: list[str]
    test_years: list[int]
    val_years: list[int]
    batch_size: int = 4
    class_weights: list[float] | None = None
    front_dilation: int = 0
    time_resolution: str = "6h"
    norm_stats_cache_dir: str | None = None
    normalization_method: inputs.NormalizationMethod = "standardization"
    max_queue_size: int = 4
    max_pydataset_workers: int = 16
    coordinates: utils.BoundingBox | None = None
    volume_inputs: bool = False
    pressure_levels: list[int] | None = None
    patch_config: PatchConfig | None = None


class FrontsPyDataset(tf.keras.utils.PyDataset):
    """Batches a split's ERA5/fronts DataArrays for training or evaluation via the PyDataset interface.

    Each ``__getitem__`` call gathers exactly one batch's timesteps with a single
    ``isel(time=idxs)`` take. ``input_ds``/``target_da`` must already be sliced
    to this split (e.g. ``input_ds.isel(time=train_indices)``) and backed by non-dask
    (``chunks=None``) arrays so each take reads directly through the zarr store rather
    than building a dask graph; concurrency across batches comes entirely from
    ``tf.keras.utils.PyDataset``'s own thread pool (``workers``/``max_queue_size``
    passed through ``**kwargs``).

    Yields a single (unreplicated) target per batch — the model's
    ``SharedTargetModel`` (see ``fronts.model``) is responsible for broadcasting it
    across any deep-supervision outputs, not the dataset.

    Shuffling is block-aligned at the *timestep* level, not the flat sample-index level:
    blocks of ``batch_size // gcd(batch_size, n_patches)`` contiguous timesteps (all
    patches of each timestep included, in time order) are kept together, and only the
    order in which blocks are visited is randomized per epoch. That block size is the
    smallest one whose patches divide evenly into whole batches, so every batch's
    ``.isel(time=...)`` read lands on a single contiguous run of timesteps instead of
    up to ``batch_size`` scattered ones. In non-patch mode (``n_patches == 1``) this
    reduces to one block per batch — a straight contiguous ``batch_size``-timestep read.
    Both icechunk stores backing this dataset chunk at 1 timestep, so a fully random
    per-sample shuffle measured 10-30x slower than a sequential read of the same size
    (see ``scripts/diagnose_read_throughput.py``).

    Attributes:
        input_ds: This split's input Dataset, shape (time, latitude, longitude) per variable.
        target_da: This split's raw integer front-code DataArray, shape (time, latitude, longitude).
        batch_size: Number of timesteps per batch.
        shuffle: If True, reshuffles the sample order at the end of every epoch.
        drop_remainder: If True, drop the final under-sized batch instead of yielding it,
            so every batch has exactly ``batch_size`` samples. A trailing batch smaller
            than ``batch_size`` splits unevenly across replicas under
            ``tf.distribute.MirroredStrategy``, which triggers a cuDNN backend bug
            (``CUDNN_STATUS_BAD_PARAM`` in ``Conv3DBackpropFilterV2``) on that batch's
            backward pass (see https://github.com/tensorflow/tensorflow/issues/60935).
    """

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
        self._order = self._build_order() if shuffle else np.arange(self._total)

    @property
    def _total(self) -> int:
        return self.input_ds.sizes["time"] * self._n_patches

    @property
    def n_samples(self) -> int:
        """Number of individual samples in this split (timesteps, or timesteps x patches in patch mode)."""
        return self._total

    def patch_sample_index(self, time_idx: int, patch_idx: int = 0) -> int:
        """Map a timestep index to a global sample index, picking one patch within it.

        In patch mode, ``get_at_indices`` expects global sample indices
        (``time_idx * n_patches + patch_idx``), not raw timestep indices. Non-patch mode
        returns ``time_idx`` unchanged, since ``n_patches`` is 1.
        """
        return time_idx * self._n_patches + patch_idx

    def __len__(self) -> int:
        """Returns the number of batches per epoch."""
        if self.drop_remainder:
            return self._total // self.batch_size
        return math.ceil(self._total / self.batch_size)

    def _build_order(self) -> np.ndarray:
        """Builds a shuffled global sample-index order, block-aligned by timestep.

        Groups every patch of each timestep together (in time order) into blocks of
        ``batch_size // gcd(batch_size, n_patches)`` contiguous timesteps, then shuffles
        the order in which whole blocks are visited. That block size is the smallest one
        whose sample count (``block_timesteps * n_patches``) is a multiple of
        ``batch_size``, so batch boundaries always land on block boundaries — every batch
        drawn from the resulting order is one contiguous, in-order run of timesteps,
        never a scattered or straddled one. If the total timestep count isn't an exact
        multiple of the block size, the leftover timesteps form a final ragged block that
        is always placed last (never shuffled into the middle), matching
        ``drop_remainder``'s existing "final batch may be undersized" behavior instead of
        introducing a new mid-epoch discontinuity. See the class docstring for why
        contiguity matters for read throughput.
        """
        n_time = self.input_ds.sizes["time"]
        block_timesteps = self.batch_size // math.gcd(self.batch_size, self._n_patches)
        n_full_blocks = n_time // block_timesteps
        full_block_starts = np.arange(n_full_blocks) * block_timesteps
        shuffled_starts = full_block_starts[self._rng.permutation(n_full_blocks)]
        order = np.concatenate(
            [
                np.arange(start * self._n_patches, (start + block_timesteps) * self._n_patches)
                for start in shuffled_starts
            ]
            or [np.array([], dtype=int)]
        )
        remainder_start = n_full_blocks * block_timesteps
        if remainder_start < n_time:
            remainder = np.arange(remainder_start * self._n_patches, n_time * self._n_patches)
            order = np.concatenate([order, remainder])
        return order

    def on_epoch_end(self) -> None:
        """Reshuffles the batch visitation order for the next epoch, if shuffling is enabled."""
        if self.shuffle:
            self._order = self._build_order()

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

        # Convert inputs to a DataArray — (time, latitude, longitude, channel) for 2D models,
        # (time, latitude, longitude, level, variable) for 3D — and load into memory as float32.
        if self.data_config.volume_inputs:
            x = inputs.inputs_ds_to_volume_dataarray(x_xarray, self.data_config.variables).values
        else:
            x = inputs.inputs_ds_to_dataarray(x_xarray, self.data_config.variables).values

        # One-hot encode targets, remap front classes to the configured set, and load into memory as float32.
        # Dilate fronts if > 0
        y_da = targets.one_hot_encode_to_dataarray(targets.remap_fronts(y_da))
        if self.data_config.front_dilation > 0:
            y_da = targets.dilate_fronts(y_da, self.data_config.front_dilation)

        # Convert to numpy arrays in memory. The model's SharedTargetModel is responsible for broadcasting the single
        # target across any deep-supervision outputs, not the dataset.
        y = y_da.values
        return x, y

    def _get_patches_at_indices(self, idxs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Patch-mode ``get_at_indices``.

        Slices a buffered input window and unbuffered target window per sample, then
        applies flip augmentation if ``self.augment``.
        """
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
        y = np.stack([y_full[i, :, starts[p] : starts[p] + width, :] for i, p in enumerate(patch_idxs)], axis=0)

        if self.augment and pc.flip_probability > 0:
            x, y = self._apply_flip_augmentation(x, y, pc.flip_probability)
        return x, y

    def _apply_flip_augmentation(
        self, x: np.ndarray, y: np.ndarray, flip_probability: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Independently flips each sample along latitude and/or longitude.

        Applies with probability ``flip_probability`` per axis — Justin et al. (2025)'s
        augmentation.
        """
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

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns the (input, target) batch at ``idx``."""
        local_idxs = self._order[idx * self.batch_size : (idx + 1) * self.batch_size]
        t0 = time.time()
        result = self.get_at_indices(local_idxs)
        elapsed = time.time() - t0
        if elapsed > 30:
            logger.warning(f"Slow batch {idx}: {elapsed:.1f}s")
        return result
