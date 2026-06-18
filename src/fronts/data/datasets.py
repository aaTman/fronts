import dataclasses
import math

import numpy as np
import tensorflow as tf
import xarray as xr

from fronts import utils


@dataclasses.dataclass
class DatasetConfig:
    """Configuration for loading and splitting input and fronts data.

    Attributes:
        inputs_icechunk_config: Icechunk store config for ERA5 input data.
        targets_icechunk_config: Icechunk store config for fronts data.
        variables: ERA5 variable names to load as input channels.
        train_split: Fraction of all filtered time steps to use for training.
        val_split: Fraction of all filtered time steps to use for validation.
        test_split: Fraction of timesteps per meteorological season to hold out as a sequestered
            test set (never seen during training or validation). train_split + val_split +
            test_split should sum to 1.
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
        prefetch_workers: Number of background threads ``TrainingDataset`` uses to
            prefetch batches ahead of consumption (passed to
            ``tf.keras.utils.PyDataset(workers=...)``).
        max_queue_size: Maximum number of prefetched batches kept in RAM ahead of the
            training loop (passed to ``tf.keras.utils.PyDataset(max_queue_size=...)``).
    """

    inputs_icechunk_config: utils.IcechunkStorageConfig
    targets_icechunk_config: utils.IcechunkStorageConfig
    variables: list[str]
    train_split: float
    val_split: float
    test_split: float
    batch_size: int = 4
    class_weights: list[float] | None = None
    front_dilation: int = 0
    time_resolution: str | None = None
    norm_stats_cache_dir: str | None = None
    prefetch_workers: int = 2
    max_queue_size: int = 4


class TrainingDataset(tf.keras.utils.PyDataset):
    """Batches a split's ERA5/fronts DataArrays for training via the PyDataset interface.

    Each ``__getitem__`` call gathers exactly one batch's timesteps with a single
    ``isel(time=idxs)`` take. ``input_array``/``target_array`` must already be sliced
    to this split (e.g. ``input_da.isel(time=train_indices)``) and backed by non-dask
    (``chunks=None``) arrays so each take reads directly through the zarr store rather
    than building a dask graph; concurrency across batches comes entirely from
    ``tf.keras.utils.PyDataset``'s own thread pool (``workers``/``max_queue_size``
    passed through ``**kwargs``).

    Attributes:
        input_array: This split's input DataArray, shape (time, latitude, longitude, channel).
        target_array: This split's target DataArray, shape (time, latitude, longitude, class).
        batch_size: Number of timesteps per batch.
        n_supervision_outputs: Number of deep supervision outputs; the target is
            replicated this many times per batch.
        shuffle: If True, reshuffles the sample order at the end of every epoch.
    """

    def __init__(
        self,
        input_array: xr.DataArray,
        target_array: xr.DataArray,
        batch_size: int,
        n_supervision_outputs: int,
        shuffle: bool = False,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if input_array.sizes["time"] != target_array.sizes["time"]:
            raise ValueError(
                f"Input and target time lengths differ: {input_array.sizes['time']} vs {target_array.sizes['time']}"
            )
        self.input_array = input_array
        self.target_array = target_array
        self.batch_size = batch_size
        self.n_supervision_outputs = n_supervision_outputs
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)
        self._order = self._rng.permutation(self._total) if shuffle else np.arange(self._total)

    @property
    def _total(self) -> int:
        return self.input_array.sizes["time"]

    def __len__(self) -> int:
        """Returns the number of batches per epoch."""
        return math.ceil(self._total / self.batch_size)

    def on_epoch_end(self) -> None:
        """Reshuffles the sample order for the next epoch, if shuffling is enabled."""
        if self.shuffle:
            self._order = self._rng.permutation(self._total)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        """Returns the (input, target) batch at ``idx``, target replicated per supervision output."""
        local_idxs = self._order[idx * self.batch_size : (idx + 1) * self.batch_size]
        x = self.input_array.isel(time=local_idxs).values
        y = self.target_array.isel(time=local_idxs).values
        return x, tuple(y for _ in range(self.n_supervision_outputs))
