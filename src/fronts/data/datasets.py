import dataclasses
import math

import tensorflow as tf

from fronts import utils


@dataclasses.dataclass
class DatasetConfig:
    """Configuration for loading and splitting input and fronts data.

    Attributes:
        inputs_icechunk_config: Icechunk store config for input data.
        targets_icechunk_config: Icechunk store config for fronts data.
        variables: ERA5 variable names to load as input channels.
        train_split: Fraction of all filtered time steps to use for training.
        val_split: Fraction of all filtered time steps to use for validation.
        test_split: Fraction of timesteps per meteorological season to hold out as a sequestered
            test set (never seen during training or validation). train_split + val_split +
            test_split should sum to 1.
        batch_size: Number of timesteps per training batch.
        steps_per_epoch: Number of batches per epoch passed to model.fit. None lets
            the dataset size determine it (one full pass per epoch). A smaller value
            means each epoch covers only a subset, so a full training pass spans more
            epochs and the effective early-stopping patience is floored accordingly
            (see ``utils.epochs_per_full_pass``).
        load_chunk_steps: Number of steps' worth of samples to load per background
            prefetch. None falls back to steps_per_epoch.
        prefetch_chunks: Number of chunks to keep loaded in RAM ahead of the batch
            generator.
        load_num_workers: Dask threads per background chunk load. Peak host RAM
            scales with ``prefetch_chunks * load_num_workers``; keep small to
            bound memory.
        load_subblock: Maximum timesteps materialized per dask compute when
            gathering a chunk; caps the size of a single large allocation
            independently of ``load_chunk_steps``.
        class_weights: Per-class loss weights. None means equal weighting.
        front_dilation: Number of binary dilation iterations applied to each non-background
            front class. 0 means no dilation.
        time_resolution: Optional pandas offset string (e.g. ``"6h"``) used to subsample
            the loaded timesteps. Only timestamps whose hour is already aligned to this
            interval are kept (e.g. ``"6h"`` retains 00, 06, 12, 18 UTC). ``None`` keeps
            all available timesteps.
        input_sources: Optional additional gridded input sources (e.g. satellite
            groups). Their channels are concatenated after the ERA5 channels in
            the order listed, and their timestamps are intersected with ERA5 and
            fronts when aligning training data.
        norm_stats_cache_dir: Optional directory for caching normalization
            statistics, keyed by store snapshot, channels, and train indices.
            None recomputes the statistics on every run.
    """

    inputs_icechunk_config: utils.IcechunkStorageConfig
    targets_icechunk_config: utils.IcechunkStorageConfig
    variables: list[str]
    train_split: float
    val_split: float
    test_split: float
    batch_size: int = 4
    steps_per_epoch: int | None = None
    load_chunk_steps: int | None = None
    prefetch_chunks: int = 2
    load_num_workers: int = 4
    load_subblock: int = 32
    class_weights: list[float] | None = None
    front_dilation: int = 0
    time_resolution: str | None = None
    norm_stats_cache_dir: str | None = None


class TrainingDataset(tf.keras.utils.PyDataset):
    """TensorFlow Dataset for training a model on ERA5 (and possibly satellite) data.

    This dataset loads data from a store, applies necessary
    preprocessing and transformations, and yields batches of data suitable
    for training a machine learning model.

    Attributes:
        config: Configuration for loading and processing the data, including store paths, variables, batch size, etc.
    """

    def __init__(self, config: DatasetConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def __len__(self):
        """Returns the number of batches per epoch based on the total number of samples and batch size."""
        return math.ceil(len(self.x) / self.config.batch_size)

    def _extract_icechunk_data(self, idx):
        """Extracts a batch of data from the icechunk store based on the provided index."""

    def __getitem__(self, idx):
        """Retrieves a batch of data for the given index, applying necessary preprocessing and transformations."""
        pass
