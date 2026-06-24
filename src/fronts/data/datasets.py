import dataclasses
import logging

import numpy as np
import tensorflow as tf
import xarray as xr

from fronts import utils
from fronts.data import inputs, targets

logger = logging.getLogger(__name__)


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
        max_queue_size: Maximum number of prefetched batches kept in RAM ahead of the
            training loop (passed to ``tf.keras.utils.PyDataset(max_queue_size=...)``).
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
    max_queue_size: int = 4


def load_timesteps(
    input_ds: xr.Dataset,
    target_da: xr.DataArray,
    data_config: DatasetConfig,
    idxs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Load specific timesteps from the icechunk store into RAM.

    Mirrors the per-batch logic previously in TrainingDataset.get_at_indices,
    but accepts arbitrary indices and is intended for one-off loads (e.g.
    test-set visualization) rather than the training loop.

    Args:
        input_ds: Split input Dataset.
        target_da: Split target DataArray.
        data_config: Dataset configuration.
        idxs: Integer indices into the time axis to load.

    Returns:
        Tuple of (x, y) as float32 numpy arrays, shapes
        (len(idxs), lat, lon, channel) and (len(idxs), lat, lon, class).
    """
    x_xarray = input_ds.isel(time=idxs)
    y_raw = target_da.isel(time=idxs)

    x = inputs.inputs_ds_to_dataarray(x_xarray, data_config.variables).values.astype(np.float32)

    y_da = targets.one_hot_encode_to_dataarray(targets.remap_fronts(y_raw))
    if data_config.front_dilation > 0:
        y_da = targets.dilate_fronts(y_da, data_config.front_dilation)

    return x, y_da.values.astype(np.float32)


def build_tf_dataset(
    input_ds: xr.Dataset,
    target_da: xr.DataArray,
    data_config: DatasetConfig,
    batch_size: int,
    n_channels: int,
    n_classes: int = 6,
    shuffle: bool = False,
    seed: int = 0,
) -> tuple[tf.data.Dataset, int]:
    """Build a lazy tf.data.Dataset that reads from the icechunk store per-timestep.

    Reads are issued one timestep at a time via tf.py_function, keeping individual
    Lustre reads small. MirroredStrategy integrates natively with tf.data, avoiding
    the PyDataset/AllReduce deadlock.

    Args:
        input_ds: This split's input Dataset (not loaded into RAM).
        target_da: This split's target DataArray (not loaded into RAM).
        data_config: Dataset configuration.
        batch_size: Number of timesteps per batch.
        n_channels: Number of input channels after variable expansion (e.g. 77).
        n_classes: Number of output classes including background (default 6).
        shuffle: Whether to shuffle each epoch.
        seed: Random seed for shuffling.

    Returns:
        Tuple of (tf.data.Dataset, n_samples).
    """
    import logging

    logger = logging.getLogger(__name__)

    n_samples = input_ds.sizes["time"]
    n_lat = input_ds.sizes["latitude"]
    n_lon = input_ds.sizes["longitude"]

    logger.info(
        "Building tf.data pipeline: %d timesteps, %d lat, %d lon, %d channels, %d classes",
        n_samples,
        n_lat,
        n_lon,
        n_channels,
        n_classes,
    )

    # Shuffle happens over indices, not data — no Lustre reads here
    index_ds = tf.data.Dataset.range(n_samples)
    if shuffle:
        index_ds = index_ds.shuffle(
            buffer_size=n_samples,
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def load_single_timestep(idx: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Load one timestep from the icechunk store."""
        i = int(idx.numpy())
        x_xarray = input_ds.isel(time=[i])
        y_raw = target_da.isel(time=[i])

        x = inputs.inputs_ds_to_dataarray(x_xarray, data_config.variables).values.astype(
            np.float32
        )  # (lat, lon, channel)

        y_da = targets.one_hot_encode_to_dataarray(targets.remap_fronts(y_raw))
        if data_config.front_dilation > 0:
            y_da = targets.dilate_fronts(y_da, data_config.front_dilation)

        return x[0], y_da.values.astype(np.float32)[0]  # (lat, lon, class)

    def tf_load(idx: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x, y = tf.py_function(
            func=load_single_timestep,
            inp=[idx],
            Tout=[tf.float32, tf.float32],
        )
        # tf.py_function loses static shape info — restore it so
        # MirroredStrategy can split batches across GPUs correctly
        x.set_shape([n_lat, n_lon, n_channels])
        y.set_shape([n_lat, n_lon, n_classes])
        return x, y

    dataset = (
        index_ds.map(tf_load, num_parallel_calls=1).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    )

    return dataset, n_samples
