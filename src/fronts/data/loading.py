"""Load, align, and encode gridded input sources and fronts data for training.

``load_training_data`` opens the icechunk stores and delegates the I/O-free
alignment/encoding to ``assemble_training_data``, which is unit-testable with
in-memory datasets. Both return a :class:`TrainingData` carrying the raw
store-axis sources plus a logical-to-native position map; aligned views for
normalization stats, splitting, and logging are derived lazily via a single
positional take per source.
"""

import dataclasses
import logging

import numpy as np
import xarray as xr

from fronts import utils
from fronts.data import config, inputs, targets
from fronts.utils import apply_time_resolution

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TrainingData:
    """Aligned, encoded training sources plus the common time axis they share.

    Attributes:
        input_sources: One source per input, concatenated along ``channel`` during
            batching. Each carries the raw store time axis plus a position map into it.
        target_source: One-hot (optionally dilated) front target source.
        times: The common, filtered timestamps shared by every source, in sample order.
    """

    input_sources: list[inputs.LazyTimeSource]
    target_source: inputs.LazyTimeSource
    times: np.ndarray

    def lazy_inputs(self, idxs: "np.ndarray | list[int] | None" = None) -> xr.DataArray:
        """Return the lazy aligned input DataArray, optionally restricted to logical ``idxs``.

        Uses a single positional take per source (``array.isel(time=positions[idxs])``)
        so the time axis is never collapsed into one chunk and materialized whole.

        Args:
            idxs: Logical sample indices to select. None selects every aligned sample.

        Returns:
            Lazy float32 DataArray of shape (time, latitude, longitude, channel).
        """
        pieces: list[xr.DataArray] = []
        for source in self.input_sources:
            positions = source.positions if idxs is None else source.positions[np.asarray(idxs)]
            pieces.append(source.array.isel(time=positions))
        return pieces[0] if len(pieces) == 1 else xr.concat(pieces, dim="channel")

    @property
    def input_aligned(self) -> xr.DataArray:
        """Lazy aligned input DataArray over every sample (time, lat, lon, channel)."""
        return self.lazy_inputs()

    @property
    def target_aligned(self) -> xr.DataArray:
        """Lazy aligned target DataArray over every sample (time, lat, lon, class)."""
        return self.target_source.array.isel(time=self.target_source.positions)


def assemble_training_data(
    raw_sources: list[tuple[config.InputSourceConfig, xr.Dataset]],
    raw_fronts_da: xr.DataArray,
    data_config: config.DataConfig,
    seed: int = 0,
) -> TrainingData:
    """Align opened source datasets and fronts into a :class:`TrainingData` without any I/O.

    Intersects the source and fronts time axes, optionally subsamples by
    ``time_resolution``, applies the Justin et al. (2025) presence filter, then builds
    one ``LazyTimeSource`` per input plus the encoded target source. Input channels are
    ordered by ``raw_sources`` (ERA5 first), each source contributing
    ``era5_to_dataarray`` channels.

    Args:
        raw_sources: Pairs of (source config, opened raw Dataset). Each dataset must have
            a ``time`` dimension for single-take chunk loading.
        raw_fronts_da: Opened raw fronts identifier DataArray (time, latitude, longitude).
        data_config: DataConfig supplying variables, time_resolution, and front_dilation.
        seed: Seed for the RNG used by the presence filter.

    Returns:
        A :class:`TrainingData` with one input source per entry in ``raw_sources``.

    Raises:
        ValueError: If any input source lacks a ``time`` dimension.
    """
    fronts_da = utils.drop_duplicate_times(raw_fronts_da)

    common_times = fronts_da.time.values
    for _source, raw_ds in raw_sources:
        if "time" in raw_ds.dims:
            common_times = np.intersect1d(common_times, raw_ds.time.values)
    if data_config.time_resolution is not None:
        common_times = apply_time_resolution(common_times, data_config.time_resolution)
        logger.info("After time_resolution=%r filter: %d steps", data_config.time_resolution, len(common_times))

    rng = np.random.default_rng(seed)
    keep = targets.filter_timesteps(fronts_da.sel(time=common_times), rng)
    common_times = common_times[keep]
    logger.info("Matched time steps: %d", len(common_times))

    input_sources: list[inputs.LazyTimeSource] = []
    for source, raw_ds in raw_sources:
        if "time" not in raw_ds.dims:
            raise ValueError(f"Input source '{source.name}' has no time dimension; single-take loading requires one.")
        array = inputs.era5_to_dataarray(raw_ds, source.variables)
        input_sources.append(inputs.LazyTimeSource.aligned(array, raw_ds.time.values, common_times))

    target_array = targets.encode_targets(raw_fronts_da, data_config.front_dilation)
    target_source = inputs.LazyTimeSource.aligned(target_array, raw_fronts_da.time.values, common_times)

    return TrainingData(input_sources=input_sources, target_source=target_source, times=common_times)


def load_training_data(data_config: config.DataConfig, seed: int = 0) -> TrainingData:
    """Open the ERA5 and fronts icechunk stores and assemble aligned training data.

    Opens the ERA5 store plus any additional sources in ``data_config.input_sources``,
    then delegates alignment and encoding to :func:`assemble_training_data`.

    Args:
        data_config: DataConfig specifying store paths, branch names, variables, and splits.
        seed: Integer seed for the RNG used when subsampling timesteps.

    Returns:
        A :class:`TrainingData` ready for splitting, normalization stats, and batching.
    """
    source_configs = [
        config.InputSourceConfig(
            name="era5",
            icechunk_config=data_config.era5_icechunk_config,
            variables=data_config.variables,
        ),
        *(data_config.input_sources or []),
    ]

    raw_sources: list[tuple[config.InputSourceConfig, xr.Dataset]] = []
    for source in source_configs:
        logger.info("Loading input source '%s'...", source.name)
        ds = utils.open_readonly_icechunk_store(
            store_path=source.icechunk_config.store_path,
            branch=source.icechunk_config.branch_name,
            group=source.icechunk_config.group_name,
            zarr_format=source.icechunk_config.zarr_format,
            virtual_chunk_local_path=source.icechunk_config.virtual_chunk_local_path,
        )
        logger.info("Source '%s' store: %s", source.name, ds)
        raw_sources.append((source, ds))

    logger.info("Loading fronts...")
    raw_fronts_da = utils.open_readonly_icechunk_store(
        store_path=data_config.fronts_icechunk_config.store_path,
        branch=data_config.fronts_icechunk_config.branch_name,
        group=data_config.fronts_icechunk_config.group_name,
        zarr_format=data_config.fronts_icechunk_config.zarr_format,
        virtual_chunk_local_path=data_config.fronts_icechunk_config.virtual_chunk_local_path,
    )["identifier"]
    logger.info("Fronts store: %s", raw_fronts_da)

    return assemble_training_data(raw_sources, raw_fronts_da, data_config, seed)
