import argparse
import dataclasses
import logging
import sys
from collections.abc import Iterator

import dask.utils
import icechunk as ic
import icechunk.xarray
import numpy as np
import obstore.store
import pandas as pd
import tqdm
import xarray as xr
import zarr.storage

from fronts import utils
from fronts.data import config, derived

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


@dataclasses.dataclass
class StoreContents:
    """Snapshot of what is currently present in an icechunk store.

    Attributes:
        variables: Data variable names found in the store.
        times: DatetimeIndex of time steps present.
        levels: Pressure levels stored.
        coordinates: Spatial bounding box of stored data.
    """

    variables: list[str]
    times: pd.DatetimeIndex
    levels: list[int]
    coordinates: utils.BoundingBox


@dataclasses.dataclass
class WriteStrategy:
    """Decision record produced by determine_write_strategy.

    Fields are not mutually exclusive: both missing_variables and missing_times may be
    set, in which case execute appends the times first then writes the new variables.
    - skip_reason non-empty: log and return immediately.
    - error_reason non-empty: raise ValueError with the reason.
    - Otherwise: proceed using missing_variables and/or missing_times.

    Attributes:
        missing_variables: Variables absent from the store that need to be added.
        missing_times: Time steps present in the request but absent from the store.
        skip_reason: Non-empty when no write is needed; human-readable explanation.
        error_reason: Non-empty when write is impossible; raised as ValueError.
    """

    missing_variables: list[str]
    missing_times: pd.DatetimeIndex
    skip_reason: str
    error_reason: str
    merge_required: bool

    def execute(
        self,
        era5_config: config.ERA5DataLoaderConfig,
        icechunk_config: config.IcechunkStorageConfig,
        slurm_config: config.SlurmConfig | None = None,
    ) -> None:
        """Generate and write the ERA5 data described by this strategy.

        Args:
            era5_config: Base ERA5 configuration; time range or variables will be
                narrowed to only what is missing before generating data.
            icechunk_config: Configuration for the target icechunk store.
            slurm_config: SLURM cluster parameters, used to size write chunks by
                available memory. If None, a fixed fallback memory budget is used.
        """
        if slurm_config is not None:
            memory_bytes = dask.utils.parse_bytes(slurm_config.memory) // slurm_config.cores
        else:
            memory_bytes = _FALLBACK_MEMORY_BYTES

        if self.missing_times.size:
            if self.merge_required:
                ds_download = generate_era5_download_data(era5_config).sel(time=self.missing_times)
            else:
                narrow_config = dataclasses.replace(
                    era5_config,
                    time_start=self.missing_times[0].to_pydatetime(),
                    time_end=self.missing_times[-1].to_pydatetime(),
                )
                ds_download = generate_era5_download_data(narrow_config)

            logger.info("Phase 1: downloading raw ERA5 variables to icechunk store...")
            write_or_append_icechunk_store(icechunk_config, ds_download)

            _, derived_vars = derived.classify_variables(era5_config.variables, set(ds_download.data_vars))
            if derived_vars:
                logger.info("Phase 2: computing derived variables from local store...")
                time_chunks = _compute_time_chunks(self.missing_times.values, ds_download, memory_bytes)
                required_inputs = {
                    inp for v in derived_vars for inp in derived.DERIVED_VARIABLE_REGISTRY[v].required_inputs
                }
                local_ds = utils.open_readonly_icechunk_store(
                    icechunk_config.store_path,
                    icechunk_config.branch_name,
                    group=icechunk_config.group_name,
                    zarr_format=icechunk_config.zarr_format,
                    drop_variables=_non_input_data_variables(icechunk_config, required_inputs),
                ).sel(time=self.missing_times)
                if not local_ds.chunks:
                    local_ds = local_ds.chunk(era5_config.chunks)
                derived_config = dataclasses.replace(era5_config, variables=derived_vars)
                ds_derived = generate_era5_derived_data(derived_config, local_ds)
                write_variables_to_icechunk_store(icechunk_config, ds_derived, time_chunks=time_chunks)

        if self.missing_variables:
            current_store = inspect_store(icechunk_config)
            still_missing = [v for v in self.missing_variables if v not in current_store.variables]
            if still_missing:
                narrow_config = dataclasses.replace(era5_config, variables=still_missing)
                logger.info("Generating ERA5 data for missing variables...")
                all_times = current_store.times
                ds_download = generate_era5_download_data(narrow_config).sel(time=all_times)

                time_chunks = _compute_time_chunks(all_times.values, ds_download, memory_bytes)

                vars_to_download = [v for v in ds_download.data_vars if v not in current_store.variables]
                if vars_to_download:
                    logger.info("Writing missing direct variables to icechunk store...")
                    write_variables_to_icechunk_store(
                        icechunk_config, ds_download[vars_to_download], time_chunks=time_chunks
                    )

                _, derived_vars = derived.classify_variables(narrow_config.variables, set(ds_download.data_vars))
                if derived_vars:
                    logger.info("Computing and writing missing derived variables...")
                    required_inputs = {
                        inp for v in derived_vars for inp in derived.DERIVED_VARIABLE_REGISTRY[v].required_inputs
                    }
                    local_ds = utils.open_readonly_icechunk_store(
                        icechunk_config.store_path,
                        icechunk_config.branch_name,
                        group=icechunk_config.group_name,
                        zarr_format=icechunk_config.zarr_format,
                        drop_variables=_non_input_data_variables(icechunk_config, required_inputs),
                    ).sel(time=all_times)
                    if not local_ds.chunks:
                        local_ds = local_ds.chunk(era5_config.chunks)
                    derived_config = dataclasses.replace(narrow_config, variables=derived_vars)
                    ds_derived = generate_era5_derived_data(derived_config, local_ds)
                    write_variables_to_icechunk_store(icechunk_config, ds_derived, time_chunks=time_chunks)


def _open_era5_zarr_store(era5_config: config.ERA5DataLoaderConfig) -> xr.Dataset:
    """Open the lazy ERA5 zarr dataset described by ``era5_config``.

    For ``gs://`` URIs, the store is opened via obstore's GCS backend, which is
    substantially faster than the default fsspec/gcsfs backend for the many
    per-chunk reads done while writing icechunk stores. Other URIs (e.g. local
    paths used in tests) are opened directly with ``xr.open_zarr``.

    Args:
        era5_config: Configuration containing the ERA5 URI and storage options.

    Returns:
        The opened (lazy, unchunked-per-zarr-chunk) ERA5 dataset.
    """
    if era5_config.era5_uri.startswith("gs://"):
        gcs_config: dict = {}
        if era5_config.storage_options and era5_config.storage_options.get("token") == "anon":
            gcs_config["skip_signature"] = True
        object_store = obstore.store.GCSStore.from_url(era5_config.era5_uri, config=gcs_config)
        return xr.open_zarr(zarr.storage.ObjectStore(object_store, read_only=True), chunks={})

    open_kwargs: dict = {"chunks": {}}
    if era5_config.storage_options:
        open_kwargs["storage_options"] = era5_config.storage_options
    return xr.open_zarr(era5_config.era5_uri, **open_kwargs)


def generate_era5_download_data(era5_config: config.ERA5DataLoaderConfig) -> xr.Dataset:
    """Open ARCO and subset to the variables that must be downloaded.

    This deliberately excludes derived-variable computation, so the returned
    dataset's dask graph covers only remote zarr reads and subsetting (level,
    time, and spatial bounding box). Derived variables should be computed by
    ``generate_era5_derived_data`` from the materialized output of this function,
    keeping remote I/O and derivation in separate dask graphs.

    Args:
        era5_config: Configuration object containing all necessary parameters for
            data generation.

    Returns:
        Lazy dataset of download variables, subset and chunked per
        ``era5_config.chunks``.

    Raises:
        ValueError: If any requested variable is not in ARCO and has no registered
            derivation function.
    """
    ds = _open_era5_zarr_store(era5_config)

    direct_vars, derived_vars = derived.classify_variables(era5_config.variables, {str(k) for k in ds.data_vars})
    download_vars = derived.resolve_download_variables(era5_config.variables, direct_vars, derived_vars)

    date_range = pd.date_range(era5_config.time_start, era5_config.time_end, freq=era5_config.time_resolution)

    # Attach periodic index to longitude for > 360 to avoid wrap-crossing slice issues
    ds_download = utils.attach_periodic_lon_index(ds[download_vars])
    ds_subset = ds_download.sel(
        time=date_range,
        latitude=slice(era5_config.coordinates.lat_max, era5_config.coordinates.lat_min),
        longitude=slice(era5_config.coordinates.lon_min, era5_config.coordinates.lon_max),
        level=era5_config.pressure_levels,
    )

    return ds_subset.chunk(era5_config.chunks)


def generate_era5_derived_data(era5_config: config.ERA5DataLoaderConfig, source_ds: xr.Dataset) -> xr.Dataset:
    """Compute requested derived variables from their required inputs.

    Args:
        era5_config: Configuration whose ``variables`` determine which derived
            variables to compute.
        source_ds: Dataset containing at least the inputs required by each
            derived variable, e.g. the output of ``generate_era5_download_data``
            or a dataset read back from a local icechunk store.

    Returns:
        Dataset containing only the derived variables requested in
        ``era5_config.variables``, with the same dask-or-eager backing as
        ``source_ds``.
    """
    _, derived_vars = derived.classify_variables(era5_config.variables, set(source_ds.data_vars))

    ds_derived = xr.Dataset()
    for var_name in derived_vars:
        spec = derived.DERIVED_VARIABLE_REGISTRY[var_name]
        logger.info(f"Computing derived variable '{var_name}' from {spec.required_inputs}")
        ds_derived[var_name] = spec.compute(*[source_ds[inp] for inp in spec.required_inputs])

    return ds_derived


def generate_era5_data(era5_config: config.ERA5DataLoaderConfig) -> xr.Dataset:
    """Generate a subset of ERA5 data for model training, validation, and testing.

    Variables are split into those available directly in the ARCO store and those
    that must be derived. Direct variables are downloaded first; derived variables
    are then computed from their required inputs. Intermediate inputs not in the
    original variable list are dropped before returning.

    Args:
        era5_config: Configuration object containing all necessary parameters for
            data generation.

    Returns:
        An xarray Dataset containing the subset of ERA5 data specified by the config.

    Raises:
        ValueError: If any requested variable is not in ARCO and has no registered
            derivation function.
    """
    ds_download = generate_era5_download_data(era5_config)
    ds_derived = generate_era5_derived_data(era5_config, ds_download)
    ds = ds_download.assign(**{name: ds_derived[name] for name in ds_derived.data_vars})

    requested_set = set(era5_config.variables)
    vars_to_drop = [v for v in ds.data_vars if v not in requested_set]
    if vars_to_drop:
        ds = ds.drop_vars(vars_to_drop)

    return ds.chunk(era5_config.chunks)


def get_local_icechunk_repository(
    storage_config: config.IcechunkStorageConfig,
) -> tuple[ic.Repository, str | None]:
    """Open or create an icechunk repository for storing processed data.

    Args:
        storage_config: Configuration object containing parameters for icechunk storage.

    Returns:
        An icechunk Repository object ready for use.
    """
    storage = ic.local_filesystem_storage(storage_config.store_path)

    if ic.Repository.exists(storage):
        append = "time"
        logger.info(f"Opening existing icechunk repository at {storage_config.store_path}")
    else:
        append = None
        logger.info(f"Creating new icechunk repository at {storage_config.store_path}")
    repo = ic.Repository.open_or_create(storage)
    return repo, append


def inspect_store(storage_config: config.IcechunkStorageConfig) -> StoreContents | None:
    """Return a snapshot of the variables, times, levels, and spatial extent in the store.

    Args:
        storage_config: Configuration for the icechunk store.

    Returns:
        StoreContents if the store exists, None otherwise.
    """
    storage = ic.local_filesystem_storage(storage_config.store_path)
    if not ic.Repository.exists(storage):
        return None
    ds = utils.open_readonly_icechunk_store(
        storage_config.store_path,
        storage_config.branch_name,
        group=storage_config.group_name,
        zarr_format=storage_config.zarr_format,
    )
    if not ds.data_vars:
        return None
    ds = utils.unwrap_longitude(ds)
    return StoreContents(
        variables=[str(k) for k in ds.data_vars],
        times=pd.DatetimeIndex(ds.time.values),
        levels=list(map(int, ds.level.values)),
        coordinates=utils.BoundingBox(
            lat_min=float(ds.latitude.min()),
            lat_max=float(ds.latitude.max()),
            lon_min=float(ds.longitude.min()),
            lon_max=float(ds.longitude.max()),
        ),
    )


def determine_write_strategy(
    era5_config: config.ERA5DataLoaderConfig,
    store_contents: StoreContents | None,
) -> WriteStrategy:
    """Determine what needs to be written or appended to the icechunk store.

    Args:
        era5_config: Configuration describing what data is requested.
        store_contents: Current state of the store, or None if the store does not exist.

    Returns:
        WriteStrategy describing what action to take.
    """
    requested_times = pd.date_range(era5_config.time_start, era5_config.time_end, freq=era5_config.time_resolution)
    missing_variables: list[str] = []
    missing_times: pd.DatetimeIndex = pd.DatetimeIndex([])
    skip_reason = ""
    error_reason = ""
    merge_required = False

    if store_contents is None:
        missing_variables = list(era5_config.variables)
        missing_times = requested_times
    elif list(era5_config.pressure_levels) != store_contents.levels or (
        era5_config.coordinates != store_contents.coordinates
    ):
        error_reason = (
            "Non-time dimension mismatch between config and store. "
            "Delete the store and rerun to regenerate with the new configuration."
        )
    else:
        missing_variables = [v for v in era5_config.variables if v not in store_contents.variables]
        missing_times = requested_times[~requested_times.isin(store_contents.times)]

        if len(missing_variables) == 0 and len(missing_times) == 0:
            skip_reason = "All requested variables and time steps are already present in the store."
        if len(missing_times) > 0:
            if len(store_contents.times) == 0:
                error_reason = "Store exists but contains no time steps. Delete the store and rerun."
            elif missing_times[0] <= store_contents.times[-1]:
                merge_required = True

    return WriteStrategy(
        missing_variables=missing_variables,
        missing_times=missing_times,
        skip_reason=skip_reason,
        error_reason=error_reason,
        merge_required=merge_required,
    )


def write_new_variables_to_icechunk_store(
    storage_config: config.IcechunkStorageConfig,
    ds: xr.Dataset | xr.DataArray,
) -> None:
    """Write new variables into an existing icechunk store.

    Opens the existing store to validate that the incoming dataset's dimensions
    are identical, then writes a minimal Dataset containing only the new
    variable arrays (no coordinate arrays) so existing dimension coordinates
    are never re-written or extended.

    Args:
        storage_config: Configuration for the icechunk store.
        ds: Dataset containing only the new variables to add.

    Raises:
        ValueError: If the dimensions or shape of ``ds`` do not exactly match
            those already present in the store.
    """
    storage = ic.local_filesystem_storage(storage_config.store_path)
    repo = ic.Repository.open(storage)

    existing_ds = utils.open_readonly_icechunk_store(
        storage_config.store_path,
        storage_config.branch_name,
        group=storage_config.group_name,
        zarr_format=storage_config.zarr_format,
    )
    if dict(ds.sizes) != dict(existing_ds.sizes):
        raise ValueError(f"Dimension mismatch: incoming {dict(ds.sizes)} != store {dict(existing_ds.sizes)}")

    logger.info(
        f"Adding new variables {list(ds.data_vars)} to icechunk store at "
        f"{storage_config.store_path} with dataset of shape {ds.sizes}"
    )

    new_ds = xr.Dataset(
        {
            name: xr.DataArray(ds[name].drop_encoding().data, dims=list(ds[name].dims), attrs=ds[name].attrs)
            for name in ds.data_vars
        }
    )
    session = repo.writable_session(storage_config.branch_name)
    icechunk.xarray.to_icechunk(new_ds, session, mode="a", safe_chunks=False)
    log = f"{storage_config.commit_message} - added variables: {list(ds.data_vars)}"
    session.commit(log)
    logger.info(log)


_WRITE_MEMORY_FRACTION = 0.5
_FALLBACK_MEMORY_BYTES = 32 * 1024**3  # 32GB, used when no slurm_config is available


def _compute_time_chunks(
    times: np.ndarray,
    ds_for_sizing: xr.Dataset | xr.DataArray,
    memory_bytes: int,
) -> list[np.ndarray]:
    """Split ``times`` into write-sized chunks based on ``ds_for_sizing``'s footprint.

    Args:
        times: Time values to split into chunks, in order.
        ds_for_sizing: Dataset whose per-variable per-timestep byte footprint
            determines the chunk size, via ``_compute_time_chunk_size``.
        memory_bytes: Memory available per dask thread for materializing one
            variable's chunk.

    Returns:
        ``times`` split into consecutive, non-overlapping chunks of at most
        ``_compute_time_chunk_size(ds_for_sizing, memory_bytes)`` time steps.
    """
    time_chunk_size = _compute_time_chunk_size(ds_for_sizing, memory_bytes)
    return [times[i : i + time_chunk_size] for i in range(0, len(times), time_chunk_size)]


def _materialize_time_chunks(
    ds: xr.Dataset | xr.DataArray,
    time_chunks: list[np.ndarray],
    global_attrs: dict,
) -> Iterator[tuple[int, np.ndarray, xr.Dataset | xr.DataArray]]:
    """Materialize ``ds`` one time chunk at a time, reporting progress via tqdm.

    Args:
        ds: Lazy dataset to select from and compute, one time chunk at a time.
        time_chunks: Time-value chunks to iterate over, in order.
        global_attrs: Attributes to assign to each materialized chunk.

    Yields:
        Tuples of (chunk index, time values for this chunk, materialized chunk).
    """
    with tqdm.tqdm(time_chunks, unit="chunk", desc="Writing icechunk") as pbar:
        for i, time_slice in enumerate(pbar):
            ds_slice = ds.sel(time=time_slice).compute()
            ds_slice.attrs = global_attrs
            pbar.set_postfix(start=str(time_slice[0])[:10], steps=f"{len(time_slice)} steps")
            yield i, time_slice, ds_slice


def _compute_time_chunk_size(ds: xr.Dataset | xr.DataArray, memory_bytes: int) -> int:
    """Determine how many time steps can be safely materialized in one write chunk.

    The chunk size is bounded by the largest single variable's per-timestep byte
    footprint, since ``.compute()`` on a distributed dask client finalizes
    (concatenates) each variable's chunks independently, and several of these
    finalize tasks can run concurrently on the threads of a single worker.

    Args:
        ds: Dataset to be written; used to compute the per-variable per-timestep
            byte footprint via ``nbytes`` (metadata-only, does not trigger computation).
        memory_bytes: Memory available per dask thread for materializing one
            variable's chunk.

    Returns:
        Number of time steps per write chunk, at least 1 and at most ``ds.sizes["time"]``.
    """
    n_times = ds.sizes["time"]
    arrays = ds.data_vars.values() if isinstance(ds, xr.Dataset) else [ds]
    max_bytes_per_step = max(array.nbytes / n_times for array in arrays)
    chunk_size = max(1, int(memory_bytes * _WRITE_MEMORY_FRACTION // max_bytes_per_step))
    return min(chunk_size, n_times)


def write_or_append_icechunk_store(
    storage_config: config.IcechunkStorageConfig,
    ds: xr.Dataset | xr.DataArray,
    dry_run: bool = False,
    time_chunks: list[np.ndarray] | None = None,
) -> None:
    """Write or append an xarray Dataset to an icechunk store.

    If the store already exists, the new data will be appended along the time dimension.

    Args:
        storage_config: Configuration object containing parameters for icechunk storage.
        ds: The xarray Dataset to write or append to the store.
        dry_run: If True, perform a dry run without actually committing to the store.
        time_chunks: Time-value chunks to materialize and write one at a time, in order.
            If None (the default), ``ds`` is written directly in a single lazy
            ``to_icechunk`` call, letting icechunk's internal I/O drive the dask graph
            instead of eagerly computing each chunk. Pass explicit chunks to instead
            materialize ``ds`` in memory-bounded pieces (e.g. for computed data).
    """
    # Drop encoding due to zarr v3 compatibility issues; see
    # https://github.com/pydata/xarray/issues/10032
    ds = ds.drop_encoding()
    global_attrs = ds.attrs

    repo, append = get_local_icechunk_repository(storage_config)

    logger.info(
        f"{'Appending to' if append else 'Writing new'} icechunk store at "
        f"{storage_config.store_path} with variables {list(ds.data_vars)} and shape {ds.sizes}"
    )

    n_times = ds.sizes["time"]
    session = repo.writable_session(storage_config.branch_name)
    if time_chunks is None:
        icechunk.xarray.to_icechunk(ds, session, append_dim=append, safe_chunks=False, group=storage_config.group_name)
    else:
        for _, _, ds_slice in _materialize_time_chunks(ds, time_chunks, global_attrs):
            icechunk.xarray.to_icechunk(
                ds_slice, session, append_dim=append, safe_chunks=False, group=storage_config.group_name
            )
            append = "time"  # always append after first write

    if dry_run:
        logger.info(f"Dry run: would commit {n_times} time steps")
        session.discard_changes()
    else:
        session.commit(f"{storage_config.commit_message}, time steps 0 to {n_times}")


def _non_input_data_variables(storage_config: config.IcechunkStorageConfig, required_inputs: set[str]) -> list[str]:
    """List existing store data variables that are not required derived-variable inputs.

    Excluding these when reading the store back avoids cross-variable
    dimension-size conflicts that arise mid-pipeline, when some variables have
    been extended to a new time range and others have not yet.

    Args:
        storage_config: Configuration for the icechunk store to inspect.
        required_inputs: Variable names needed to compute the requested derived
            variables; these are kept regardless of their current dimensions.

    Returns:
        Names of data variables present in the store but not in
        ``required_inputs``.
    """
    storage = ic.local_filesystem_storage(storage_config.store_path)
    repo = ic.Repository.open(storage)
    session = repo.readonly_session(storage_config.branch_name)
    group = zarr.open_group(
        store=session.store,
        path=storage_config.group_name or "",
        mode="r",
        zarr_format=storage_config.zarr_format,
    )

    def _dims(name: str) -> list[str]:
        array = group[name]
        dimension_names = getattr(array.metadata, "dimension_names", None)
        if dimension_names is not None:
            return list(dimension_names)
        return list(array.attrs.get("_ARRAY_DIMENSIONS", []))

    return [name for name in group.array_keys() if _dims(name) != [name] and name not in required_inputs]


def write_variables_to_icechunk_store(
    storage_config: config.IcechunkStorageConfig,
    ds: xr.Dataset,
    time_chunks: list[np.ndarray],
    dry_run: bool = False,
) -> None:
    """Write one or more variables into an existing icechunk store, chunked along time.

    Variables not yet present in the store are created at the size of the first time
    chunk only (independent of other variables' larger time extent), without writing
    any coordinate arrays. Every chunk, including the first, is then appended along the
    time dimension for variables already present in the store. Positional alignment
    with the store's existing arrays requires ``time_chunks`` to match the chunk
    boundaries used to write those arrays (e.g. via ``write_or_append_icechunk_store``).

    Args:
        storage_config: Configuration for the icechunk store.
        ds: Dataset containing the variables to write, possibly a mix of variables
            already present in the store and new variables.
        time_chunks: Time-value chunks to write, in order, aligned with the store's
            existing arrays.
        dry_run: If True, perform a dry run without actually committing to the store.
    """
    ds = ds.drop_encoding()
    global_attrs = ds.attrs

    storage = ic.local_filesystem_storage(storage_config.store_path)
    repo = ic.Repository.open(storage)

    # Variables in the store may currently have mismatched time extents (e.g. a
    # download variable just extended by write_or_append_icechunk_store, but a
    # derived variable from a previous write_variables_to_icechunk_store call still
    # at its old length), so list arrays directly via zarr rather than
    # utils.open_readonly_icechunk_store, which builds an xarray Dataset and requires
    # all variables sharing a dimension to have consistent sizes.
    readonly_session = repo.readonly_session(storage_config.branch_name)
    existing_group = zarr.open_group(
        store=readonly_session.store,
        path=storage_config.group_name or "",
        mode="r",
        zarr_format=storage_config.zarr_format,
    )
    existing_array_names = set(existing_group.array_keys())
    new_vars = [name for name in ds.data_vars if name not in existing_array_names]

    logger.info(
        f"Writing variables {list(ds.data_vars)} (new: {new_vars}) to icechunk store at {storage_config.store_path}"
    )

    n_times = sum(len(time_slice) for time_slice in time_chunks)
    session = repo.writable_session(storage_config.branch_name)
    for i, _, ds_slice in _materialize_time_chunks(ds, time_chunks, global_attrs):
        # Write only bare data arrays (no coordinate variables): the store's coordinate
        # arrays are already fully populated by the corresponding write_or_append_icechunk_store
        # call, so re-writing or re-appending them here would desynchronize their length
        # from these variables', which are appended one chunk at a time.
        bare_ds = xr.Dataset(
            {name: xr.DataArray(da.data, dims=list(da.dims), attrs=da.attrs) for name, da in ds_slice.data_vars.items()}
        )
        if i == 0 and new_vars:
            icechunk.xarray.to_icechunk(
                bare_ds[new_vars], session, mode="a", safe_chunks=False, group=storage_config.group_name
            )
            existing_vars = [name for name in ds_slice.data_vars if name not in new_vars]
            if existing_vars:
                icechunk.xarray.to_icechunk(
                    bare_ds[existing_vars],
                    session,
                    append_dim="time",
                    safe_chunks=False,
                    group=storage_config.group_name,
                )
        else:
            icechunk.xarray.to_icechunk(
                bare_ds, session, append_dim="time", safe_chunks=False, group=storage_config.group_name
            )

    if dry_run:
        logger.info(f"Dry run: would commit {n_times} time steps")
        session.discard_changes()
    else:
        session.commit(f"{storage_config.commit_message}, time steps 0 to {n_times}")


def create_dask_client(
    slurm_config: config.SlurmConfig,
    scheduler_options: dict | None = None,
):
    """Create a Dask client backed by a SLURM cluster for parallel derivation.

    Args:
        slurm_config: SLURM cluster parameters (partition, account, cores, etc.).
        scheduler_options: Optional overrides for the Dask scheduler (e.g. dashboard
            port, network interface). Defaults to ``{"dashboard_address": ":13921",
            "interface": "ib0"}``.

    Returns:
        A Dask Client object connected to the created cluster.
    """
    import dask_jobqueue  # type: ignore
    from dask.distributed import Client

    if scheduler_options is None:
        scheduler_options = {"dashboard_address": ":13921", "interface": "ib0"}

    slurm_cluster = dask_jobqueue.SLURMCluster(
        queue=slurm_config.queue,
        cores=slurm_config.cores,
        processes=slurm_config.processes,
        memory=slurm_config.memory,
        walltime=slurm_config.walltime,
        shebang="#!/usr/bin/bash",
        job_extra_directives=[
            f"--output={slurm_config.stdout}",
            f"--error={slurm_config.stderr}",
        ],
        scheduler_options=scheduler_options,
    )
    slurm_cluster.scale(jobs=slurm_config.n_jobs)
    client = Client(slurm_cluster)
    return client


def main():
    """Entry point for generating ERA5 icechunk data from config."""
    parser = argparse.ArgumentParser(description="Generate ERA5 data and store in icechunk")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML data generation config")
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Launch a dask-jobqueue SLURM cluster for parallel derivation (requires slurm_config in YAML)",
    )
    parser.add_argument(
        "--zarr-async-concurrency",
        type=int,
        default=None,
        help="Override zarr's async.concurrency (max in-flight chunk requests per store). "
        "Default is zarr's built-in default (10).",
    )
    args = parser.parse_args()

    if args.zarr_async_concurrency is not None:
        zarr.config.set({"async.concurrency": args.zarr_async_concurrency})
        logger.info(f"Set zarr async.concurrency to {args.zarr_async_concurrency}")

    bounding_box_type_hook = {utils.BoundingBox: lambda d: utils.BoundingBox(*d)}

    era5_config = utils.open_config_yaml_as_dataclass(
        args.config,
        config.ERA5DataLoaderConfig,
        config_key="era5_config",
        type_hooks=bounding_box_type_hook,
    )
    logger.info(f"ERA5 data config loaded: {era5_config}")

    icechunk_config = utils.open_config_yaml_as_dataclass(
        args.config, config.IcechunkStorageConfig, config_key="icechunk_storage_config"
    )
    logger.info(f"Icechunk storage config loaded: {icechunk_config}")

    slurm_config: config.SlurmConfig | None = None
    if args.slurm:
        slurm_config = utils.open_config_yaml_as_dataclass(args.config, config.SlurmConfig, config_key="slurm_config")
        logger.info(f"SLURM config loaded: {slurm_config}")
        client = create_dask_client(slurm_config)
        logger.info(f"Dask client started: {client}")

    store_contents = inspect_store(icechunk_config)
    strategy = determine_write_strategy(era5_config, store_contents)

    if strategy.error_reason:
        raise ValueError(strategy.error_reason)
    if strategy.skip_reason:
        logger.info(strategy.skip_reason)
        return

    logger.info(f"Variables to add:      {strategy.missing_variables}")
    logger.info(f"Time steps to add:     {len(strategy.missing_times)}")
    strategy.execute(era5_config, icechunk_config, slurm_config)
    logger.info("Data generation and storage complete.")


if __name__ == "__main__":
    main()
