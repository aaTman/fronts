import argparse
import dataclasses
import logging
import sys

import icechunk as ic
import icechunk.xarray
import pandas as pd
import xarray as xr
import zarr
import zarr.errors

from fronts import utils
from fronts.data import config, derived, sources

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
        levels: Pressure levels stored. Empty when the store holds only
            single-level variables.
        coordinates: Spatial bounding box of stored data.
    """

    variables: list[str]
    times: pd.DatetimeIndex
    levels: list[int]
    coordinates: utils.BoundingBox


@dataclasses.dataclass
class WriteStrategy:
    """Decision record produced by determine_write_strategy.

    Exactly one of skip_reason, error_reason, or an action field will be populated:
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
    ) -> None:
        """Generate and write the ERA5 data described by this strategy.

        Runs in two phases when possible: the raw (downloaded) variables are
        written and committed first, then derived variables are computed from
        the local store and committed separately. When appending time steps to
        a store that already contains derived variables, raw and derived data
        for the new times are computed together and appended in a single commit
        so all arrays stay aligned along the time dimension.

        Args:
            era5_config: Base ERA5 configuration; time range or variables will be
                narrowed to only what is missing before generating data.
            icechunk_config: Configuration for the target icechunk store.
        """
        store_contents = inspect_store(icechunk_config)

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

            _, derived_vars = derived.classify_variables(era5_config.variables, set(ds_download.data_vars))
            stored_derived = (
                [v for v in derived_vars if v in store_contents.variables] if store_contents is not None else []
            )

            if stored_derived:
                logger.info("Appending raw and derived variables together to keep time axis aligned...")
                derived_config = dataclasses.replace(era5_config, variables=derived_vars)
                ds_derived = generate_era5_derived_data(derived_config, ds_download)
                write_or_append_icechunk_store(icechunk_config, xr.merge([ds_download, ds_derived]))
            else:
                logger.info("Phase 1: writing raw ERA5 variables to icechunk store...")
                write_or_append_icechunk_store(icechunk_config, ds_download)
                if derived_vars:
                    logger.info("Phase 2: computing derived variables from the local store...")
                    write_derived_variables(era5_config, icechunk_config, derived_vars)
            store_contents = inspect_store(icechunk_config)

        if self.missing_variables and store_contents is not None:
            still_missing = [v for v in self.missing_variables if v not in store_contents.variables]
            if still_missing:
                missing_pressure = [v for v in still_missing if v not in era5_config.single_level_variables]
                missing_single = [v for v in still_missing if v in era5_config.single_level_variables]
                direct_vars, derived_vars = _classify_missing(era5_config, missing_pressure)

                if direct_vars or missing_single:
                    logger.info(f"Downloading missing variables {direct_vars + missing_single}...")
                    narrow_config = dataclasses.replace(
                        era5_config, variables=direct_vars, single_level_variables=missing_single
                    )
                    ds_download = generate_era5_download_data(narrow_config).sel(time=store_contents.times)
                    write_new_variables_to_icechunk_store(icechunk_config, ds_download)
                if derived_vars:
                    logger.info(f"Computing missing derived variables {derived_vars}...")
                    write_derived_variables(era5_config, icechunk_config, derived_vars)


def _classify_missing(
    era5_config: config.ERA5DataLoaderConfig, missing_pressure: list[str]
) -> tuple[list[str], list[str]]:
    """Split missing pressure-level variables into directly downloadable and derived."""
    if era5_config.era5_uri.startswith(sources.ARRAYLAKE_URI_PREFIX):
        available = set(sources.GOOGLE_TO_ARRAYLAKE_PRESSURE)
    else:
        ds = _open_source_dataset(era5_config)
        available = {str(k) for k in ds.data_vars}
    return derived.classify_variables(missing_pressure, available)


def _open_source_dataset(era5_config: config.ERA5DataLoaderConfig) -> xr.Dataset:
    """Open a non-arraylake zarr source lazily without dask."""
    open_kwargs: dict = {"chunks": None}
    if era5_config.storage_options:
        open_kwargs["storage_options"] = era5_config.storage_options
    return xr.open_zarr(era5_config.era5_uri, **open_kwargs)


def generate_era5_download_data(era5_config: config.ERA5DataLoaderConfig) -> xr.Dataset:
    """Open the ERA5 source and subset to the variables that must be downloaded.

    This deliberately excludes derived-variable computation: the returned dataset
    contains all directly available variables plus the inputs required to derive
    the rest, subset by time, pressure level, and spatial bounding box. Variables
    and coordinates follow Google ARCO naming regardless of the source.

    Args:
        era5_config: Configuration object containing all necessary parameters for
            data generation.

    Returns:
        Lazy dataset of download variables, chunked per ``era5_config.chunks``
        when provided.

    Raises:
        ValueError: If any requested variable is unavailable in the source and has
            no registered derivation function.
    """
    if era5_config.era5_uri.startswith(sources.ARRAYLAKE_URI_PREFIX):
        available = set(sources.GOOGLE_TO_ARRAYLAKE_PRESSURE)
        direct_vars, derived_vars = derived.classify_variables(era5_config.variables, available)
        download_vars = derived.resolve_download_variables(era5_config.variables, direct_vars, derived_vars)
        ds = sources.open_arraylake_era5(era5_config.era5_uri, download_vars, era5_config.single_level_variables)
    else:
        ds = _open_source_dataset(era5_config)
        available = {str(k) for k in ds.data_vars}
        direct_vars, derived_vars = derived.classify_variables(era5_config.variables, available)
        download_vars = derived.resolve_download_variables(era5_config.variables, direct_vars, derived_vars)
        ds = ds[download_vars + era5_config.single_level_variables]

    date_range = pd.date_range(era5_config.time_start, era5_config.time_end, freq=era5_config.time_resolution)

    # Attach periodic index to longitude for > 360 to avoid wrap-crossing slice issues
    ds = utils.attach_periodic_lon_index(ds)
    selection: dict = {
        "time": date_range,
        "latitude": slice(era5_config.coordinates.lat_max, era5_config.coordinates.lat_min),
        "longitude": slice(era5_config.coordinates.lon_min, era5_config.coordinates.lon_max),
    }
    if "level" in ds.dims:
        selection["level"] = era5_config.pressure_levels
    ds_subset = ds.sel(**selection)

    if era5_config.chunks:
        return ds_subset.chunk(era5_config.chunks)
    return ds_subset


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
    """Generate the full requested ERA5 dataset, combining downloaded and derived variables.

    Args:
        era5_config: Configuration object containing all necessary parameters for
            data generation.

    Returns:
        Lazy dataset containing exactly the requested pressure-level and
        single-level variables.
    """
    ds_download = generate_era5_download_data(era5_config)
    ds_derived = generate_era5_derived_data(era5_config, ds_download)
    ds = ds_download.assign(**{str(name): ds_derived[name] for name in ds_derived.data_vars})
    requested = set(era5_config.variables) | set(era5_config.single_level_variables)
    return ds.drop_vars([v for v in ds.data_vars if v not in requested])


def write_derived_variables(
    era5_config: config.ERA5DataLoaderConfig,
    icechunk_config: config.IcechunkStorageConfig,
    derived_vars: list[str],
) -> None:
    """Compute derived variables from the local store and write them as new variables.

    Args:
        era5_config: Configuration providing chunking for the dask computation.
        icechunk_config: Configuration for the icechunk store to read from and write to.
        derived_vars: Names of the derived variables to compute and write.
    """
    required_inputs = sorted(
        {inp for v in derived_vars for inp in derived.DERIVED_VARIABLE_REGISTRY[v].required_inputs}
    )
    local_ds = utils.open_readonly_icechunk_store(
        icechunk_config.store_path,
        icechunk_config.branch_name,
        group=icechunk_config.group_name,
        zarr_format=icechunk_config.zarr_format,
    )[required_inputs]
    local_ds = local_ds.chunk(era5_config.chunks or "auto")
    derived_config = dataclasses.replace(era5_config, variables=list(derived_vars))
    ds_derived = generate_era5_derived_data(derived_config, local_ds)
    write_new_variables_to_icechunk_store(icechunk_config, ds_derived)


def inspect_store(storage_config: config.IcechunkStorageConfig) -> StoreContents | None:
    """Return a snapshot of the variables, times, levels, and spatial extent in the store.

    Args:
        storage_config: Configuration for the icechunk store.

    Returns:
        StoreContents if the store and its configured group exist, None otherwise.
    """
    storage = ic.local_filesystem_storage(storage_config.store_path)
    if not ic.Repository.exists(storage):
        return None
    try:
        ds = utils.open_readonly_icechunk_store(
            storage_config.store_path,
            storage_config.branch_name,
            group=storage_config.group_name,
            zarr_format=storage_config.zarr_format,
        )
    except (FileNotFoundError, KeyError, zarr.errors.GroupNotFoundError):
        return None
    if not ds.data_vars:
        return None
    ds = utils.unwrap_longitude(ds)
    return StoreContents(
        variables=[str(k) for k in ds.data_vars],
        times=pd.DatetimeIndex(ds.time.values),
        levels=list(map(int, ds.level.values)) if "level" in ds.coords else [],
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
    requested_variables = list(era5_config.variables) + list(era5_config.single_level_variables)
    missing_variables: list[str] = []
    missing_times: pd.DatetimeIndex = pd.DatetimeIndex([])
    skip_reason = ""
    error_reason = ""
    merge_required = False

    if store_contents is None:
        missing_variables = requested_variables
        missing_times = requested_times
    elif (store_contents.levels and list(era5_config.pressure_levels) != store_contents.levels) or (
        era5_config.coordinates != store_contents.coordinates
    ):
        error_reason = (
            "Non-time dimension mismatch between config and store. "
            "Delete the store and rerun to regenerate with the new configuration."
        )
    else:
        missing_variables = [v for v in requested_variables if v not in store_contents.variables]
        missing_times = requested_times[~requested_times.isin(store_contents.times)]

        if len(missing_variables) > 0 and len(missing_times) > 0:
            error_reason = (
                "Store has both missing variables and missing time steps. "
                "Add the missing time steps first, then rerun to add new variables."
            )
        elif len(missing_variables) == 0 and len(missing_times) == 0:
            skip_reason = "All requested variables and time steps are already present in the store."
        elif len(missing_times) > 0:
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
    incoming_sizes = dict(ds.sizes)
    existing_sizes = {dim: size for dim, size in existing_ds.sizes.items() if dim in incoming_sizes}
    if incoming_sizes != existing_sizes:
        raise ValueError(f"Dimension mismatch: incoming {incoming_sizes} != store {existing_sizes}")

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
    icechunk.xarray.to_icechunk(new_ds, session, group=storage_config.group_name, mode="a", safe_chunks=False)
    log = f"{storage_config.commit_message} - added variables: {list(ds.data_vars)}"
    session.commit(log)
    logger.info(log)


def write_or_append_icechunk_store(
    storage_config: config.IcechunkStorageConfig, ds: xr.Dataset | xr.DataArray, dry_run: bool = False
) -> None:
    """Write or append an xarray Dataset to an icechunk store in a single commit.

    If the configured group already contains data, the new data is appended along
    the time dimension; otherwise the group is created. The whole dataset is
    written in one ``to_icechunk`` call and committed once.

    Args:
        storage_config: Configuration object containing parameters for icechunk storage.
        ds: The xarray Dataset to write or append to the store.
        dry_run: If True, perform the write without committing.
    """
    # Drop encoding due to zarr v3 compatibility issues; see
    # https://github.com/pydata/xarray/issues/10032
    ds = ds.drop_encoding()

    existing = inspect_store(storage_config)
    append = "time" if existing is not None else None

    storage = ic.local_filesystem_storage(storage_config.store_path)
    repo = ic.Repository.open_or_create(storage)

    logger.info(
        f"{'Appending to' if append else 'Writing new'} icechunk store at "
        f"{storage_config.store_path} (group={storage_config.group_name}) "
        f"with variables {list(ds.data_vars)} and shape {ds.sizes}"
    )

    session = repo.writable_session(storage_config.branch_name)
    icechunk.xarray.to_icechunk(ds, session, group=storage_config.group_name, append_dim=append, safe_chunks=False)
    if dry_run:
        logger.info(f"Dry run: would commit {ds.sizes.get('time', 0)} time steps")
        return
    session.commit(f"{storage_config.commit_message}, {ds.sizes.get('time', 0)} time steps")


def create_dask_client(slurm_config: config.SlurmConfig, scheduler_options: dict | None = None):
    """Create a Dask client backed by a SLURM cluster.

    Args:
        slurm_config: SLURM cluster parameters from the YAML config.
        scheduler_options: Optional dask scheduler options; defaults to a fixed
            dashboard port on the infiniband interface.

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
    return Client(slurm_cluster)


def main():
    """Entry point for generating ERA5 icechunk data from config."""
    parser = argparse.ArgumentParser(description="Generate ERA5 data and store in icechunk")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML data generation config")
    parser.add_argument("--slurm", action="store_true", help="Launch a dask-jobqueue SLURM cluster for derivation")
    parser.add_argument(
        "--zarr-async-concurrency",
        type=int,
        default=None,
        help="Override the zarr_async_concurrency value from the YAML config",
    )
    args = parser.parse_args()

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

    concurrency = args.zarr_async_concurrency or era5_config.zarr_async_concurrency
    zarr.config.set({"async.concurrency": concurrency})
    logger.info(f"Set zarr async.concurrency to {concurrency}")

    client = None
    if args.slurm:
        slurm_config = utils.open_config_yaml_as_dataclass(args.config, config.SlurmConfig, config_key="slurm_config")
        logger.info(f"SLURM config loaded: {slurm_config}")
        client = create_dask_client(slurm_config)
        logger.info(f"Dask client started: {client.dashboard_link}")

    store_contents = inspect_store(icechunk_config)
    strategy = determine_write_strategy(era5_config, store_contents)

    if strategy.error_reason:
        raise ValueError(strategy.error_reason)
    if strategy.skip_reason:
        logger.info(strategy.skip_reason)
        return

    logger.info(f"Variables to add:      {strategy.missing_variables}")
    logger.info(f"Time steps to add:     {len(strategy.missing_times)}")
    strategy.execute(era5_config, icechunk_config)
    if client is not None:
        client.close()
    logger.info("Data generation and storage complete.")


if __name__ == "__main__":
    main()
