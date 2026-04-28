import dataclasses
import datetime
import logging
from typing import Sequence, Literal, Callable
import icechunk
import xarray as xr
import zarr
from fronts.utils import calc, constants, data_utils
from fronts.utils.calc import derived_variable_callable_mapping

log = logging.getLogger("fronts.data.era5")

ARCO_ERA5_GCP_URI = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)

SURFACE_VARIABLE_MAP: dict[str, str] = {
    "temperature": "2m_temperature",
    "u_component_of_wind": "10m_u_component_of_wind",
    "v_component_of_wind": "10m_v_component_of_wind",
    "dewpoint_temperature": "2m_dewpoint_temperature",
    "specific_humidity": "surface_specific_humidity",
}

# Variables that exist only at the surface (no pressure-level equivalent)
# These are included in the output whenever 1013 is in the levels list
SURFACE_ONLY_VARIABLES: set[str] = {
    "mean_sea_level_pressure",
    "total_precipitation",
    "sea_surface_temperature",
    "skin_temperature",
    "10m_wind_speed",
}
# Helper function to check if a level is a surface level
def _surface_level(level: int | str) -> bool:
    return level == 1013 or level == "surface"

def subset_variables(
    ds: xr.Dataset,
    variables: Sequence[str],
    levels: Sequence[int],
) -> list[str]:
    """Subset a list of variable names from *ds* at the requested levels.

    Collects the appropriate variable names (mapping surface counterparts via
    :data:`SURFACE_VARIABLE_MAP` and checking :data:`SURFACE_ONLY_VARIABLES`)

    Args:
        ds: An xarray Dataset already subsetted spatially and temporally.
        variables: Canonical variable names to include.
        levels: Ordered list of levels.  May include ``1013``, "surface", and/or 
            integer hPa values.
    
    Returns a list of variable names that are subset from *ds*.
    """

    include_surface = any(_surface_level(lv) for lv in levels)
    pressure_levels = [lv for lv in levels if not _surface_level(lv)]

    var_names: list[str] = []
    for var in variables:
        surface_var_name = SURFACE_VARIABLE_MAP.get(var)
        is_surface_only = var in SURFACE_ONLY_VARIABLES

        if is_surface_only:
            if include_surface:
                var_names.append(var)
        elif surface_var_name is not None:
            if pressure_levels:
                var_names.append(var)
            if include_surface and surface_var_name in ds:
                var_names.append(surface_var_name)
        else:
            if pressure_levels:
                var_names.append(var)

    log.info("variables: %s", var_names)
    ds_var_list = list(ds.data_vars)
    var_names_subset = [variable for variable in var_names if variable in ds_var_list]
    return var_names_subset


def maybe_derive_variables(ds: xr.Dataset, variables: Sequence[str]) -> xr.Dataset:
    """Compute derived meteorological variables and add them to the dataset.

    Variables are computed in the order given.  Dependencies must appear
    earlier in the list (e.g. "dewpoint" before "virtual_temperature").

    Args:
        ds: Stacked xr.Dataset with raw ERA5 variables.
        variables: Ordered list of variable names. Valid names are keys of
            DERIVED_VARIABLE_REGISTRY.
    """
    for name in variables:
        fn = calc.derived_variable_callable_mapping.get(name, None)
        if fn:
            log.info("Deriving %s...", name)
            ds[name] = fn(ds)
    return ds


def maybe_stack_variables(
    ds: xr.Dataset,
    variables: Sequence[str],
    levels: Sequence[int],
) -> xr.Dataset:
    """Stack surface and pressure-level arrays into a unified ``level`` dim.

    Only performed when ``1013`` is in *levels*.  Surface arrays are given
    ``level=[1013]`` and concatenated with their pressure-level counterparts
    so every variable shares a single ``level`` coordinate.

    If ``1013`` is **not** in *levels* the dataset is returned unchanged.

    Args:
        ds: Dataset returned by :func:`subset_variables`.
        variables: The same canonical variable list passed to
            :func:`subset_variables`.
        levels: Ordered list of levels (may include ``1013``).
    
    Returns a Dataset with the variables stacked into a unified "level" dimension.
    """
    if not any(_surface_level(lv) for lv in levels):
        return ds

    result: dict[str, xr.DataArray] = {}
    for var in variables:
        surface_var_name = SURFACE_VARIABLE_MAP.get(var)
        is_surface_only = var in SURFACE_ONLY_VARIABLES

        if is_surface_only:
            if var in ds:
                result[var] = ds[var].expand_dims({"level": [1013]})
        elif surface_var_name is not None:
            pieces: list[xr.DataArray] = []
            if surface_var_name in ds:
                pieces.append(ds[surface_var_name].expand_dims({"level": [1013]}))
            if var in ds and "level" in ds[var].dims:
                pieces.append(ds[var])
            if pieces:
                result[var] = (
                    xr.concat(pieces, dim="level") if len(pieces) > 1 else pieces[0]
                )
        else:
            if var in ds:
                result[var] = ds[var]

    return xr.Dataset(result)


# Maps ARCO variable names to the legacy short-name prefixes used as keys
# in constants.NORMALIZATION_PARAMS (e.g. "temperature" → "T", so the lookup
# key becomes "T_850" or "T_surface").
# TODO: align all variable names within repo
_ARCO_TO_LEGACY_NORM_KEY: dict[str, str] = {
    "temperature": "T",
    "dewpoint": "Td",
    "virtual_temperature": "Tv",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
    "relative_humidity": "RH",
    "geopotential_height": "sp_z",
    "equivalent_potential_temperature": "theta_e",
    "mean_sea_level_pressure": "mslp_z",
    "geopotential": "sp_z",
}


def normalize_legacy_arco_era5(
    ds: xr.Dataset,
    method: str = "min-max",
    params: dict | None = None,
) -> xr.Dataset:
    """Normalize the stacked ARCO ERA5 Dataset using legacy normalization constants.

    For each variable and level, looks up the normalization parameters using
    the legacy naming convention (e.g. ``"T_850"``, ``"q_surface"``) and applies
    the requested normalization method.

    Args:
        ds: Stacked xr.Dataset with ARCO variable names and a ``"level"``
            coordinate containing ``1013`` and/or integer hPa values.
        method: ``"min-max"``, ``"standard"``, or ``"standard_weighted"``.
        params: Normalization parameter dict. Defaults to
            ``constants.NORMALIZATION_PARAMS``.

    Returns a copy of the dataset with each variable-level slice normalized.
    """
    if params is None:
        params = constants.NORMALIZATION_PARAMS

    # Map method name to the two param-list indices: (subtract, divide-by)
    idx = {"min-max": (0, 1), "standard": (2, 3), "standard_weighted": (4, 5)}
    if method not in idx:
        raise ValueError(f"Unknown normalization method {method!r}")
    i_a, i_b = idx[method]

    # Build each normalized variable via pure xarray arithmetic
    # In-place mutation (ds[var].loc[...] = ...) raises on dask-backed arrays
    # because dask graphs are immutable.  Instead, accumulate into a dict and
    # return a new Dataset
    result: dict = {}
    for var in list(ds.data_vars):
        prefix = _ARCO_TO_LEGACY_NORM_KEY.get(str(var))
        if prefix is None:
            log.warning("No normalization mapping for variable %r — skipping.", var)
            result[var] = ds[var]
            continue

        da = ds[var]
        vals_a: list[float] = []
        vals_b: list[float] = []
        for lv in da.level.values:
            key = f"{prefix}_{lv}"
            if key in params:
                vals_a.append(float(params[key][i_a]))
                vals_b.append(float(params[key][i_b]))
            else:
                # Level has no normalization entry (e.g. surface-only variable at
                # a pressure level — the data is NaN there anyway after outer-join
                # merging).  Use identity (subtract 0, divide by 1) to preserve NaN.
                log.warning("Normalization key %r not found — using identity.", key)
                vals_a.append(0.0)
                vals_b.append(1.0)

        p_a = xr.DataArray(vals_a, dims="level", coords={"level": da.level})
        p_b = xr.DataArray(vals_b, dims="level", coords={"level": da.level})

        if method == "min-max":
            result[var] = (da - p_a) / (p_b - p_a)
        else:  # standard or standard_weighted: (x - mean) / std
            result[var] = (da - p_a) / p_b

    return xr.Dataset(result)


@dataclasses.dataclass
class ERA5Config:
    """Configuration to load ERA5 from the ARCO ERA5 store (ideally).

    Attributes:
        domain_extent: [lon_min, lon_max, lat_min, lat_max] geographic extent.
        variables: Variable names to include.  May contain both raw ERA5 names
            (e.g. ``"temperature"``, ``"mean_sea_level_pressure"``) and derived
            variable names registered in :data:`DERIVED_VARIABLE_REGISTRY`
            (e.g. ``"dewpoint"``, ``"virtual_temperature"``).  Raw variables are
            loaded from the zarr store; derived variables are computed after
            stacking.  Order matters for derived variables — dependencies must
            appear first (e.g. ``"dewpoint"`` before ``"virtual_temperature"``).
        levels: Ordered list of levels to include.  May contain ``1013``
            and/or integer hPa values, e.g. ``[1013, 1000, 950, 900, 850]``
            or ``[1000, 900, 750]``. 1013 refers to surface variables in the
            scenario that stacking them into the pressure-level variables is
            desired.
        store: URI of the zarr store to open.
        chunks: Chunk sizes for lazy loading, e.g. {"time": 48}.
        consolidated: Whether to use consolidated zarr metadata.
        years: The years to include in the dataset.
        local_zarr_path: The path to the local zarr store.
    """

    domain_extent: list[float]
    variables: list[str]
    levels: list[int]
    store: str
    consolidated: bool
    years: list[str] | None
    local_zarr_path: str

    def build(self) -> xr.Dataset:
        """Loads and stacks ERA5 data into a unified xarray Dataset.

        Returns an xarray Dataset with a ``"level"`` coordinate that includes
        ``1013`` (for surface variables) and integer hPa values (for
        pressure-level variables).  Time is filtered to ``self.years``.

        Variables listed in :data:`DERIVED_VARIABLE_REGISTRY` are computed
        elsewhere after stacking the raw variables.  Derived variables
        are processed in the order they appear in ``self.variables``, so
        dependencies must come first (e.g. ``"dewpoint"`` before
        ``"virtual_temperature"``).
        """
        log.info("ERA5PredictorConfig.build() — opening zarr store: %s", self.store)
        ds = xr.open_dataset(
            self.store,
            chunks=None,
            engine="zarr",
            backend_kwargs={"storage_options": {"anon": True}},
        )
        log.debug("Zarr store opened. Variables available: %s", list(ds.data_vars))

        # Partition into raw (in zarr store) and derived (computed after stacking)
        raw_vars = [v for v in self.variables if v not in derived_variable_callable_mapping]
        to_derive = [v for v in self.variables if v in derived_variable_callable_mapping]

        # 1. Variable subset — use canonical raw names only
        log.debug(
            "Subsetting variables=%s at levels=%s...", raw_vars, self.levels
        )
        subset_variables_list = subset_variables(ds, variables=raw_vars, levels=self.levels)
        subset_variables_ds = ds[subset_variables_list]

        # 2. Spatiotemporal subset
        log.info(
            "Applying spatiotemporal subset: domain_extent=%s, years=%s",
            self.domain_extent,
            self.years,
        )
        bbox = data_utils.convert_domain_extent_to_bounding_box(self.domain_extent)
        lon_min = data_utils.maybe_convert_lon(bbox.lon_min, subset_variables_ds.longitude)
        lon_max = data_utils.maybe_convert_lon(bbox.lon_max, subset_variables_ds.longitude)

        # Drop surface levels from the levels list; already loaded if provided via subset_variables
        non_surface_levels = [lv for lv in self.levels if not _surface_level(lv)]
        subset_variables_ds = subset_variables_ds.sel(
            time=self.years,
            latitude=slice(bbox.lat_max, bbox.lat_min),
            longitude=slice(lon_min, lon_max),
            level=non_surface_levels,
        )

        # 3. Stack surface and pressure levels — pass canonical raw names
        log.debug("Stacking raw variables=%s...", raw_vars)
        subset_stacked_variables_ds = maybe_stack_variables(
            subset_variables_ds, variables=raw_vars, levels=self.levels
        )

        # 4. Compute derived variables (e.g. dewpoint, virtual_temperature)
        if to_derive:
            log.info("Deriving variables: %s", to_derive)
            subset_stacked_variables_ds = maybe_derive_variables(
                subset_stacked_variables_ds, to_derive
            )

        log.debug(
            "Spatiotemporal subset done. lat shape=%s, lon shape=%s, years=%s",
            subset_stacked_variables_ds.latitude.shape,
            subset_stacked_variables_ds.longitude.shape,
            self.years,
        )

        log.info(
            "ERA5PredictorConfig.build() complete. Output vars: %s",
            list(subset_stacked_variables_ds.data_vars),
        )
        return subset_stacked_variables_ds


def create_icechunk_session(icechunk_path: str):
    # Initialize local storage
    local_storage = icechunk.local_filesystem_storage(icechunk_path)
    # Build the RepositoryConfig with default config settings
    config = icechunk.RepositoryConfig.default()

    # Create icechunk repository and session using "main" branch
    repo = icechunk.Repository.create(local_storage, config)
    session = repo.writable_session("main")

    return session


@dataclasses.dataclass
class ERA5IcechunkConfig:
    """Configuration for processing ARCO ERA5 to an icechunk store."""

    era5_config: ERA5Config
    repo_name: str
    group: Literal["raw", "derived", "raw_and_derived_normalized"]
    start_date: datetime.datetime
    end_date: datetime.datetime

    def generate(self):
        """Process and generate the icechunk store and group."""
        self.era5 = self.era5_config.build()
        self.generate_icechunk_store(repo_name=self.repo_name, group=self.group)

    def arco_era5_to_raw_group(self, session: icechunk.Session) -> icechunk.Session:
        store = session.store
        group = zarr.group(store=store, path="raw", overwrite=True)
        return group

    def raw_group_to_derived_group(self, session: icechunk.Session) -> icechunk.Session:
        pass

    def raw_and_derived_to_normalized_group(
        self, session: icechunk.Session
    ) -> icechunk.Session:
        pass

    def generate_icechunk_store(
        self,
        repo_name: str,
        group: Literal["raw", "derived", "raw_and_derived_normalized"],
    ):
        group_callable_mapping: dict[str, Callable] = {
            "raw": self.arco_era5_to_raw_group,
            "derived": self.raw_group_to_derived_group,
            "raw_and_derived_normalized": self.raw_and_derived_to_normalized_group,
        }

        session = create_icechunk_session(self.repo_name)
        fork_session = session.fork()
        group_callable_mapping[group](fork_session)
