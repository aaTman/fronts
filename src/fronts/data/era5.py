import dataclasses
import datetime
import logging
from typing import Sequence

import xarray as xr

from fronts.utils import calc, constants, data_utils

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


def subset_variables(
    ds: xr.Dataset,
    variables: Sequence[str],
    levels: Sequence[int],
) -> xr.Dataset:
    """Subset ERA5 variables from *ds* at the requested levels.

    Collects the appropriate variable names (mapping surface counterparts via
    :data:`SURFACE_VARIABLE_MAP` and checking :data:`SURFACE_ONLY_VARIABLES`)
    and returns a single ``ds[var_names].sel(level=pressure_levels)`` slice.

    No stacking or concatenation along ``level`` is performed here; use
    :func:`maybe_stack_variables` afterwards to unify the level dimension.

    Args:
        ds: An xarray Dataset already subsetted spatially and temporally.
        variables: Canonical variable names to include.
        levels: Ordered list of levels.  May include ``1013`` and/or integer
            hPa values.
    """
    include_surface = 1013 in levels
    pressure_levels = [lv for lv in levels if lv != 1013]

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
    result = ds[var_names_subset]
    return result


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
    """
    if 1013 not in levels:
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


def stack_variables(
    ds: xr.Dataset,
    variables: Sequence[str],
    levels: Sequence[int],
) -> xr.Dataset:
    """Subset and stack ERA5 variables into a unified Dataset.

    Convenience wrapper that calls :func:`subset_variables` followed by
    :func:`maybe_stack_variables`.
    """
    subsetted = subset_variables(ds, variables, levels)
    return maybe_stack_variables(subsetted, variables, levels)


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
    """Configuration for loading and stacking ERA5 predictor variables.

    Variables are specified as a single ``variables`` list using canonical
    (pressure-level) names.  The ``levels`` list controls which vertical levels
    are loaded and may contain the string ``1013`` in addition to integer
    hPa values.

    When ``1013`` appears in ``levels``, the module-level
    :data:`SURFACE_VARIABLE_MAP` is consulted to find each variable's surface
    counterpart (e.g. ``"temperature"`` → ``"2m_temperature"``).  Variables
    listed in :data:`SURFACE_ONLY_VARIABLES` (e.g. ``"mean_sea_level_pressure"``)
    are included with ``level=[1013]`` automatically.

    The resulting xarray Dataset has a unified ``"level"`` coordinate whose
    values are a mix of the string ``1013`` and integer hPa values,
    following the convention used throughout the codebase.

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
            or ``[1000, 900, 750]``.
        years: Years to select data from. Typically injected by DataConfig.build()
            via dataclasses.replace() rather than set directly in YAML.
        store: URI of the zarr store to open.
        chunks: Chunk sizes for lazy loading, e.g. {"time": 48}.
        consolidated: Whether to use consolidated zarr metadata.
    """

    domain_extent: list[float]
    variables: list[str]
    levels: list[int]
    store: str
    consolidated: bool
    years: list[int]

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

        # 1. Variable subset
        log.debug(
            "Subsetting variables=%s at levels=%s...", self.variables, self.levels
        )
        ds = subset_variables(ds, variables=self.variables, levels=self.levels)

        # 2. Spatiotemporal subset
        log.info(
            "Applying spatiotemporal subset: domain_extent=%s, years=%s",
            self.domain_extent,
            self.years,
        )
        bbox = data_utils.convert_domain_extent_to_bounding_box(self.domain_extent)
        lon_min = data_utils.maybe_convert_lon(bbox.lon_min, ds.longitude)
        lon_max = data_utils.maybe_convert_lon(bbox.lon_max, ds.longitude)
        ds = ds.sel(
            time=str(self.years),
            latitude=slice(bbox.lat_max, bbox.lat_min),
            longitude=slice(lon_min, lon_max),
            level=self.levels,
        )
        log.debug(
            "Spatiotemporal subset done. lat shape=%s, lon shape=%s, years=%s",
            ds.latitude.shape,
            ds.longitude.shape,
            self.years,
        )

        # 3. Derive if needed
        ds = maybe_derive_variables(ds, self.variables)

        log.info(
            "ERA5PredictorConfig.build() complete. Output vars: %s",
            list(ds.data_vars),
        )
        return ds


def subset_arco_era5(
    ds: xr.Dataset,
    variables: list[str],
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    bounding_box: data_utils.BoundingBox,
    levels: list[int],
):
    """Subsets the ARCO ERA5 dataset by variables, specific time range, and geographic bounding box.

    Args:
        ds: The input xarray Dataset containing the ARCO ERA5 data.
        variables: A list of variable names to subset from the dataset.
        start_date: The start date of the time range to subset (inclusive).
        end_date: The end date of the time range to subset (inclusive).
        bounding_box: A BoundingBox named tuple defining the geographic bounding box
            for subsetting. Defaults to a bounding box covering the contiguous United
            States.
        levels: A list of pressure levels to subset from the dataset.
    """
    variables_to_postprocess = [var for var in variables if var not in ds.data_vars]
    if variables_to_postprocess:
        variables = [var for var in variables if var not in variables_to_postprocess]

    unknown = [
        n
        for n in variables_to_postprocess
        if n not in calc.derived_variable_callable_mapping
    ]
    if unknown:
        raise ValueError(
            f"Variables {unknown} not found in dataset and no "
            "post-processing functions available for them."
        )
    subset_ds = ds[variables].sel(
        latitude=slice(bounding_box.lat_max, bounding_box.lat_min),
        longitude=slice(bounding_box.lon_min, bounding_box.lon_max),
        time=slice(start_date, end_date),
        level=levels,
    )
    return subset_ds
