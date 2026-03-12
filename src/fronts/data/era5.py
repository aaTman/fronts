import xarray as xr

import datetime
import dataclasses
from fronts.utils import calc, data_utils, constants
from typing import Callable
import logging

log = logging.getLogger("fronts.data.era5")

ARCO_ERA5_GCP_URI = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)


DERIVED_VARIABLE_REGISTRY: dict[str, callable] = {
    "dewpoint": calc.dewpoint,
    "virtual_temperature": calc.virtual_temperature,
    "relative_humidity": calc.relative_humidity,
    "equivalent_potential_temperature": calc.theta_e,
    "geopotential_height": calc.geopotential_height,
}

SURFACE_VARIABLE_MAP: dict[str, str] = {
    "temperature": "2m_temperature",
    "u_component_of_wind": "10m_u_component_of_wind",
    "v_component_of_wind": "10m_v_component_of_wind",
    "dewpoint_temperature": "2m_dewpoint_temperature",
    "specific_humidity": "surface_specific_humidity",
}

# Variables that exist only at the surface (no pressure-level equivalent)
# These are included in the output whenever "surface" is in the levels list
SURFACE_ONLY_VARIABLES: set[str] = {
    "mean_sea_level_pressure",
    "total_precipitation",
    "sea_surface_temperature",
    "skin_temperature",
    "10m_wind_speed",
}


def stack_variables(
    ds: xr.Dataset,
    variables: list[str],
    levels: list[str | int],
) -> xr.Dataset:
    """Stacks ERA5 variables into a unified Dataset with a mixed level coordinate.

    The ``levels`` list may contain the string ``"surface"`` and/or integer hPa
    values (e.g. ``["surface", 1000, 950, 900, 850]``).  The function handles
    three categories of variable automatically:

    * **Pressure-level-only** — the variable exists only on pressure levels in
      the zarr store (e.g. ``"specific_humidity"``).  These are selected at the
      requested integer levels.
    * **Mixed surface + pressure** — the variable has both a surface counterpart
      (looked up via :data:`SURFACE_VARIABLE_MAP`) and pressure-level data.
      When ``"surface"`` is in ``levels`` the surface array is prepended; the
      result has a level coordinate of the form ``["surface", 1000, 950, ...]``.
    * **Surface-only** — the variable name appears in :data:`SURFACE_ONLY_VARIABLES`
      *or* is not found as a pressure-level variable in the store.  It is
      included with ``level=["surface"]`` whenever ``"surface"`` is in ``levels``.

    Args:
        ds: An xarray Dataset already subsetted spatially and temporally.
        variables: Canonical variable names to include.  Use pressure-level names
            (e.g. ``"temperature"``) for mixed/pressure variables; use the full
            surface name (e.g. ``"mean_sea_level_pressure"``) for surface-only ones.
        levels: Ordered list of levels to select.  May include the string
            ``"surface"`` and/or integer hPa values.

    Returns an xarray Dataset with a unified ``"level"`` coordinate whose values
    are a mix of the string ``"surface"`` and integer hPa values.
    """
    include_surface = "surface" in levels
    pressure_levels = [lv for lv in levels if lv != "surface"]

    result_datasets: list[xr.Dataset] = []

    for var in variables:
        surface_var_name = SURFACE_VARIABLE_MAP.get(var)
        is_surface_only = var in SURFACE_ONLY_VARIABLES

        if is_surface_only:
            # Surface-only variable: always has level=["surface"]
            if include_surface:
                da_sfc = ds[var].expand_dims({"level": ["surface"]})
                result_datasets.append(da_sfc.to_dataset(name=var))
        elif surface_var_name is not None:
            # Mixed variable: has a surface counterpart + pressure levels
            if pressure_levels:
                da_pl = ds[var].sel(level=pressure_levels)
            else:
                da_pl = None

            if include_surface and surface_var_name in ds:
                da_sfc = ds[surface_var_name].expand_dims({"level": ["surface"]})
                if da_pl is not None:
                    da = xr.concat([da_sfc, da_pl], dim="level")
                else:
                    da = da_sfc
            else:
                if da_pl is not None:
                    da = da_pl
                else:
                    continue  # nothing to add

            result_datasets.append(da.to_dataset(name=var))
        else:
            # Pressure-level-only variable
            if pressure_levels:
                da_pl = ds[var].sel(level=pressure_levels)
                result_datasets.append(da_pl.to_dataset(name=var))

    return xr.merge(result_datasets, join="outer")


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
            coordinate containing ``"surface"`` and/or integer hPa values.
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
        prefix = _ARCO_TO_LEGACY_NORM_KEY.get(var)
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
class ERA5PredictorConfig:
    """Configuration for loading and stacking ERA5 predictor variables.

    Variables are specified as a single ``variables`` list using canonical
    (pressure-level) names.  The ``levels`` list controls which vertical levels
    are loaded and may contain the string ``"surface"`` in addition to integer
    hPa values.

    When ``"surface"`` appears in ``levels``, the module-level
    :data:`SURFACE_VARIABLE_MAP` is consulted to find each variable's surface
    counterpart (e.g. ``"temperature"`` → ``"2m_temperature"``).  Variables
    listed in :data:`SURFACE_ONLY_VARIABLES` (e.g. ``"mean_sea_level_pressure"``)
    are included with ``level=["surface"]`` automatically.

    The resulting xarray Dataset has a unified ``"level"`` coordinate whose
    values are a mix of the string ``"surface"`` and integer hPa values,
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
        levels: Ordered list of levels to include.  May contain ``"surface"``
            and/or integer hPa values, e.g. ``["surface", 1000, 950, 900, 850]``
            or ``[1000, 900, 750]``.
        years: Years to select data from. Typically injected by DataConfig.build()
            via dataclasses.replace() rather than set directly in YAML.
        store: URI of the zarr store to open.
        chunks: Chunk sizes for lazy loading, e.g. {"time": 48}.
        consolidated: Whether to use consolidated zarr metadata.
    """

    domain_extent: list[float]
    variables: list[str]
    levels: list[str | int]
    store: str
    chunks: dict[str, int]
    consolidated: bool
    years: list[int] 
    
    def build(self) -> xr.Dataset:
        """Loads and stacks ERA5 data into a unified xarray Dataset.

        Returns an xarray Dataset with a ``"level"`` coordinate that includes
        ``"surface"`` (for surface variables) and integer hPa values (for
        pressure-level variables).  Time is filtered to ``self.years``.

        Variables listed in :data:`DERIVED_VARIABLE_REGISTRY` are computed
        automatically after stacking the raw variables.  Derived variables
        are processed in the order they appear in ``self.variables``, so
        dependencies must come first (e.g. ``"dewpoint"`` before
        ``"virtual_temperature"``).
        """
        log.info("ERA5PredictorConfig.build() — opening zarr store: %s", self.store)
        ds = xr.open_zarr(
            store=self.store,
            chunks=self.chunks,
            consolidated=self.consolidated,
        )
        log.debug("Zarr store opened. Variables available: %s", list(ds.data_vars))

        # Spatial subset
        log.debug("Applying spatial subset: domain_extent=%s", self.domain_extent)
        bbox = data_utils.convert_domain_extent_to_bounding_box(self.domain_extent)
        lon_min = data_utils.maybe_convert_lon(bbox.lon_min, ds.latitude)
        lon_max = data_utils.maybe_convert_lon(bbox.lon_max, ds.latitude)
        ds = ds.sel(
            latitude=slice(bbox.lat_max, bbox.lat_min),
            longitude=slice(lon_min, lon_max),
        )
        log.debug(
            "Spatial subset done. lat shape=%s, lon shape=%s",
            ds.latitude.shape,
            ds.longitude.shape,
        )

        # Temporal subset: keep only the requested years
        log.debug("Applying temporal subset for years=%s...", self.years)
        ds = ds.isel(time=ds.time.dt.year.isin(self.years))
        log.info(
            "ERA5 temporal subset done. %d timesteps selected.", ds.sizes.get("time", 0)
        )

        # Partition variables: raw ones are loaded from the store, derived
        # ones are computed after stacking.
        raw_vars = [v for v in self.variables if v not in DERIVED_VARIABLE_REGISTRY]
        to_derive = [v for v in self.variables if v in DERIVED_VARIABLE_REGISTRY]

        log.debug("Stacking raw variables=%s at levels=%s...", raw_vars, self.levels)
        result = stack_variables(
            ds,
            variables=raw_vars,
            levels=self.levels,
        )

        if to_derive:
            log.info("Deriving variables: %s", to_derive)
            result = derive_era5_variables(result, to_derive)

        log.info(
            "ERA5PredictorConfig.build() complete. Output vars: %s",
            list(result.data_vars),
        )
        return result


def load_arco_era5(
    store: str = ARCO_ERA5_GCP_URI,
    chunks: dict[str, int] = {"time": 48},
    consolidated: bool = True,
):
    """Opens the Google ARCO ERA5 analysis-ready dataset as an xarray Dataset.

    Args:
        store: The URI of the zarr store to open. Defaults to the Google ARCO ERA5
            analysis-ready dataset link.
        chunks: The chunk sizes to use when opening the dataset. Defaults to chunking
            the time dimension into 48-hour chunks.
        consolidated: Whether to use consolidated metadata when opening the dataset.
            Defaults to True.

    Returns an xarray Dataset containing the ERA5 analysis-ready data.
    """
    era5_ds = xr.open_zarr(
        store=store,
        chunks=chunks,
        consolidated=consolidated,
    )

    return era5_ds


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
        variables.pop(variables_to_postprocess)

    if not any(
        [n for n in variables_to_postprocess if n in calc.callable_mapping.keys()]
    ):
        raise ValueError(
            f"Variables {variables_to_postprocess} not found in dataset and no "
            "post-processing functions available for them."
        )
    ds = ds[variables]
    ds = ds.sel(
        latitude=slice(bounding_box.lat_max, bounding_box.lat_min),
        longitude=slice(bounding_box.lon_min, bounding_box.lon_max),
    )
    ds = ds.sel(time=slice(start_date, end_date))
    ds = ds.sel(level=levels)
    return ds


def derive_era5_variables(ds: xr.Dataset, derived_variables: list[str]) -> xr.Dataset:
    """Compute derived meteorological variables and add them to the dataset.

    Variables are computed in the order given.  Dependencies must appear
    earlier in the list (e.g. "dewpoint" before "virtual_temperature").

    Args:
        ds: Stacked xr.Dataset with raw ERA5 variables.
        derived_variables: Ordered list of derived variable names.  Valid
            names are keys of DERIVED_VARIABLE_REGISTRY.
    """
    for name in derived_variables:
        fn = DERIVED_VARIABLE_REGISTRY.get(name)
        if fn is None:
            raise ValueError(
                f"Unknown derived variable {name!r}. "
                f"Valid options: {list(DERIVED_VARIABLE_REGISTRY.keys())}"
            )
        log.debug("Deriving %s...", name)
        ds = fn(ds)
    return ds


@dataclasses.dataclass
class ERA5TrainingDataConfig:
    """A dataclass for generating data from the ARCO ERA5 dataset.

    This class provides methods for loading and subsetting the variables, spatial
        bounds, and time of the ARCO ERA5 dataset.

    Attributes:
        domain_extent: A list of four floats representing the geographic domain extent
            in the format [lon_min, lon_max, lat_min, lat_max].
        variables: A list of variable names to subset from the dataset.
        start_date: The start date of the time range to subset (inclusive).
        end_date: The end date of the time range to subset (inclusive).
        store: The URI of the zarr store to open.
        chunks: The chunk sizes to use when opening the dataset.
        consolidated: Whether to use consolidated metadata when opening the dataset.
    """

    domain_extent: list[float]
    variables: list[str]
    start_date: datetime.datetime
    end_date: datetime.datetime
    levels: list[int]
    store: str
    chunks: dict[str, int]
    consolidated: bool

    def build(self) -> xr.Dataset:
        """Builds the training dataset by loading and subsetting the ARCO ERA5 dataset.

        Returns an xarray Dataset containing the subset ARCO ERA5 data.
        """
        # Load the ARCO ERA5 dataset with default params
        ds = load_arco_era5(
            store=self.store, chunks=self.chunks, consolidated=self.consolidated
        )

        # Subset the dataset by variables, time range, and geographic bounding box
        ds = subset_arco_era5(
            ds,
            variables=self.variables,
            start_date=self.start_date,
            end_date=self.end_date,
            bounding_box=data_utils.convert_domain_extent_to_bounding_box(
                self.domain_extent
            ),
            levels=self.levels,
        )
        return ds


def _default_postprocess(ds: xr.Dataset):
    """Default postprocessor that passes through data unmodified."""
    return ds


def maybe_postprocess_era5(
    ds: xr.Dataset, postprocess_func: Callable = _default_postprocess, **kwargs
) -> xr.Dataset:
    """Applies any necessary post-processing steps to the ERA5 dataset.

    This function is a placeholder for any future post-processing steps that may be
    required for the ERA5 dataset. Currently, it returns the dataset unchanged.

    Args:
        ds: The input xarray Dataset containing the ERA5 data.
        postprocess_func: A callable function that takes an xarray Dataset as input.
            Defaults to a no-op function that returns the dataset unchanged.
        **kwargs: Additional keyword arguments to pass to the post-processing function.

    Returns the possibly post-processed Dataset.
    """
    ds = postprocess_func(ds, **kwargs)
    return ds


def dewpoint_postprocessor(ds: xr.Dataset):
    ds["dewpoint"] = calc.dewpoint_from_specific_humidity(
        ds.level, ds.specific_humidity
    )
    return ds


def potential_temperature_postprocessor(ds: xr.Dataset):
    ds["potential_temperature"] = calc.potential_temperature(ds.level, ds.temperature)
    return ds


def equivalent_potential_temperature_postprocessor(ds: xr.Dataset):
    ds["equivalent_potential_temperature"] = calc.equivalent_potential_temperature(
        ds.level, ds.temperature, ds.dewpoint
    )
    return ds


def virtual_potential_temperature_postprocessor(ds: xr.Dataset):
    ds["virtual_potential_temperature"] = calc.virtual_potential_temperature(
        ds.level, ds.temperature, ds.dewpoint
    )
    return ds


def wet_bulb_temperature_postprocessor(ds: xr.Dataset):
    ds["wet_bulb_temperature"] = calc.wet_bulb_temperature(ds.temperature, ds.dewpoint)
    return ds


def wet_bulb_potential_temperature_postprocessor(ds: xr.Dataset):
    ds["wet_bulb_potential_temperature"] = calc.wet_bulb_potential_temperature(
        ds.level, ds.temperature, ds.dewpoint
    )
    return ds


def relative_humidity_postprocessor(ds: xr.Dataset):
    ds["relative_humidity"] = calc.relative_humidity_from_dewpoint(
        ds.temperature, ds.dewpoint
    )
    return ds


callable_mapping = {
    "dewpoint": dewpoint_postprocessor,
    "potential_temperature": potential_temperature_postprocessor,
    "equivalent_potential_temperature": equivalent_potential_temperature_postprocessor,
    "virtual_potential_temperature": virtual_potential_temperature_postprocessor,
    "wet_bulb_temperature": wet_bulb_temperature_postprocessor,
    "wet_bulb_potential_temperature": wet_bulb_potential_temperature_postprocessor,
    "relative_humidity": relative_humidity_postprocessor,
}
