"""Data configuration dataclasses for the FrontFinder training pipeline.

This module provides composable config dataclasses for loading ERA5 predictor data
and front label (truth) data, splitting by year, stacking surface and pressure-level
variables in xarray, and building tf.data.Dataset objects for train/val/test splits.

Follows the same pattern as the rest of the codebase: typed dataclasses with .build()
methods that return runtime objects, loadable from YAML via dacite.
"""

import dataclasses
import datetime
import glob as glob_module
from typing import Any, Optional, Union

import xarray as xr

from fronts.data.batch import BatchGeneratorConfig, create_dataloader
from fronts.data.era5 import convert_domain_extent_to_bounding_box
from fronts.utils import data_utils


# ---------------------------------------------------------------------------
# Constant mapping: pressure-level variable name -> surface variable name
#
# When "surface" appears in the levels list, the code looks up each requested
# variable name here to find its surface counterpart in the zarr store.
# Variables absent from this map are pressure-level-only (no surface analogue).
# ---------------------------------------------------------------------------

SURFACE_VARIABLE_MAP: dict[str, str] = {
    "temperature": "2m_temperature",
    "u_component_of_wind": "10m_u_component_of_wind",
    "v_component_of_wind": "10m_v_component_of_wind",
    "dewpoint_temperature": "2m_dewpoint_temperature",
    "specific_humidity": "surface_specific_humidity",
}

# Variables that exist only at the surface (no pressure-level equivalent).
# These are included in the output whenever "surface" is in the levels list.
SURFACE_ONLY_VARIABLES: set[str] = {
    "mean_sea_level_pressure",
    "total_precipitation",
    "sea_surface_temperature",
    "skin_temperature",
    "10m_wind_speed",
}


def _stack_era5_variables(
    ds: xr.Dataset,
    variables: list[str],
    levels: list[Union[str, int]],
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
        variables: Canonical variable names to load.  For variables with both
            surface and pressure-level representations, use the pressure-level
            name (e.g. ``"temperature"``); for surface-only variables, use the
            store name directly (e.g. ``"mean_sea_level_pressure"``).
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
    levels: list[Union[str, int]]
    store: str
    chunks: dict[str, int]
    consolidated: bool
    years: list[int] = dataclasses.field(default_factory=list)

    def build(self) -> xr.Dataset:
        """Loads and stacks ERA5 data into a unified xarray Dataset.

        Returns an xarray Dataset with a ``"level"`` coordinate that includes
        ``"surface"`` (for surface variables) and integer hPa values (for
        pressure-level variables).  Time is filtered to ``self.years``.
        """
        ds = xr.open_zarr(
            store=self.store,
            chunks=self.chunks,
            consolidated=self.consolidated,
        )

        # Spatial subset
        bbox = convert_domain_extent_to_bounding_box(self.domain_extent)
        ds = ds.sel(
            latitude=slice(bbox.lat_max, bbox.lat_min),
            longitude=slice(bbox.lon_min, bbox.lon_max),
        )

        # Temporal subset: keep only the requested years
        ds = ds.isel(time=ds.time.dt.year.isin(self.years))

        return _stack_era5_variables(
            ds,
            variables=self.variables,
            levels=self.levels,
        )


@dataclasses.dataclass
class FrontsDataConfig:
    """Configuration for loading front label (truth) data from netCDF files.

    Front files are expected to contain an "identifier" variable with integer class
    values (0=no front, 1=CF, 2=WF, etc.), following the existing convention in the
    codebase.

    Attributes:
        directory: Path to directory containing per-timestep front netCDF files.
            Files are matched using the glob pattern `{directory}/*{year}*.nc`.
        years: Years to load. Typically injected by DataConfig.build() via
            dataclasses.replace() rather than set directly in YAML.
        front_types: Code(s) passed to reformat_fronts() to regroup front classes.
            Examples: "MERGED-ALL", "F_BIN", ["CF", "WF"]. None = no reformatting.
    """

    directory: str
    front_types: Optional[Any]  # str | list[str] | None
    years: list[int] = dataclasses.field(default_factory=list)

    def build(self) -> xr.Dataset:
        """Loads and optionally reformats front label data for the given years.

        Returns an xarray Dataset with the "identifier" variable, optionally
        reformatted according to self.front_types.
        """
        files = sorted(
            f
            for year in self.years
            for f in glob_module.glob(f"{self.directory}/*{year}*.nc")
        )
        ds = xr.open_mfdataset(files, engine="netcdf4", combine="by_coords")

        if self.front_types is not None:
            ds = data_utils.reformat_fronts(ds, self.front_types)

        return ds


@dataclasses.dataclass
class ModelData:
    """Runtime holder for train/validation/test tf.data.Dataset objects.

    Returned by DataConfig.build(). Trainer accesses .train_data and
    .validation_data directly (train.py:221,228).

    Attributes:
        train_data: tf.data.Dataset for training.
        validation_data: tf.data.Dataset for validation.
        test_data: Optional tf.data.Dataset for testing. None if test_years is empty.
    """

    train_data: Any
    validation_data: Any
    test_data: Optional[Any] = None


@dataclasses.dataclass
class DataConfig:
    """Top-level data configuration for the FrontFinder training pipeline.

    Composes ERA5 predictor and front label configs, handles train/val/test year
    splits, normalization, and shuffling, and builds the tf.data.Dataset objects
    consumed by Trainer.

    Year lists (train_years, val_years, test_years) are specified at this level and
    injected into ERA5PredictorConfig and FrontsDataConfig at build time using
    dataclasses.replace() — they should NOT be set in the era5/fronts YAML blocks.

    Attributes:
        era5: ERA5PredictorConfig defining the predictor variable source.
        fronts: FrontsDataConfig defining the front label source.
        train_years: Years to use for the training split.
        val_years: Years to use for the validation split.
        test_years: Years to use for the test split. May be empty [].
        batch: BatchGeneratorConfig defining spatial patch sizes and prefetch settings.
        shuffle: Whether to shuffle the training dataset. Defaults to True.
        normalization_method: One of "standard", "standard_weighted", "min-max".
            Defaults to "standard".
    """

    era5: ERA5PredictorConfig
    fronts: FrontsDataConfig
    train_years: list[int]
    val_years: list[int]
    test_years: list[int]
    batch: BatchGeneratorConfig
    shuffle: bool = True
    normalization_method: str = "standard"

    def build(self) -> ModelData:
        """Builds train, validation, and test tf.data.Dataset objects.

        For each split, clones the ERA5 and fronts configs with the appropriate years,
        builds the xarray datasets, normalizes the predictors, and wraps them in a
        tf.data.Dataset via create_dataloader.

        Returns a ModelData with train_data, validation_data, and optionally test_data.
        """

        def _build_split(years: list[int]) -> Optional[Any]:
            if not years:
                return None

            # Build ERA5 predictor dataset for this split
            era5_cfg = dataclasses.replace(self.era5, years=years)
            inputs_ds = era5_cfg.build()
            # TODO: normalize_dataset expects a "pressure_level" dimension and legacy
            # short variable-name keys (e.g. "T_850", "u_1000"). Our stacked dataset
            # uses dimension "level" and ARCO variable names ("temperature", etc.).
            # Normalization constants and the normalize_dataset function need to be
            # updated for the new naming scheme before this call can be re-enabled.
            # inputs_ds = data_utils.normalize_dataset(
            #     inputs_ds, method=self.normalization_method
            # )

            # Build fronts dataset for this split
            fronts_cfg = dataclasses.replace(self.fronts, years=years)
            targets_ds = fronts_cfg.build()

            # Build tf.data.Dataset via create_dataloader directly
            # (BatchGeneratorConfig.build() is not used because it lacks inputs/targets fields)
            tf_ds = create_dataloader(
                inputs=inputs_ds,
                targets=targets_ds,
                input_sizes=self.batch.input_sizes,
                target_sizes=self.batch.target_sizes,
                prefetch_number=self.batch.prefetch_number,
                preload_batch=self.batch.preload_batch,
            )
            return tf_ds

        train_ds = _build_split(self.train_years)
        if self.shuffle and train_ds is not None:
            train_ds = train_ds.shuffle(buffer_size=1000)

        val_ds = _build_split(self.val_years)
        test_ds = _build_split(self.test_years)

        return ModelData(
            train_data=train_ds,
            validation_data=val_ds,
            test_data=test_ds,
        )


@dataclasses.dataclass
class TimeSelection:
    """Specifies which ERA5 timesteps to load for prediction.

    Exactly one of most_recent, timestamps, or date_range must be set.
    Validation is performed in __post_init__.

    Attributes:
        most_recent: If True, selects the single latest timestep available in the
            store. The zarr store's time dimension is assumed to be sorted ascending
            (latest last), which is true for ARCO ERA5.
        timestamps: An explicit list of datetimes, each representing a single
            analysis time (date + hour). Selection uses method="nearest" to tolerate
            minor floating-point or timezone differences.
        date_range: A two-element list [start, end] of datetimes. All timesteps
            between start and end (inclusive) are selected.

    YAML usage — choose exactly one block:

        # Most recent timestep in the store:
        time_selection:
          most_recent: true

        # Explicit individual timesteps (use ISO 8601 with T separator):
        time_selection:
          timestamps:
            - "2024-06-01T12:00:00"
            - "2024-06-02T00:00:00"

        # Inclusive date range:
        time_selection:
          date_range:
            - "2024-06-01T00:00:00"
            - "2024-06-07T18:00:00"
    """

    most_recent: bool = False
    timestamps: Optional[list[datetime.datetime]] = None
    date_range: Optional[list[datetime.datetime]] = None  # exactly [start, end]

    def __post_init__(self):
        modes_set = sum([
            bool(self.most_recent),
            self.timestamps is not None,
            self.date_range is not None,
        ])
        if modes_set != 1:
            raise ValueError(
                "Exactly one of most_recent, timestamps, or date_range must be set "
                f"in TimeSelection (got {modes_set} modes set)."
            )
        if self.date_range is not None and len(self.date_range) != 2:
            raise ValueError(
                "date_range must be a list of exactly two datetimes [start, end], "
                f"got {len(self.date_range)} element(s)."
            )

    def apply(self, ds: xr.Dataset) -> xr.Dataset:
        """Applies this time selection to a spatially-subsetted xarray Dataset.

        Args:
            ds: An xarray Dataset that has already been subsetted spatially.

        Returns the Dataset subsetted to the specified timesteps.
        """
        if self.most_recent:
            return ds.isel(time=[-1])
        elif self.timestamps is not None:
            return ds.sel(time=self.timestamps, method="nearest")
        else:  # date_range
            return ds.sel(time=slice(self.date_range[0], self.date_range[1]))


@dataclasses.dataclass
class PredictConfig:
    """Configuration for ERA5-based model inference.

    Mirrors DataConfig for training but uses TimeSelection instead of year lists
    for temporal filtering, and returns a plain xr.Dataset rather than a
    tf.data.Dataset — inference runs one spatial domain at a time rather than
    batching many patches.

    Attributes:
        era5: ERA5PredictorConfig defining the variable source, spatial domain,
            and zarr store. The `years` field on era5 is unused by PredictConfig;
            time selection is fully controlled by time_selection.
        time_selection: TimeSelection specifying which timesteps to load. Exactly
            one of most_recent, timestamps, or date_range must be set.
        normalization_method: One of "standard", "standard_weighted", "min-max".
            Should match the normalization used during training. Defaults to
            "standard".
    """

    era5: ERA5PredictorConfig
    time_selection: TimeSelection
    normalization_method: str = "standard"

    def build(self) -> xr.Dataset:
        """Loads, stacks, and normalizes ERA5 data for the selected timesteps.

        Opens the zarr store lazily, applies spatial subsetting, time selection,
        surface/pressure variable stacking, and normalization.

        Returns a normalized xarray Dataset ready for model inference.
        """
        ds = xr.open_zarr(
            store=self.era5.store,
            chunks=self.era5.chunks,
            consolidated=self.era5.consolidated,
        )

        # Spatial subset
        bbox = convert_domain_extent_to_bounding_box(self.era5.domain_extent)
        ds = ds.sel(
            latitude=slice(bbox.lat_max, bbox.lat_min),
            longitude=slice(bbox.lon_min, bbox.lon_max),
        )

        # Time selection
        ds = self.time_selection.apply(ds)

        # Stack surface and pressure-level variables
        stacked = _stack_era5_variables(
            ds,
            variables=self.era5.variables,
            levels=self.era5.levels,
        )

        # TODO: normalize_dataset expects a "pressure_level" dimension and legacy
        # short variable-name keys (e.g. "T_850", "u_1000"). Our stacked dataset
        # uses dimension "level" and ARCO variable names ("temperature", etc.).
        # Normalization constants and the normalize_dataset function need to be
        # updated for the new naming scheme before this call can be re-enabled.
        # return data_utils.normalize_dataset(stacked, method=self.normalization_method)
        return stacked
