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
import logging
import os
from typing import Any

import tensorflow as tf
import xarray as xr

from fronts.data.batch import BatchGeneratorConfig, create_dataloader
from fronts.data.era5 import convert_domain_extent_to_bounding_box
from fronts.utils import calc, data_utils

log = logging.getLogger("fronts.data.config")


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


# ---------------------------------------------------------------------------
# Derived variable computation
# ---------------------------------------------------------------------------

# Mapping from derived variable name → computation function.
# Each function receives the stacked xr.Dataset (which must already contain
# the prerequisite raw variables) and returns the dataset with the new
# variable added.  Order matters — e.g. dewpoint must be derived before
# virtual_temperature, relative_humidity, or theta_e.
GEOPOTENTIAL_TO_DAM = 98.0665  # geopotential (m²/s²) → geopotential height (dam)


def _derive_dewpoint(ds: xr.Dataset) -> xr.Dataset:
    """Derive dewpoint temperature from specific_humidity and pressure."""
    # Build a pressure array matching the data shape from the level coordinate.
    # Level values are hPa (int) or "surface" (str).  For "surface" we
    # approximate with 1013.25 hPa.
    level_hpa = [1013.25 if lv == "surface" else float(lv) for lv in ds.level.values]
    pressure_pa = xr.DataArray(level_hpa, dims="level", coords={"level": ds.level}) * 100
    ds["dewpoint"] = calc.dewpoint_from_specific_humidity(pressure_pa, ds["specific_humidity"])
    return ds


def _derive_virtual_temperature(ds: xr.Dataset) -> xr.Dataset:
    """Derive virtual temperature from temperature, dewpoint, and pressure."""
    level_hpa = [1013.25 if lv == "surface" else float(lv) for lv in ds.level.values]
    pressure_pa = xr.DataArray(level_hpa, dims="level", coords={"level": ds.level}) * 100
    ds["virtual_temperature"] = calc.virtual_temperature_from_dewpoint(
        pressure_pa, ds["temperature"], ds["dewpoint"]
    )
    return ds


def _derive_relative_humidity(ds: xr.Dataset) -> xr.Dataset:
    """Derive relative humidity from temperature and dewpoint."""
    ds["relative_humidity"] = calc.relative_humidity_from_dewpoint(
        ds["temperature"], ds["dewpoint"]
    )
    return ds


def _derive_theta_e(ds: xr.Dataset) -> xr.Dataset:
    """Derive equivalent potential temperature from temperature, dewpoint, and pressure."""
    level_hpa = [1013.25 if lv == "surface" else float(lv) for lv in ds.level.values]
    pressure_pa = xr.DataArray(level_hpa, dims="level", coords={"level": ds.level}) * 100
    ds["equivalent_potential_temperature"] = calc.equivalent_potential_temperature(
        pressure_pa, ds["temperature"], ds["dewpoint"]
    )
    return ds


def _derive_geopotential_height(ds: xr.Dataset) -> xr.Dataset:
    """Convert geopotential (m²/s²) to geopotential height in decameters."""
    ds["geopotential_height"] = ds["geopotential"] / GEOPOTENTIAL_TO_DAM
    return ds


DERIVED_VARIABLE_REGISTRY: dict[str, callable] = {
    "dewpoint": _derive_dewpoint,
    "virtual_temperature": _derive_virtual_temperature,
    "relative_humidity": _derive_relative_humidity,
    "equivalent_potential_temperature": _derive_theta_e,
    "geopotential_height": _derive_geopotential_height,
}


def _derive_era5_variables(
    ds: xr.Dataset, derived_variables: list[str]
) -> xr.Dataset:
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
    years: list[int] = dataclasses.field(default_factory=list)

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
        log.info(
            "ERA5PredictorConfig.build() — opening zarr store: %s", self.store
        )
        ds = xr.open_zarr(
            store=self.store,
            chunks=self.chunks,
            consolidated=self.consolidated,
        )
        log.debug("Zarr store opened. Variables available: %s", list(ds.data_vars))

        # Spatial subset
        log.debug("Applying spatial subset: domain_extent=%s", self.domain_extent)
        bbox = convert_domain_extent_to_bounding_box(self.domain_extent)
        ds = ds.sel(
            latitude=slice(bbox.lat_max, bbox.lat_min),
            longitude=slice(bbox.lon_min, bbox.lon_max),
        )
        log.debug(
            "Spatial subset done. lat shape=%s, lon shape=%s",
            ds.latitude.shape, ds.longitude.shape,
        )

        # Temporal subset: keep only the requested years
        log.debug("Applying temporal subset for years=%s...", self.years)
        ds = ds.isel(time=ds.time.dt.year.isin(self.years))
        log.info("ERA5 temporal subset done. %d timesteps selected.", ds.sizes.get("time", 0))

        # Partition variables: raw ones are loaded from the store, derived
        # ones are computed after stacking.
        raw_vars = [v for v in self.variables if v not in DERIVED_VARIABLE_REGISTRY]
        to_derive = [v for v in self.variables if v in DERIVED_VARIABLE_REGISTRY]

        log.debug("Stacking raw variables=%s at levels=%s...", raw_vars, self.levels)
        result = _stack_era5_variables(
            ds,
            variables=raw_vars,
            levels=self.levels,
        )

        if to_derive:
            log.info("Deriving variables: %s", to_derive)
            result = _derive_era5_variables(result, to_derive)

        log.info(
            "ERA5PredictorConfig.build() complete. Output vars: %s", list(result.data_vars)
        )
        return result


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
    front_types: str | list[str] | None
    years: list[int] = dataclasses.field(default_factory=list)

    def build(self) -> xr.Dataset:
        """Loads and optionally reformats front label data for the given years.

        Supports two directory layouts automatically:
        - Flat:  ``{directory}/*{year}*.nc``
        - Monthly subdirs: ``{directory}/{year}MM/*.nc`` (e.g. ``200701/``)

        Returns an xarray Dataset with the "identifier" variable, optionally
        reformatted according to self.front_types.
        """
        log.info(
            "FrontsDataConfig.build() — globbing files for years=%s in %r...",
            self.years, self.directory,
        )
        files = sorted(
            f
            for year in self.years
            for f in (
                # Monthly subdirectory layout: <dir>/<YYYYMM>/*.nc
                glob_module.glob(f"{self.directory}/{year}*/*.nc")
                or
                # Flat layout fallback: <dir>/*<year>*.nc
                glob_module.glob(f"{self.directory}/*{year}*.nc")
            )
        )
        log.info("FrontsDataConfig — found %d file(s). Opening with open_mfdataset...", len(files))
        ds = xr.open_mfdataset(files, engine="netcdf4", combine="by_coords", coords="minimal", compat="override")
        log.info("FrontsDataConfig — dataset opened. Variables: %s", list(ds.data_vars))

        if self.front_types is not None:
            log.debug("Reformatting fronts with front_types=%s...", self.front_types)
            ds = data_utils.reformat_fronts(ds, self.front_types)
            log.debug("reformat_fronts complete.")

        log.info("FrontsDataConfig.build() complete.")
        return ds


@dataclasses.dataclass
class AugmentationConfig:
    """Runtime augmentation applied to training data via .map().

    Attributes:
        flip_chance_lat: Probability of flipping the latitude axis per sample.
        flip_chance_lon: Probability of flipping the longitude axis per sample.
        u_wind_index: Index of u_component_of_wind in the variable dimension
            (last axis). Required for lon flip to negate u-wind. None to skip.
        v_wind_index: Index of v_component_of_wind in the variable dimension
            (last axis). Required for lat flip to negate v-wind. None to skip.
    """

    flip_chance_lat: float = 0.0
    flip_chance_lon: float = 0.0
    u_wind_index: int | None = None
    v_wind_index: int | None = None


@dataclasses.dataclass
class TFDatasetConfig:
    """Configuration for loading pre-built tf.data.Dataset snapshots from disk.

    The on-disk datasets were produced by the previous pipeline and are stored
    as saved ``tf.data.Dataset`` snapshots in year-labelled subdirectories under
    a common root, e.g.::

        <directory>/
            2010-1_tf/
            2010-2_tf/
            ...
            2020-12_tf/

    All monthly subdirectories whose name starts with a requested year are
    concatenated in sorted order to form the split dataset.

    This is the fastest path to training — it bypasses ERA5 zarr loading,
    front netCDF loading, stacking, and normalization entirely, using data that
    is already preprocessed and on local disk.

    Attributes:
        directory: Root directory containing the year-month subdirectories.
        train_years: Years to include in the training split.
        val_years: Years to include in the validation split.
        test_years: Years to include in the test split. May be empty [].
        shuffle: Whether to shuffle the training dataset. Defaults to True.
        shuffle_buffer: Buffer size passed to ``tf.data.Dataset.shuffle()``.
            Defaults to 1000.
        prefetch: Number of batches to prefetch. Defaults to 3.
    """

    directory: str
    # Years default to empty lists — DataConfig.build() injects the correct
    # values via dataclasses.replace() at runtime, so the YAML doesn't need
    # to repeat them when tf_dataset is nested inside a DataConfig block.
    train_years: list[int] = dataclasses.field(default_factory=list)
    val_years: list[int] = dataclasses.field(default_factory=list)
    test_years: list[int] = dataclasses.field(default_factory=list)
    shuffle: bool = True
    shuffle_buffer: int = 1000
    prefetch: int = 3
    # Optional metadata for WandB logging only — not used by the data pipeline.
    # ERA5 variable names and pressure levels are baked into the pre-built TF
    # dataset; these fields let you record what was used for experiment tracking.
    variables: list[str] | None = None
    levels: list[str | int] | None = None

    def _load_years(self, years: list[int]) -> Any | None:
        """Loads and concatenates all monthly TF dataset snapshots for ``years``.

        Subdirectories are matched by prefix: a directory named ``2010-3_tf``
        matches year ``2010``.

        Returns a ``tf.data.Dataset``, or ``None`` if ``years`` is empty or no
        matching subdirectories are found.
        """
        if not years:
            log.debug("_load_years called with empty years list — returning None.")
            return None

        log.debug("Scanning %r for subdirs matching years %s...", self.directory, years)
        subdirs = sorted(
            os.path.join(self.directory, d)
            for d in os.listdir(self.directory)
            if any(d.startswith(str(y)) for y in years)
            and os.path.isdir(os.path.join(self.directory, d))
        )

        if not subdirs:
            raise FileNotFoundError(
                f"No subdirectories found in {self.directory!r} matching years "
                f"{years}. Expected names like '2010-1_tf', '2010-2_tf', etc."
            )

        log.info("Loading %d TF dataset snapshot(s) for years %s...", len(subdirs), years)
        datasets = []
        for i, s in enumerate(subdirs):
            log.debug("  [%d/%d] Loading %s", i + 1, len(subdirs), s)
            datasets.append(tf.data.Dataset.load(s))
        log.debug("All snapshots loaded. Concatenating...")

        combined = datasets[0]
        for ds in datasets[1:]:
            combined = combined.concatenate(ds)
        log.debug("Concatenation complete. Applying prefetch=%d.", self.prefetch)
        return combined.prefetch(self.prefetch)

    def build(self) -> "ModelData":
        """Builds train, validation, and test ``tf.data.Dataset`` objects.

        Returns a :class:`ModelData` with ``train_data``, ``validation_data``,
        and optionally ``test_data``.
        """
        log.info("TFDatasetConfig.build() — loading train split (years=%s)...", self.train_years)
        train_ds = self._load_years(self.train_years)
        if self.shuffle and train_ds is not None:
            log.debug("Shuffling train dataset (buffer_size=%d).", self.shuffle_buffer)
            train_ds = train_ds.shuffle(buffer_size=self.shuffle_buffer)

        log.info("TFDatasetConfig.build() — loading val split (years=%s)...", self.val_years)
        val_ds = self._load_years(self.val_years)

        if self.test_years:
            log.info("TFDatasetConfig.build() — loading test split (years=%s)...", self.test_years)
        test_ds = self._load_years(self.test_years)

        log.info("TFDatasetConfig.build() complete.")
        return ModelData(
            train_data=train_ds,
            validation_data=val_ds,
            test_data=test_ds,
        )


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
    test_data: Any | None = None


@dataclasses.dataclass
class DataConfig:
    """Top-level data configuration for the FrontFinder training pipeline.

    Supports two mutually exclusive data sources:

    1. **Pre-built TF datasets** (``tf_dataset`` key) — fastest path, loads
       saved ``tf.data.Dataset`` snapshots directly from disk.  Set
       ``tf_dataset:`` in YAML and leave ``era5``, ``fronts``, ``batch`` unset.

    2. **ARCO ERA5 + front netCDF** (``era5`` + ``fronts`` + ``batch`` keys) —
       full pipeline that loads from the zarr store and front label files,
       stacks variables, and builds batches via xbatcher.

    Year lists (``train_years``, ``val_years``, ``test_years``) are always
    specified at this level.  For the ERA5 path they are injected into
    ``ERA5PredictorConfig`` and ``FrontsDataConfig`` at build time via
    ``dataclasses.replace()`` — they should NOT be set in the era5/fronts YAML
    blocks.

    Attributes:
        train_years: Years to use for the training split.
        val_years: Years to use for the validation split.
        test_years: Years to use for the test split. May be empty [].
        tf_dataset: TFDatasetConfig for loading pre-built TF dataset snapshots.
            Mutually exclusive with era5/fronts/batch.
        era5: ERA5PredictorConfig defining the predictor variable source.
            Required when tf_dataset is not set.
        fronts: FrontsDataConfig defining the front label source.
            Required when tf_dataset is not set.
        batch: BatchGeneratorConfig defining spatial patch sizes and prefetch.
            Required when tf_dataset is not set.
        shuffle: Whether to shuffle the training dataset. Defaults to True.
            Ignored when tf_dataset is set (shuffle is configured there instead).
        normalization_method: One of "standard", "standard_weighted", "min-max".
            Defaults to "standard". Only used by the ERA5 path.
    """

    train_years: list[int]
    val_years: list[int]
    test_years: list[int]
    tf_dataset: TFDatasetConfig | None = None
    era5: ERA5PredictorConfig | None = None
    fronts: FrontsDataConfig | None = None
    batch: BatchGeneratorConfig | None = None
    shuffle: bool = True
    normalization_method: str = "standard"
    augmentation: AugmentationConfig | None = None

    def build(self) -> ModelData:
        """Builds train, validation, and test tf.data.Dataset objects.

        Delegates to TFDatasetConfig.build() when tf_dataset is set, otherwise
        uses the ERA5 + fronts pipeline.

        Returns a ModelData with train_data, validation_data, and optionally test_data.
        """
        if self.tf_dataset is not None:
            log.info(
                "DataConfig.build() — using TFDatasetConfig path. "
                "train_years=%s, val_years=%s, test_years=%s",
                self.train_years, self.val_years, self.test_years,
            )
            # Inject year lists into the TFDatasetConfig and build
            tf_cfg = dataclasses.replace(
                self.tf_dataset,
                train_years=self.train_years,
                val_years=self.val_years,
                test_years=self.test_years,
                shuffle=self.shuffle,
            )
            return tf_cfg.build()

        # --- ERA5 + fronts path ---
        log.info(
            "DataConfig.build() — using ERA5+fronts path. "
            "train_years=%s, val_years=%s, test_years=%s",
            self.train_years, self.val_years, self.test_years,
        )
        if self.era5 is None or self.fronts is None or self.batch is None:
            raise ValueError(
                "DataConfig requires either tf_dataset or all three of "
                "era5, fronts, and batch to be set."
            )

        def _build_split(years: list[int]) -> Any | None:
            if not years:
                return None

            # Build ERA5 predictor dataset for this split
            log.info("  Building ERA5 predictor dataset for years=%s...", years)
            era5_cfg = dataclasses.replace(self.era5, years=years)
            inputs_ds = era5_cfg.build()
            log.info("  ERA5 dataset ready.")
            # TODO: normalize_dataset expects a "pressure_level" dimension and legacy
            # short variable-name keys (e.g. "T_850", "u_1000"). Our stacked dataset
            # uses dimension "level" and ARCO variable names ("temperature", etc.).
            # Normalization constants and the normalize_dataset function need to be
            # updated for the new naming scheme before this call can be re-enabled.
            # inputs_ds = data_utils.normalize_dataset(
            #     inputs_ds, method=self.normalization_method
            # )

            # Build fronts dataset for this split
            log.info("  Building fronts dataset for years=%s...", years)
            fronts_cfg = dataclasses.replace(self.fronts, years=years)
            targets_ds = fronts_cfg.build()
            log.info("  Fronts dataset ready.")

            # Align timestamps: keep only times present in BOTH datasets.
            # ERA5 is hourly; fronts exist only at analysis times (e.g. 00Z/12Z).
            common_times = sorted(
                set(inputs_ds.time.values) & set(targets_ds.time.values)
            )
            if not common_times:
                raise ValueError(
                    f"No overlapping timestamps between ERA5 and fronts "
                    f"for years={years}."
                )
            n_era5 = inputs_ds.sizes["time"]
            n_fronts = targets_ds.sizes["time"]
            inputs_ds = inputs_ds.sel(time=common_times)
            targets_ds = targets_ds.sel(time=common_times)
            log.info(
                "  Timestamp alignment: ERA5=%d, fronts=%d → %d common timesteps.",
                n_era5, n_fronts, len(common_times),
            )

            # Build tf.data.Dataset via create_dataloader directly
            # (BatchGeneratorConfig.build() is not used because it lacks inputs/targets fields)
            log.info("  Wrapping into tf.data.Dataset via create_dataloader...")
            tf_ds = create_dataloader(
                inputs=inputs_ds,
                targets=targets_ds,
                input_sizes=self.batch.input_sizes,
                target_sizes=self.batch.target_sizes,
                prefetch_number=self.batch.prefetch_number,
                preload_batch=self.batch.preload_batch,
            )
            log.info("  tf.data.Dataset ready for years=%s.", years)
            return tf_ds

        log.info("Building train split...")
        train_ds = _build_split(self.train_years)
        if self.shuffle and train_ds is not None:
            log.debug("Shuffling train dataset.")
            train_ds = train_ds.shuffle(buffer_size=1000)

        log.info("Building val split...")
        val_ds = _build_split(self.val_years)

        if self.test_years:
            log.info("Building test split...")
        test_ds = _build_split(self.test_years)

        log.info("DataConfig.build() complete.")
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
    timestamps: list[datetime.datetime] | None = None
    date_range: list[datetime.datetime] | None = None  # exactly [start, end]

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
        log.info("PredictConfig.build() — opening zarr store: %s", self.era5.store)
        ds = xr.open_zarr(
            store=self.era5.store,
            chunks=self.era5.chunks,
            consolidated=self.era5.consolidated,
        )
        log.debug("Zarr store opened.")

        # Spatial subset
        log.debug("Applying spatial subset...")
        bbox = convert_domain_extent_to_bounding_box(self.era5.domain_extent)
        ds = ds.sel(
            latitude=slice(bbox.lat_max, bbox.lat_min),
            longitude=slice(bbox.lon_min, bbox.lon_max),
        )

        # Time selection
        log.debug("Applying time selection: %s", self.time_selection)
        ds = self.time_selection.apply(ds)
        log.info("Time selection done. %d timestep(s) selected.", ds.sizes.get("time", 0))

        # Partition variables into raw (load) and derived (compute).
        raw_vars = [v for v in self.era5.variables if v not in DERIVED_VARIABLE_REGISTRY]
        to_derive = [v for v in self.era5.variables if v in DERIVED_VARIABLE_REGISTRY]

        log.debug("Stacking raw variables=%s...", raw_vars)
        stacked = _stack_era5_variables(
            ds,
            variables=raw_vars,
            levels=self.era5.levels,
        )

        if to_derive:
            log.info("Deriving variables: %s", to_derive)
            stacked = _derive_era5_variables(stacked, to_derive)

        log.info("PredictConfig.build() — stacking complete. Output vars: %s", list(stacked.data_vars))
        # TODO: normalize_dataset expects a "pressure_level" dimension and legacy
        # short variable-name keys (e.g. "T_850", "u_1000"). Our stacked dataset
        # uses dimension "level" and ARCO variable names ("temperature", etc.).
        # Normalization constants and the normalize_dataset function need to be
        # updated for the new naming scheme before this call can be re-enabled.
        # return data_utils.normalize_dataset(stacked, method=self.normalization_method)
        log.info("PredictConfig.build() complete.")
        return stacked
