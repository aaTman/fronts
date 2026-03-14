"""Data configuration dataclasses for the FrontFinder training pipeline.

This module provides composable config dataclasses for loading ERA5 predictor data
and front label (truth) data, splitting by year, stacking surface and pressure-level
variables in xarray, and building tf.data.Dataset objects for train/val/test splits.

Follows the same pattern as the rest of the codebase: typed dataclasses with .build()
methods that return runtime objects, loadable from YAML via dacite.
"""

import dataclasses
import datetime
import logging
from typing import Any, Literal

import numpy as np
import tensorflow as tf
import xarray as xr

from fronts.data import era5, targets
from fronts.utils import data_utils

log = logging.getLogger("fronts.data.config")


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

    flip_chance_lat: float
    flip_chance_lon: float
    u_wind_index: int | None
    v_wind_index: int | None

    def build(self):
        """Returns a tf.function that randomly flips input/target pairs.

        Wind components are negated when their spatial axis is flipped:
        v-wind is negated on latitude flip, u-wind on longitude flip.
        """

        @tf.function
        def augment(x, y):
            # Latitude flip
            if self.flip_chance_lat > 0:
                if tf.random.uniform(()) <= self.flip_chance_lat:
                    x = tf.reverse(x, axis=[0])
                    y = tf.reverse(y, axis=[0])
                    # Negate v-wind across all levels
                    if self.v_wind_index is not None:
                        n_vars = tf.shape(x)[-1]
                        sign = tf.where(
                            tf.equal(tf.range(n_vars), self.v_wind_index), -1.0, 1.0
                        )
                        x = x * tf.cast(sign, x.dtype)

            # Longitude flip
            if self.flip_chance_lon > 0:
                if tf.random.uniform(()) <= self.flip_chance_lon:
                    x = tf.reverse(x, axis=[1])
                    y = tf.reverse(y, axis=[1])
                    # Negate u-wind across all levels
                    if self.u_wind_index is not None:
                        n_vars = tf.shape(x)[-1]
                        sign = tf.where(
                            tf.equal(tf.range(n_vars), self.u_wind_index), -1.0, 1.0
                        )
                        x = x * tf.cast(sign, x.dtype)

            return x, y

        return augment


@dataclasses.dataclass
class ModelTrainingData:
    """Runtime holder for train/validation/test tf.data.Dataset objects.

    Returned by DataConfig.build(). Trainer accesses .train_data and
    .validation_data directly (train.py:221,228).

    Attributes:
        train_data: tf.data.Dataset for training.
        validation_data: tf.data.Dataset for validation.
        test_data: Optional tf.data.Dataset for testing. None if test_years is empty.
    """

    train_data: tf.data.Dataset
    validation_data: tf.data.Dataset
    test_data: tf.data.Dataset | None


@dataclasses.dataclass
class DataConfig:
    """Top-level data configuration for the FrontFinder training pipeline.

    Year lists (``train_years``, ``val_years``, ``test_years``) are always
    specified at this level.  For the ERA5 path they are injected into
    ``ERA5PredictorConfig`` and ``FrontsDataConfig`` at build time via
    ``dataclasses.replace()`` — they should not be set in the era5/fronts YAML
    blocks.

    Attributes:
        train_years: Years to use for the training split.
        val_years: Years to use for the validation split.
        test_years: Years to use for the test split. May be empty [].
        num_classes: Number of front classes (including no-front class 0).
        shuffle: Whether to shuffle the training dataset.
        normalization_method: One of "standard", "standard_weighted", "min-max".
        era5: ERA5PredictorConfig defining the predictor variable source.
        fronts: TargetDataConfig defining the front label source.
        augmentation_config: AugmentationConfig defining runtime data augmentation.
    """

    train_years: list[int]
    val_years: list[int]
    test_years: list[int]
    num_classes: int
    shuffle: bool
    normalization_method: str
    era5_config: era5.ERA5PredictorConfig
    fronts_config: targets.TargetDataConfig
    augmentation_config: AugmentationConfig | None

    def build(self) -> ModelTrainingData:
        """Builds train, validation, and test tf.data.Dataset objects.

        Returns ModelTrainingData with train_data, validation_data, and optionally test_data.
        """
        # --- ERA5 + fronts path ---
        log.info(
            "DataConfig.build() — using ERA5+fronts path. "
            "train_years=%s, val_years=%s, test_years=%s",
            self.train_years,
            self.val_years,
            self.test_years,
        )

        def _build_split(years: list[int]) -> Any | None:
            if not years:
                return None

            # Build ERA5 predictor dataset for this split
            log.info("  Building ERA5 predictor dataset for years=%s...", years)
            era5_cfg = dataclasses.replace(self.era5_config, years=years)
            inputs_ds = era5_cfg.build()
            log.info("  ERA5 dataset ready.")

            # Normalize using ARCO→legacy name bridging
            log.info("  Normalizing with method=%r...", self.normalization_method)
            inputs_ds = era5.normalize_legacy_arco_era5(
                inputs_ds, method=self.normalization_method
            )

            # Ensure all variables are consistently dask-backed before stacking.
            # xr.merge(join="outer") fills missing coordinate slots (e.g. MSLP at
            # pressure levels) with numpy NaN rather than dask arrays.  When
            # to_array() calls dask.stack() on a mix of dask + numpy arrays it
            # falls through to dask.asarray(chunks="auto"), which triggers a
            # ZeroDivisionError in auto_chunks on dask 2023.3.x.  Rechunking the
            # whole Dataset forces every variable into the dask graph with an
            # explicit chunk spec, so all arrays passed to dask.stack are
            # consistently chunked dask arrays.
            inputs_ds = inputs_ds.chunk({"latitude": -1, "longitude": -1, "level": -1})

            # Convert to 4D DataArray: (time, latitude, longitude, level, variable)
            # The model expects (lat, lon, level, variable) per timestep.
            inputs_da = inputs_ds.to_array(dim="variable")
            inputs_da = inputs_da.transpose(
                "time", "latitude", "longitude", "level", "variable"
            )
            log.info(
                "  Input DataArray shape: %s (time, lat, lon, level, variable)",
                dict(inputs_da.sizes),
            )

            # Build fronts file index for this split — O(n_files glob) only,
            # NO netCDF files are opened here.  Timestamps are parsed from
            # filenames matching FrontObjects_YYYYMMDDHH_full.nc.
            log.info("  Building fronts file index for years=%s...", years)

            fronts_ds = self.fronts_config.build()  # {iso_str: filepath}
            log.info("  Fronts index ready: %d timestep(s).", len(fronts_ds.time))

            # Align timestamps: keep only times present in BOTH ERA5 and fronts.
            # ERA5 is hourly; fronts exist only at analysis times (00Z/06Z/12Z/18Z).
            # Use ISO second-precision strings as the common key to avoid
            # numpy datetime64 precision mismatches (zarr uses ns, filenames give s).
            def _iso(t) -> str:
                """Truncate a numpy datetime64 scalar to second-precision ISO."""
                return str(t)[:19]  # "2008-01-01T00:00:00"

            era5_time_map: dict[str, Any] = {_iso(t): t for t in inputs_da.time.values}
            common_keys = sorted(set(era5_time_map.keys()) & set(fronts_ds.time.values))
            if not common_keys:
                raise ValueError(
                    f"No overlapping timestamps between ERA5 and fronts "
                    f"for years={years}."
                )
            log.info(
                "  Timestamp alignment: ERA5=%d, fronts=%d → %d common timesteps.",
                inputs_da.sizes["time"],
                len(fronts_ds.time),
                len(common_keys),
            )

            # Subset ERA5 DataArray to aligned timestamps.
            aligned_era5_times = [era5_time_map[k] for k in common_keys]
            inputs_da = inputs_da.sel(time=aligned_era5_times)

            # Build tf.data.Dataset via generator.
            # For each timestep the generator:
            #   1. Reads the ERA5 slice (triggers one dask chunk load from zarr).
            #   2. Opens the corresponding front netCDF file (one file, one timestep).
            #   3. Applies reformat_fronts + expand_fronts per-file — avoids loading
            #      all 67k+ files with open_mfdataset at startup.
            n_times = len(common_keys)
            lat_size = inputs_da.sizes["latitude"]
            lon_size = inputs_da.sizes["longitude"]
            n_levels = inputs_da.sizes["level"]
            n_vars = inputs_da.sizes["variable"]

            log.info(
                "  Building tf.data.Dataset from generator: "
                "%d timesteps, input=(%d,%d,%d,%d), target=(%d,%d).",
                n_times,
                lat_size,
                lon_size,
                n_levels,
                n_vars,
                lat_size,
                lon_size,
            )

            # Coordinate arrays used inside gen() to subset each front file.
            # inputs_da uses ARCO's 0-360 longitude; front files typically use -180/180.
            era5_lats = inputs_da.latitude.values  # e.g. [60.0, 59.75, ..., 20.0]
            era5_lons_360 = inputs_da.longitude.values  # e.g. [220.0, ..., 300.0]
            era5_lons_180 = np.where(
                era5_lons_360 > 180, era5_lons_360 - 360, era5_lons_360
            )  # e.g. [-140.0, ..., -60.0]

            def gen():
                for i, iso_key in enumerate(common_keys):
                    # ERA5 input — isel by position for speed
                    x = inputs_da.isel(time=i).values.astype("float32")

                    # Front label — open one file, apply transforms, discard
                    front_ds = fronts_ds.sel(time=iso_key)
                    if self.fronts_config.front_types is not None:
                        front_ds = data_utils.reformat_fronts(
                            front_ds, self.fronts_config.front_types
                        )
                    if self.fronts_config.front_dilation > 0:
                        front_ds = data_utils.expand_fronts(
                            front_ds, iterations=self.fronts_config.front_dilation
                        )
                    # squeeze() removes any degenerate time dim present in some files
                    identifier = front_ds["identifier"].squeeze(drop=True)
                    # Front files cover a broader domain than the ERA5 subset.
                    # Detect the longitude convention used by this file and pick the
                    # matching ERA5 lon array (0-360 or -180/180).
                    sel_lons = (
                        era5_lons_360
                        if float(identifier.longitude.min()) >= 0
                        else era5_lons_180
                    )
                    identifier = identifier.sel(
                        latitude=era5_lats,
                        longitude=sel_lons,
                        method="nearest",
                    )
                    # Guarantee (latitude, longitude) ordering regardless of storage order.
                    identifier = identifier.transpose("latitude", "longitude")
                    y = identifier.values.astype("float32")
                    yield x, y

            tf_ds = tf.data.Dataset.from_generator(
                gen,
                output_signature=(
                    tf.TensorSpec(
                        shape=(lat_size, lon_size, n_levels, n_vars),
                        dtype=tf.float32,
                    ),
                    tf.TensorSpec(
                        shape=(lat_size, lon_size),
                        dtype=tf.float32,
                    ),
                ),
            )

            # One-hot encode targets: (lat, lon) int → (lat, lon, num_classes)
            def encode_targets(x, y):
                y = tf.one_hot(tf.cast(y, tf.int32), depth=self.num_classes)
                return x, y

            tf_ds = tf_ds.map(encode_targets, num_parallel_calls=tf.data.AUTOTUNE)

            log.info("  tf.data.Dataset ready for years=%s.", years)
            return tf_ds

        log.info("Building train split...")
        train_ds = _build_split(self.train_years)
        if self.shuffle and train_ds is not None:
            log.debug("Shuffling train dataset.")
            train_ds = train_ds.shuffle(buffer_size=1000)

        log.info("Building val split...")
        val_tf_ds = _build_split(self.val_years)

        if self.test_years:
            log.info("Building test split...")
        test_ds = _build_split(self.test_years)
        if self.augmentation_config:
            log.info("Building augmentation function...")
            augment_fn = self.augmentation_config.build()
            log.info("Applying augmentation to train dataset...")
            train_tf_ds = train_ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
            log.info(
                "Data augmentation applied to training dataset with config: %s",
                self.augmentation_config,
            )
        log.info("DataConfig.build() complete.")
        return ModelTrainingData(
            train_data=train_tf_ds,
            validation_data=val_tf_ds,
            test_data=test_ds,
        )


@dataclasses.dataclass
class PredictConfig:
    """Configuration for ERA5-based model inference.

    Mirrors DataConfig for training but uses TimeSelection instead of year lists
    for temporal filtering, and returns a normalized xr.DataArray shaped
    ``(time, latitude, longitude, level, variable)`` — inference runs one
    spatial domain at a time rather than batching many patches.

    Attributes:
        era5: ERA5PredictorConfig defining the variable source, spatial domain,
            and zarr store. The `years` field on era5 is unused by PredictConfig;
            time selection is fully controlled by time_selection.
        time_selection: One or two datetimes specifying which timesteps to load. Two
            datetimes will select a range of timesteps.
        normalization_method: One of "standard", "standard_weighted", "min-max".
            Should match the normalization used during training. Defaults to
            "standard".
    """

    era5_config: era5.ERA5PredictorConfig
    time_selection: list[datetime.datetime]
    normalization_method: Literal["standard", "standard_weighted", "min-max"]

    def build(self) -> xr.DataArray:
        """Loads, stacks, and normalizes ERA5 data for the selected timesteps.

        Opens the zarr store lazily, applies spatial subsetting, time selection,
        surface/pressure variable stacking, normalization, and converts to a 4D
        DataArray shaped ``(time, latitude, longitude, level, variable)``.

        Returns a normalized xarray DataArray ready for model inference.
        """
        log.info("PredictConfig.build() — opening zarr store: %s", self.era5_config.store)
        ds = xr.open_zarr(
            store=self.era5_config.store,
            chunks=self.era5_config.chunks,
            consolidated=self.era5_config.consolidated,
        )
        log.debug("Zarr store opened.")

        # Spatial subset
        log.debug("Applying spatial subset...")
        bbox = data_utils.convert_domain_extent_to_bounding_box(self.era5_config.domain_extent)
        lon_min = data_utils.maybe_convert_lon(bbox.lon_min, ds.longitude)
        lon_max = data_utils.maybe_convert_lon(bbox.lon_max, ds.longitude)
        ds = ds.sel(
            latitude=slice(bbox.lat_max, bbox.lat_min),
            longitude=slice(lon_min, lon_max),
        )

        # Time selection
        log.debug("Applying time selection: %s", self.time_selection)
        if len(self.time_selection) == 1:
            ds = ds.sel(time=self.time_selection[0])
        elif len(self.time_selection) == 2:
            ds = ds.sel(
                time=slice(self.time_selection[0], self.time_selection[1]),
                method="nearest",
            )
        else:
            raise ValueError(f"Invalid time selection: {self.time_selection}")
        log.info(
            "Time selection done. %d timestep(s) selected.", ds.sizes.get("time", 0)
        )

        # Partition variables into raw (load) and derived (compute).
        raw_vars = [
            v for v in self.era5_config.variables if v not in era5.derived_variable_callable_mapping
        ]
        to_derive = [
            v for v in self.era5_config.variables if v in era5.derived_variable_callable_mapping
        ]

        log.debug("Stacking raw variables=%s...", raw_vars)
        stacked = era5.stack_variables(
            ds,
            variables=raw_vars,
            levels=self.era5_config.levels,
        )

        if to_derive:
            log.info("Deriving variables: %s", to_derive)
            stacked = era5.derive_era5_variables(stacked, to_derive)

        log.info(
            "PredictConfig.build() — stacking complete. Output vars: %s",
            list(stacked.data_vars),
        )

        # Normalize using ARCO→legacy name bridging
        log.info("Normalizing with method=%r...", self.normalization_method)
        stacked = era5.normalize_legacy_arco_era5(
            stacked, method=self.normalization_method
        )

        # Rechunk before to_array() for the same reason as _build_split():
        # outer-join merge fills missing levels with numpy NaN; rechunking ensures
        # all variables are consistently dask-backed before dask.stack is called.
        stacked = stacked.chunk({"latitude": -1, "longitude": -1, "level": -1})

        # Convert to 4D DataArray: (time, latitude, longitude, level, variable)
        result = stacked.to_array(dim="variable")
        result = result.transpose("time", "latitude", "longitude", "level", "variable")
        log.info(
            "PredictConfig.build() complete. Output shape: %s",
            dict(result.sizes),
        )
        return result
