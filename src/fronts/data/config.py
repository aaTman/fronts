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
from typing import Literal

import numpy as np
import tensorflow as tf
import xarray as xr

from fronts.data import batch, era5, targets
from fronts.utils import calc, data_utils

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
            # Latitude flip — outer check is Python (trace-time); inner uses tf.cond
            if self.flip_chance_lat > 0:

                def _do_lat_flip():
                    _x = tf.reverse(x, axis=[0])
                    _y = tf.reverse(y, axis=[0])
                    if self.v_wind_index is not None:
                        n_vars = tf.shape(_x)[-1]
                        sign = tf.where(
                            tf.equal(tf.range(n_vars), self.v_wind_index), -1.0, 1.0
                        )
                        _x = _x * tf.cast(sign, _x.dtype)
                    return _x, _y

                def _no_lat_flip():
                    return x, y

                x, y = tf.cond(
                    tf.random.uniform(()) <= self.flip_chance_lat,
                    _do_lat_flip,
                    _no_lat_flip,
                )

            # Longitude flip — outer check is Python (trace-time); inner uses tf.cond
            if self.flip_chance_lon > 0:

                def _do_lon_flip():
                    _x = tf.reverse(x, axis=[1])
                    _y = tf.reverse(y, axis=[1])
                    if self.u_wind_index is not None:
                        n_vars = tf.shape(_x)[-1]
                        sign = tf.where(
                            tf.equal(tf.range(n_vars), self.u_wind_index), -1.0, 1.0
                        )
                        _x = _x * tf.cast(sign, _x.dtype)
                    return _x, _y

                def _no_lon_flip():
                    return x, y

                x, y = tf.cond(
                    tf.random.uniform(()) <= self.flip_chance_lon,
                    _do_lon_flip,
                    _no_lon_flip,
                )

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
        target_config: TargetDataConfig defining the front label source.
        augmentation_config: AugmentationConfig defining runtime data augmentation.
    """

    train_years: list[int]
    val_years: list[int]
    test_years: list[int]
    num_classes: int
    shuffle: bool
    normalization_method: str
    era5_config: era5.ERA5Config
    target_config: targets.TargetDataConfig
    augmentation_config: AugmentationConfig | None

    def build(self) -> ModelTrainingData:
        """Builds train, validation, and test tf.data.Dataset objects.

        Uses xbatcher BatchGenerator via batch.create_dataloader() for efficient
        lazy-loading from cloud-backed xarray Datasets with prefetching.

        Returns ModelTrainingData with train_data, validation_data, and optionally test_data.
        """
        log.info(
            "DataConfig.build() — using xbatcher path. "
            "train_years=%s, val_years=%s, test_years=%s",
            self.train_years,
            self.val_years,
            self.test_years,
        )

        def _build_split(years: list[int]) -> tf.data.Dataset:

            # Build ERA5 predictor dataset for this split
            log.info("  Building ERA5 predictor dataset for years=%s...", years)

            # Convert years to strings for xarray sel compatibility
            years = [str(year) for year in years]
            era5_cfg = dataclasses.replace(self.era5_config, years=years)
            inputs_ds = era5_cfg.build()
            log.info("  ERA5 dataset ready.")

            # Normalize using ARCO→legacy name bridging
            log.info("  Normalizing with method=%r...", self.normalization_method)
            inputs_ds = era5.normalize_legacy_arco_era5(
                inputs_ds, method=self.normalization_method
            )

            # Ensure all variables are consistently dask-backed before stacking.
            inputs_ds = inputs_ds.chunk({"time": 1})

            # Convert to 4D DataArray: (time, latitude, longitude, level, variable)
            inputs_da = inputs_ds.to_array(dim="variable")
            inputs_da = inputs_da.transpose(
                "time", "latitude", "longitude", "level", "variable"
            )
            log.info(
                "  Input DataArray shape: %s (time, lat, lon, level, variable)",
                dict(inputs_da.sizes),
            )

            # Build fronts target dataset (reformat + dilation already applied)
            log.info("  Building fronts dataset for years=%s...", years)
            fronts_ds = self.target_config.build()
            log.info("  Fronts dataset ready: %d timestep(s).", len(fronts_ds.time))

            # Align on time only — avoid xr.align which merges all dims
            # (lat/lon grids differ between ERA5 and fronts, creating a huge graph).
            era5_lats = inputs_da.latitude.values
            era5_lons = inputs_da.longitude.values
            common_times = np.intersect1d(fronts_ds.time.values, inputs_da.time.values)
            if len(common_times) == 0:
                raise RuntimeError(
                    f"No common timesteps between ERA5 and fronts for years={years}. "
                    f"Check time coordinate dtypes: ERA5={inputs_da.time.dtype}, "
                    f"fronts={fronts_ds.time.dtype}"
                )

            inputs_aligned = inputs_da.sel(time=common_times)
            fronts_da = (
                fronts_ds["identifier"]
                .sel(time=common_times)
                .sel(
                    latitude=era5_lats,
                    longitude=era5_lons,
                    method="nearest",
                )
                .transpose("time", "latitude", "longitude")
            )

            lat_size = inputs_aligned.sizes["latitude"]
            lon_size = inputs_aligned.sizes["longitude"]
            n_levels = inputs_aligned.sizes["level"]
            n_vars = inputs_aligned.sizes["variable"]

            log.info(
                "  Building xbatcher DataLoader: "
                "%d timesteps, input=(%d,%d,%d,%d), target=(%d,%d).",
                len(fronts_da),
                lat_size,
                lon_size,
                n_levels,
                n_vars,
                lat_size,
                lon_size,
            )

            input_sizes = {
                "time": 64,
                "latitude": lat_size,
                "longitude": lon_size,
                "level": n_levels,
                "variable": n_vars,
            }
            target_sizes = {
                "time": 64,
                "latitude": lat_size,
                "longitude": lon_size,
            }

            tf_ds = batch.create_dataloader(
                inputs=inputs_aligned,
                targets=fronts_da,
                input_sizes=input_sizes,
                target_sizes=target_sizes,
                preload_batch=False,
            )

            # # Squeeze time=1 dim and one-hot encode targets
            # def squeeze_and_encode(x, y):
            #     x = tf.squeeze(x, axis=0)  # (1,lat,lon,level,var) → (lat,lon,level,var)
            #     y = tf.squeeze(y, axis=0)  # (1,lat,lon) → (lat,lon)
            #     y = tf.one_hot(tf.cast(y, tf.int32), depth=self.num_classes)
            #     return x, y

            # tf_ds = tf_ds.map(squeeze_and_encode, num_parallel_calls=tf.data.AUTOTUNE)

            log.info("  tf.data.Dataset ready for years=%s.", years)
            return tf_ds

        log.info("Building train split...")
        train_ds = _build_split(self.train_years)
        if self.shuffle:
            log.debug("Shuffling train dataset.")
            train_ds = train_ds.shuffle(buffer_size=64)

        log.info("Building val split...")
        val_tf_ds = _build_split(self.val_years)

        test_ds = None
        if self.test_years:
            log.info("Building test split...")
            test_ds = _build_split(self.test_years)

        train_tf_ds = train_ds
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

    era5_config: era5.ERA5Config
    time_selection: list[datetime.datetime]
    normalization_method: Literal["standard", "standard_weighted", "min-max"]

    def build(self) -> xr.DataArray:
        """Loads, stacks, and normalizes ERA5 data for the selected timesteps.

        Opens the zarr store lazily, applies spatial subsetting, time selection,
        surface/pressure variable stacking, normalization, and converts to a 4D
        DataArray shaped ``(time, latitude, longitude, level, variable)``.

        Returns a normalized xarray DataArray ready for model inference.
        """
        log.info(
            "PredictConfig.build() — opening zarr store: %s", self.era5_config.store
        )
        ds = xr.open_dataset(
            self.era5_config.store,
            chunks=None,
            engine='zarr',
            consolidated=self.era5_config.consolidated,
        )
        log.debug("Zarr store opened.")

        # Partition variables into raw (load) and derived (compute).
        raw_vars = [
            v
            for v in self.era5_config.variables
            if v not in calc.derived_variable_callable_mapping
        ]
        to_derive = [
            v
            for v in self.era5_config.variables
            if v in calc.derived_variable_callable_mapping
        ]

        # 1. Variable subset (cheapest — narrows the graph immediately)
        log.debug(
            "Subsetting variables=%s at levels=%s...", raw_vars, self.era5_config.levels
        )
        ds = era5.subset_variables(
            ds, variables=raw_vars, levels=self.era5_config.levels
        )

        # 2. Time subset
        log.debug("Applying time selection: %s", self.time_selection)
        if len(self.time_selection) == 1:
            ds = ds.sel(time=[self.time_selection[0]])
        elif len(self.time_selection) == 2:
            ds = ds.sel(
                time=slice(self.time_selection[0], self.time_selection[1]),
            )
        else:
            raise ValueError(f"Invalid time selection: {self.time_selection}")
        log.info(
            "Time selection done. %d timestep(s) selected.", ds.sizes.get("time", 0)
        )

        # 3. Spatial subset
        log.debug("Applying spatial subset...")
        bbox = data_utils.convert_domain_extent_to_bounding_box(
            self.era5_config.domain_extent
        )
        lon_min = data_utils.maybe_convert_lon(bbox.lon_min, ds.longitude)
        lon_max = data_utils.maybe_convert_lon(bbox.lon_max, ds.longitude)
        ds = ds.sel(
            latitude=slice(bbox.lat_max, bbox.lat_min),
            longitude=slice(lon_min, lon_max),
        )

        # 4. Stack surface + pressure levels
        log.debug("Stacking raw variables=%s...", raw_vars)
        stacked = era5.maybe_stack_variables(
            ds,
            variables=raw_vars,
            levels=self.era5_config.levels,
        )

        # 5. Derive variables
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
        stacked = stacked.chunk(
            {"time": 1, "latitude": 90, "longitude": 180, "level": -1}
        )

        # Convert to 4D DataArray: (time, latitude, longitude, level, variable)
        result = stacked.to_array(dim="variable")
        result = result.transpose("time", "latitude", "longitude", "level", "variable")
        log.info(
            "PredictConfig.build() complete. Output shape: %s",
            dict(result.sizes),
        )
        return result
