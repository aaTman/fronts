"""
Generate performance statistics for a model.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2025.2.16
"""

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
import os
import regionmask
from fronts.utils import data_utils
from fronts.utils.constants import DOMAIN_EXTENTS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Directory for the models."
    )
    parser.add_argument("--model_number", type=str, required=True, help="Model name/number string (e.g. 1702_retrain).")
    parser.add_argument(
        "--tf_indir",
        type=str,
        help="Directory for the TensorFlow dataset used for model evaluation.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset for which to make predictions. Options are: 'training', 'validation', 'test'",
    )
    parser.add_argument(
        "--year_and_month",
        type=int,
        nargs=2,
        help="Year and month for which to make predictions.",
    )
    parser.add_argument("--domain", type=str, help="Domain of the data.")
    parser.add_argument("--gpu_device", type=int, nargs="+", help="GPU device number.")
    parser.add_argument(
        "--memory_growth", action="store_true", help="Use memory growth on the GPU"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing statistics files.",
    )
    parser.add_argument(
        "--front_types",
        type=str,
        nargs="+",
        default=None,
        help="Front type names (e.g. CF WF SF OF DL). Required when no model properties pkl exists.",
    )
    parser.add_argument(
        "--prediction_file",
        type=str,
        default=None,
        help=(
            "Path to the predictions NetCDF (output of predict_from_tf_shards.py). "
            "When provided, overrides the auto-computed path from --model_dir/--model_number."
        ),
    )
    parser.add_argument(
        "--fronts_file",
        type=str,
        default=None,
        help=(
            "Path to a single truth fronts NetCDF file (e.g. fronts_subset.nc). "
            "When provided, bypasses the legacy front_files_YYYYMM.pkl lookup and "
            "reads truth labels directly from this file."
        ),
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        choices=["land", "ocean"],
        help=(
            "Restrict statistics to land or ocean grid points only. "
            "Uses the Natural Earth 110m land mask via regionmask. "
            "Omit for full-domain statistics."
        ),
    )
    args = vars(parser.parse_args())

    domain = args["domain"]

    # Load model properties pkl if it exists; fall back to CLI args otherwise.
    pkl_path = "%s/model_%s/model_%s_properties.pkl" % (
        args["model_dir"], args["model_number"], args["model_number"]
    )
    if os.path.isfile(pkl_path):
        model_properties = pd.read_pickle(pkl_path)
        try:
            front_types = model_properties["dataset_properties"]["front_types"]
        except KeyError:
            front_types = model_properties["front_types"]
        num_front_types = model_properties["classes"] - 1
    else:
        model_properties = None
        if not args["front_types"]:
            parser.error(
                "--front_types is required when no model properties pkl exists "
                "(expected at %s)" % pkl_path
            )
        front_types = args["front_types"]
        num_front_types = len(front_types)

    if args["dataset"] is not None and args["year_and_month"] is not None:
        raise ValueError("--dataset and --year_and_month cannot be passed together.")
    elif args["dataset"] is None and args["year_and_month"] is None:
        if args["fronts_file"] is not None:
            # Derive years directly from the fronts file
            _ds = xr.open_dataset(args["fronts_file"])
            years = sorted(set(pd.DatetimeIndex(_ds.time.values).year.tolist()))
            _ds.close()
            months = range(1, 13)
        else:
            raise ValueError(
                "Provide --dataset, --year_and_month, or --fronts_file "
                "(years will be derived from the fronts file automatically)."
            )
    elif args["year_and_month"] is not None:
        years, months = [args["year_and_month"][0]], [args["year_and_month"][1]]
    else:
        if model_properties is None:
            raise ValueError(
                "--dataset requires a model properties pkl. "
                "Use --year_and_month or --fronts_file instead."
            )
        years, months = model_properties["%s_years" % args["dataset"]], range(1, 13)

    if args["gpu_device"] is not None:
        gpus = tf.config.list_physical_devices(device_type="GPU")
        tf.config.set_visible_devices(
            devices=[gpus[gpu] for gpu in args["gpu_device"]], device_type="GPU"
        )

        # Allow for memory growth on the GPU. This will only use the GPU memory that is required rather than allocating all the GPU's memory.
        if args["memory_growth"]:
            tf.config.experimental.set_memory_growth(
                device=[gpus[gpu] for gpu in args["gpu_device"]][0], enable=True
            )

    # ------------------------------------------------------------------
    # Pre-compute the spatial land/ocean mask once (same grid for all months).
    # We defer actual masking until lats/lons are known from the first file.
    # ------------------------------------------------------------------
    spatial_mask_2d: np.ndarray | None = None  # shape (Nlat, Nlon), True = include

    for year in years:
        for month in months:
            # ------------------------------------------------------------------
            # Load front truth labels — either from a single NetCDF file or the
            # legacy pkl-based monthly file list.
            # ------------------------------------------------------------------
            if args["fronts_file"] is not None:
                fronts_ds = xr.open_dataset(args["fronts_file"])
                time_sel = "%d-%02d" % (year, month)
                fronts_ds_month = data_utils.reformat_fronts(
                    fronts_ds.sel(time=time_sel), front_types
                )
            else:
                front_files_month = pd.read_pickle(
                    "%s/front_files_%d%02d.pkl" % (args["tf_indir"], year, month)
                )

                if domain != "conus":
                    for front_file in front_files_month[::-1]:
                        if any(
                            [
                                "%02d_full.nc" % hour in front_file
                                for hour in np.arange(3, 21.1, 6)
                            ]
                        ):
                            front_files_month.pop(front_files_month.index(front_file))

                custom_extent = (
                    model_properties["dataset_properties"].get("override_extent")
                    if model_properties is not None
                    else None
                )
                if custom_extent is not None:
                    slice_extent = dict(
                        longitude=slice(custom_extent[0], custom_extent[1]),
                        latitude=slice(custom_extent[3], custom_extent[2]),
                    )
                else:
                    slice_extent = dict(
                        longitude=slice(
                            DOMAIN_EXTENTS[args["domain"]][0],
                            DOMAIN_EXTENTS[args["domain"]][1],
                        ),
                        latitude=slice(
                            DOMAIN_EXTENTS[args["domain"]][3],
                            DOMAIN_EXTENTS[args["domain"]][2],
                        ),
                    )

                fronts_ds = xr.open_mfdataset(
                    front_files_month, combine="nested", concat_dim="time"
                ).sel(**slice_extent)
                fronts_ds_month = data_utils.reformat_fronts(
                    fronts_ds.sel(time="%d-%02d" % (year, month)), front_types
                )

            # ------------------------------------------------------------------
            # Resolve prediction file and output path.
            # Append the mask label to the stats filename when --mask is used so
            # full / land / ocean stats files don't overwrite each other.
            # ------------------------------------------------------------------
            prediction_file = (
                args["prediction_file"]
                if args["prediction_file"] is not None
                else "%s/model_%s/probabilities/model_%s_pred_%s_%d%02d.nc"
                % (
                    args["model_dir"],
                    args["model_number"],
                    args["model_number"],
                    args["domain"],
                    year,
                    month,
                )
            )

            mask_suffix = ("_%s" % args["mask"]) if args["mask"] else ""
            stats_dataset_path = (
                "%s/model_%s/statistics/model_%s_statistics_%s_%d%02d%s.nc"
                % (
                    args["model_dir"],
                    args["model_number"],
                    args["model_number"],
                    args["domain"],
                    year,
                    month,
                    mask_suffix,
                )
            )
            if os.path.isfile(stats_dataset_path) and not args["overwrite"]:
                print(
                    "WARNING: %s exists, pass the --overwrite argument to overwrite existing data."
                    % stats_dataset_path
                )
                continue

            os.makedirs(os.path.dirname(stats_dataset_path), exist_ok=True)

            probs_ds_full = xr.open_dataset(prediction_file)
            # Subset predictions to the current year-month, then align with
            # fronts on common timestamps (the two datasets may have different
            # temporal resolutions or slightly different time coverage).
            probs_ds_month = probs_ds_full.sel(time="%d-%02d" % (year, month))
            common_times = np.intersect1d(
                fronts_ds_month["time"].values, probs_ds_month["time"].values
            )
            if len(common_times) == 0:
                print(
                    "WARNING: no common timesteps for %d-%02d — skipping." % (year, month)
                )
                continue
            fronts_ds_month = fronts_ds_month.sel(time=common_times)
            probs_ds = probs_ds_month.sel(time=common_times)

            time_array = (
                common_times
                if args["fronts_file"] is not None
                else pd.read_pickle(
                    "%s/timesteps_%d%02d.pkl" % (args["tf_indir"], year, month)
                )
            )
            num_timesteps = len(time_array)
            lons = fronts_ds_month["longitude"].values
            lats = fronts_ds_month["latitude"].values
            Nlon = len(lons)
            Nlat = len(lats)

            # ------------------------------------------------------------------
            # Build the spatial mask (land or ocean) on the first iteration when
            # we have the actual lat/lon grid.  Reuse on subsequent iterations.
            # ------------------------------------------------------------------
            if args["mask"] is not None and spatial_mask_2d is None:
                land_regions = regionmask.defined_regions.natural_earth_v5_1_2.land_110
                raw_mask = land_regions.mask(lons, lats)  # NaN = ocean, 0 = land
                land_2d = ~np.isnan(raw_mask.values)  # True where land
                spatial_mask_2d = land_2d if args["mask"] == "land" else ~land_2d
                print(
                    "Spatial mask (%s): %d / %d grid points included."
                    % (args["mask"], spatial_mask_2d.sum(), spatial_mask_2d.size)
                )

            # Latitude weights; zero out masked points when --mask is active.
            lat_weights = np.cos(np.deg2rad(lats))[:, np.newaxis]  # (Nlat, 1)
            if spatial_mask_2d is not None:
                lat_weights = lat_weights * spatial_mask_2d.astype(float)  # broadcast
            weights = tf.cast(
                tf.convert_to_tensor(lat_weights[np.newaxis, :, :]),
                tf.float32,
            )  # shape (1, Nlat, Nlon) — latitude weights with mask applied

            tp_array_temporal = np.zeros(
                shape=[num_front_types, num_timesteps, 5, 100]
            ).astype("float32")
            fp_array_temporal = np.zeros(
                shape=[num_front_types, num_timesteps, 5, 100]
            ).astype("float32")
            tn_array_temporal = np.zeros(
                shape=[num_front_types, num_timesteps, 5, 100]
            ).astype("float32")
            fn_array_temporal = np.zeros(
                shape=[num_front_types, num_timesteps, 5, 100]
            ).astype("float32")
            tp_array_spatial = np.zeros(
                shape=[num_front_types, Nlat, Nlon, 5, 100]
            ).astype("float32")
            fp_array_spatial = np.zeros(
                shape=[num_front_types, Nlat, Nlon, 5, 100]
            ).astype("float32")
            tn_array_spatial = np.zeros(
                shape=[num_front_types, Nlat, Nlon, 5, 100]
            ).astype("float32")
            fn_array_spatial = np.zeros(
                shape=[num_front_types, Nlat, Nlon, 5, 100]
            ).astype("float32")

            thresholds = np.linspace(
                0.01, 1, 100
            )  # Probability thresholds for calculating performance statistics
            neighborhoods = np.array(
                [50, 100, 150, 200, 250]
            )  # neighborhoods for checking whether a front is present (kilometers)

            bool_tn_fn_dss = dict(
                {
                    front: tf.convert_to_tensor(
                        xr.where(fronts_ds_month == front_no + 1, 1, 0)[
                            "identifier"
                        ].values
                    )
                    for front_no, front in enumerate(front_types)
                }
            )
            bool_tp_fp_dss = dict({front: None for front in front_types})
            probs_dss = dict(
                {
                    front: tf.convert_to_tensor(probs_ds[front].values)
                    for front in front_types
                }
            )

            spatial_ds = xr.Dataset(
                coords={
                    "time": time_array[0],
                    "latitude": lats,
                    "longitude": lons,
                    "neighborhood": neighborhoods,
                    "threshold": thresholds,
                }
            )
            temporal_ds = xr.Dataset(
                coords={
                    "time": time_array,
                    "neighborhood": neighborhoods,
                    "threshold": thresholds,
                }
            )

            for front_no, front_type in enumerate(front_types):
                if args["fronts_file"] is not None:
                    fronts_ds_month = data_utils.reformat_fronts(
                        xr.open_dataset(args["fronts_file"]).sel(
                            time="%d-%02d" % (year, month)
                        ),
                        front_types,
                    )
                else:
                    fronts_ds_month = data_utils.reformat_fronts(
                        fronts_ds.sel(time="%d-%02d" % (year, month)), front_types
                    )
                print("%d-%02d: %s (TN/FN)" % (year, month, front_type))
                ### Calculate true/false negatives ###
                for i in range(100):
                    """
                    True negative ==> model correctly predicts the lack of a front at a given point
                    False negative ==> model does not predict a front, but a front exists
                    
                    The numbers of true negatives and false negatives are the same for all neighborhoods and are calculated WITHOUT expanding the fronts.
                    If we were to calculate the negatives separately for each neighborhood, the number of misses would be artificially inflated, lowering the
                    final CSI scores and making the neighborhood method effectively useless.
                    """
                    tn = (
                        tf.cast(
                            tf.where(
                                (probs_dss[front_type] < thresholds[i])
                                & (bool_tn_fn_dss[front_type] == 0),
                                1,
                                0,
                            ),
                            tf.float32,
                        )
                        * weights
                    )
                    fn = (
                        tf.cast(
                            tf.where(
                                (probs_dss[front_type] < thresholds[i])
                                & (bool_tn_fn_dss[front_type] == 1),
                                1,
                                0,
                            ),
                            tf.float32,
                        )
                        * weights
                    )

                    tn_array_spatial[front_no, :, :, :, i] = tf.tile(
                        tf.expand_dims(tf.reduce_sum(tn, axis=0), axis=-1), (1, 1, 5)
                    )
                    fn_array_spatial[front_no, :, :, :, i] = tf.tile(
                        tf.expand_dims(tf.reduce_sum(fn, axis=0), axis=-1), (1, 1, 5)
                    )
                    tn_array_temporal[front_no, :, :, i] = tf.tile(
                        tf.expand_dims(tf.reduce_sum(tn, axis=(1, 2)), axis=-1), (1, 5)
                    )
                    fn_array_temporal[front_no, :, :, i] = tf.tile(
                        tf.expand_dims(tf.reduce_sum(fn, axis=(1, 2)), axis=-1), (1, 5)
                    )

                ### Calculate true/false positives ###
                for neighborhood in range(5):
                    fronts_ds_month = data_utils.expand_fronts(
                        fronts_ds_month, iterations=2
                    )  # Expand fronts by 50km
                    bool_tp_fp_dss[front_type] = tf.convert_to_tensor(
                        xr.where(fronts_ds_month == front_no + 1, 1, 0)[
                            "identifier"
                        ].values
                    )  # 1 = cold front, 0 = not a cold front
                    print(
                        "%d-%02d: %s (%d km)"
                        % (year, month, front_type, (neighborhood + 1) * 50)
                    )
                    for i in range(100):
                        """
                        True positive ==> model correctly identifies a front
                        False positive ==> model predicts a front, but no front is present within the given neighborhood
                        """
                        tp = (
                            tf.cast(
                                tf.where(
                                    (probs_dss[front_type] > thresholds[i])
                                    & (bool_tp_fp_dss[front_type] == 1),
                                    1,
                                    0,
                                ),
                                tf.float32,
                            )
                            * weights
                        )
                        fp = (
                            tf.cast(
                                tf.where(
                                    (probs_dss[front_type] > thresholds[i])
                                    & (bool_tp_fp_dss[front_type] == 0),
                                    1,
                                    0,
                                ),
                                tf.float32,
                            )
                            * weights
                        )

                        tp_array_spatial[front_no, :, :, neighborhood, i] = (
                            tf.reduce_sum(tp, axis=0)
                        )
                        fp_array_spatial[front_no, :, :, neighborhood, i] = (
                            tf.reduce_sum(fp, axis=0)
                        )
                        tp_array_temporal[front_no, :, neighborhood, i] = tf.reduce_sum(
                            tp, axis=(1, 2)
                        )
                        fp_array_temporal[front_no, :, neighborhood, i] = tf.reduce_sum(
                            fp, axis=(1, 2)
                        )

                spatial_ds["tp_spatial_%s" % front_type] = (
                    ("latitude", "longitude", "neighborhood", "threshold"),
                    tp_array_spatial[front_no],
                )
                spatial_ds["fp_spatial_%s" % front_type] = (
                    ("latitude", "longitude", "neighborhood", "threshold"),
                    fp_array_spatial[front_no],
                )
                spatial_ds["tn_spatial_%s" % front_type] = (
                    ("latitude", "longitude", "neighborhood", "threshold"),
                    tn_array_spatial[front_no],
                )
                spatial_ds["fn_spatial_%s" % front_type] = (
                    ("latitude", "longitude", "neighborhood", "threshold"),
                    fn_array_spatial[front_no],
                )
                temporal_ds["tp_temporal_%s" % front_type] = (
                    ("time", "neighborhood", "threshold"),
                    tp_array_temporal[front_no],
                )
                temporal_ds["fp_temporal_%s" % front_type] = (
                    ("time", "neighborhood", "threshold"),
                    fp_array_temporal[front_no],
                )
                temporal_ds["tn_temporal_%s" % front_type] = (
                    ("time", "neighborhood", "threshold"),
                    tn_array_temporal[front_no],
                )
                temporal_ds["fn_temporal_%s" % front_type] = (
                    ("time", "neighborhood", "threshold"),
                    fn_array_temporal[front_no],
                )

            spatial_ds.astype("float32").to_netcdf(
                path=stats_dataset_path.replace(".nc", "_spatial.nc"),
                mode="w",
                engine="netcdf4",
            )
            spatial_ds.close()
            temporal_ds.to_netcdf(
                path=stats_dataset_path.replace(".nc", "_temporal.nc"),
                mode="w",
                engine="netcdf4",
            )
            temporal_ds.close()
