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
from fronts.utils import data_utils
from fronts.utils.data_utils import DOMAIN_EXTENTS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Directory for the models."
    )
    parser.add_argument("--model_number", type=int, required=True, help="Model number.")
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
    args = vars(parser.parse_args())

    model_properties = pd.read_pickle(
        "%s/model_%d/model_%d_properties.pkl"
        % (args["model_dir"], args["model_number"], args["model_number"])
    )
    domain = args["domain"]

    variables = model_properties["dataset_properties"]["variables"]

    # Some older models do not have the 'dataset_properties' dictionary
    try:
        front_types = model_properties["dataset_properties"]["front_types"]
        num_dims = model_properties["dataset_properties"]["num_dims"]
    except KeyError:
        front_types = model_properties["front_types"]
        if args["model_number"] in [6846496, 7236500, 7507525]:
            num_dims = (3, 3)

    num_front_types = (
        model_properties["classes"] - 1
    )  # remove the "no front" class type

    if args["dataset"] is not None and args["year_and_month"] is not None:
        raise ValueError("--dataset and --year_and_month cannot be passed together.")
    elif args["dataset"] is None and args["year_and_month"] is None:
        raise ValueError(
            "At least one of [--dataset, --year_and_month] must be passed."
        )
    elif args["year_and_month"] is not None:
        years, months = [args["year_and_month"][0]], [args["year_and_month"][1]]
    else:
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

    for year in years:
        for month in months:
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

            prediction_file = (
                f"%s/model_%d/probabilities/model_%d_pred_%s_%d%02d.nc"
                % (
                    args["model_dir"],
                    args["model_number"],
                    args["model_number"],
                    args["domain"],
                    year,
                    month,
                )
            )

            stats_dataset_path = (
                "%s/model_%d/statistics/model_%d_statistics_%s_%d%02d.nc"
                % (
                    args["model_dir"],
                    args["model_number"],
                    args["model_number"],
                    args["domain"],
                    year,
                    month,
                )
            )
            if os.path.isfile(stats_dataset_path) and not args["overwrite"]:
                print(
                    "WARNING: %s exists, pass the --overwrite argument to overwrite existing data."
                    % stats_dataset_path
                )
                continue

            probs_ds = xr.open_dataset(prediction_file)
            lons = probs_ds["longitude"].values
            lats = probs_ds["latitude"].values

            try:
                custom_extent = model_properties["dataset_properties"][
                    "override_extent"
                ]
                slice_extent = dict(
                    longitude=slice(custom_extent[0], custom_extent[1]),
                    latitude=slice(custom_extent[3], custom_extent[2]),
                )
            except KeyError:
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
            else:
                if model_properties["dataset_properties"]["override_extent"] is None:
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

            time_array = pd.read_pickle(
                "%s/timesteps_%d%02d.pkl" % (args["tf_indir"], year, month)
            )
            num_timesteps = len(time_array)
            lons = fronts_ds_month["longitude"].values
            lats = fronts_ds_month["latitude"].values
            Nlon = len(lons)
            Nlat = len(lats)

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
            weights = tf.cast(
                tf.convert_to_tensor(
                    np.cos(np.deg2rad(lats))[np.newaxis, :, np.newaxis]
                ),
                tf.float32,
            )  # latitude weights for the statistics

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
