"""
Convert netCDF files containing variable and frontal boundary data into tensorflow datasets for model evaluation.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2025.5.3
"""

import argparse
import itertools
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
import file_manager as fm
from utils import data_utils, misc
from datetime import datetime
import xarray as xr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variables_indir",
        type=str,
        required=True,
        help="Input directory for the netCDF files containing variable data.",
    )
    parser.add_argument(
        "--fronts_indir",
        type=str,
        help="Input directory for the netCDF files containing frontal boundary data.",
    )
    parser.add_argument(
        "--goes_indir",
        type=str,
        help="Input directory for the netCDF files containing GOES satellite data.",
    )
    parser.add_argument(
        "--tf_outdir",
        type=str,
        required=True,
        help="Output directory for the generated tensorflow datasets.",
    )
    parser.add_argument(
        "--year_and_month",
        type=int,
        nargs=2,
        required=True,
        help="Year and month for the netcdf data to be converted to tensorflow datasets.",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="era5",
        help="Data source or model containing the variable data.",
    )
    parser.add_argument(
        "--front_types",
        type=str,
        nargs="+",
        help="Code(s) for the front types that will be generated in the tensorflow datasets. Refer to documentation in "
        "'utils.data_utils.reformat_fronts' for more information on these codes.",
    )
    parser.add_argument(
        "--variables", type=str, nargs="+", required=True, help="Variables to select"
    )
    parser.add_argument(
        "--pressure_levels",
        type=str,
        nargs="+",
        help="Variables pressure levels to select",
    )
    parser.add_argument(
        "--num_dims",
        type=int,
        nargs=2,
        default=[2, 2],
        help="Number of dimensions in the variables and front object images, repsectively.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="conus",
        help="Domain from which to pull the images.",
    )
    parser.add_argument(
        "--override_extent",
        type=float,
        nargs=4,
        help="Override the default domain extent by selecting a custom extent. [min lon, max lon, min lat, max lat]",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[128, 128],
        help="Size of the longitude and latitude dimensions of the images.",
    )
    parser.add_argument(
        "--normalization_method",
        type=str,
        default="standard",
        help="Method for normalizing the datasets. Options are 'standard', 'standard_weighted', 'min-max'.",
    )
    parser.add_argument(
        "--front_dilation",
        type=int,
        default=0,
        help="Number of pixels to expand the fronts by in all directions.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print out the progress of the dataset generation.",
    )
    parser.add_argument("--gpu_device", type=int, nargs="+", help="GPU device numbers.")

    args = vars(parser.parse_args())

    """
    It is recommended to run this script on a GPU due to the abundance of TensorFlow operations.
    """
    if args["gpu_device"] is not None:
        misc.initialize_gpus(
            args["gpu_device"], memory_growth=True
        )  # initialize the specified GPU
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    year, month = args["year_and_month"]

    """
    all_data_vars: Variables found in ERA5 reanalysis and NWP models
    all_goes_vars: Satellite bands found in the GOES datasets after the merging process
    """
    all_data_vars = [
        "T",
        "Td",
        "sp_z",
        "u",
        "v",
        "r",
        "RH",
        "Tv",
        "theta_e",
        "q",
        "theta",
        "theta_v",
    ]
    all_pressure_levels = (
        ["surface", "1000", "950", "900", "850"]
        if args["data_source"] == "era5"
        else ["surface", "1013", "1000", "950", "900", "850", "700", "500"]
    )

    # check for invalid variables
    invalid_vars = [var for var in args["variables"] if var not in all_data_vars]
    assert len(invalid_vars) == 0, "Invalid variables (%d): %s" % (
        len(invalid_vars),
        ", ".join(invalid_vars),
    )

    data_vars = [
        var for var in args["variables"] if var in all_data_vars
    ]  # ERA5/model variables that will be used

    os.makedirs(
        args["tf_outdir"], exist_ok=True
    )  # ensure that a folder exists for the monthly dataset

    tf_dataset_folder_inputs = f"%s/inputs_%d%02d_tf" % (
        args["tf_outdir"],
        year,
        month,
    )  # output directory for the inputs
    tf_dataset_folder_fronts = tf_dataset_folder_inputs.replace(
        "inputs", "fronts"
    )  # output directory for the front labels

    # ensure that the requested month does not already have a saved dataset
    if os.path.isdir(tf_dataset_folder_inputs) or os.path.isdir(
        tf_dataset_folder_fronts
    ):
        raise FileExistsError(
            "Tensorflow dataset(s) already exist for the provided year and month."
        )

    # dataset properties file - this will contain critical information about the dataset
    dataset_props_file = "%s/dataset_properties.pkl" % args["tf_outdir"]
    if not os.path.isfile(dataset_props_file):
        dataset_props = dict({})
        dataset_props["normalization_parameters"] = data_utils.NORMALIZATION_PARAMS
        for key in sorted(
            [
                "front_types",
                "variables",
                "pressure_levels",
                "num_dims",
                "image_size",
                "normalization_method",
                "front_dilation",
                "domain",
                "override_extent",
            ]
        ):
            dataset_props[key] = args[key]

        # save out the dataset properties pickle file
        with open(dataset_props_file, "wb") as f:
            pickle.dump(dataset_props, f)

        # create a text file with a human-readable output of the information saved in the dataset properties pickle file
        with open("%s/dataset_properties.txt" % args["tf_outdir"], "w") as f:
            for key in sorted(dataset_props.keys()):
                f.write(f"{key}: {dataset_props[key]}\n")
            f.write(f"\n\n\nFile generated at {datetime.utcnow()} UTC\n")

    else:
        """
        If the dataset properties pickle file already exists, many arguments declared at the command line will be overwritten
        by the values contained within the pickle file. This behavior exists to ensure that the dataset does not have inconsistent
        properties relegated to certain months.
        """
        print(
            "WARNING: Dataset properties file was found in %s. The following settings will be used from the file."
            % args["tf_outdir"]
        )
        dataset_props = pd.read_pickle(dataset_props_file)

        for key in sorted(
            [
                "front_types",
                "variables",
                "pressure_levels",
                "num_dims",
                "image_size",
                "normalization_method",
                "front_dilation",
                "domain",
                "override_extent",
            ]
        ):
            args[key] = dataset_props[key]
            print(f"%s: {args[key]}" % key)

    file_loader_domain = "global" if args["data_source"] == "era5" else args["domain"]

    # Gather all ERA5/model and front label files that can be used to generate the dataset for the current month.
    file_obj = fm.DataFileLoader(
        args["variables_indir"],
        args["data_source"],
        "netcdf",
        years=year,
        months=month,
        domains=file_loader_domain,
    )
    file_obj.add_file_list(args["fronts_indir"], "fronts", ignore_domain=True)
    variables_netcdf_files, fronts_netcdf_files = file_obj.files

    # if not looking over CONUS or the HRRR domain, remove non-synoptic hours (3, 9, 15, 21z)
    if args["domain"] not in ["conus", "hrrr"]:
        synoptic_ind = [
            variables_netcdf_files.index(file)
            for file in variables_netcdf_files
            if any(["%02d_" % hr in file for hr in [0, 6, 12, 18]])
        ]
        variables_netcdf_files = list([variables_netcdf_files[i] for i in synoptic_ind])
        fronts_netcdf_files = list([fronts_netcdf_files[i] for i in synoptic_ind])

    # if the extent crosses the Prime Meridian (0 degrees longitude), we need to load the data in differently
    extent_crosses_meridian = False
    if args["override_extent"] is not None:
        if args["override_extent"][1] > 360:
            extent_crosses_meridian = True

    if args["domain"] in ["conus", "full", "goes-merged"]:
        if args["override_extent"] is None:
            sel_kwargs = {
                "latitude": slice(
                    data_utils.DOMAIN_EXTENTS[args["domain"]][3],
                    data_utils.DOMAIN_EXTENTS[args["domain"]][2],
                ),
                "longitude": slice(
                    data_utils.DOMAIN_EXTENTS[args["domain"]][0],
                    data_utils.DOMAIN_EXTENTS[args["domain"]][1],
                ),
            }
        else:
            sel_kwargs = {
                "latitude": slice(
                    args["override_extent"][3], args["override_extent"][2]
                ),
                "longitude": slice(
                    args["override_extent"][0], args["override_extent"][1]
                ),
            }
    else:
        sel_kwargs = {}

    args["pressure_levels"] = (
        all_pressure_levels
        if args["pressure_levels"] is None
        else [lvl for lvl in all_pressure_levels if lvl in args["pressure_levels"]]
    )

    num_timesteps = len(variables_netcdf_files)
    images_kept = 0
    images_discarded = 0
    timesteps_kept = 0
    timesteps_discarded = 0

    isel_kwargs = dict(forecast_hour=0) if args["data_source"] != "era5" else dict()

    """
    In order to make sure that the final dataset comes out clean, we will keep track of all of the input shapes of
    the generated images as they are being generated. This will allow us to catch any indexing error that may produce
    an image shape that is different from the rest of the dataset (e.g., say an indexing error produces an image of
    size 128x127 when we intended to have its shape be 128x128). If the tensor shapes are not all identical, TensorFlow
    will raise an error before model training and render the dataset effectively useless.
    """
    input_tensor_shapes = []
    front_tensor_shapes = []

    front_files_kept = []

    for timestep_no in range(num_timesteps):
        # open front dataset
        front_dataset = xr.open_dataset(
            fronts_netcdf_files[timestep_no], engine="netcdf4"
        ).isel(**isel_kwargs)

        if args["data_source"] not in ["hrrr", "namnest-conus", "nam-12km"]:
            front_dataset = front_dataset.sel(**sel_kwargs).astype("float16")
            transpose_dims = (
                "latitude",
                "longitude",
            )  # spatial dimensions that need to be transposed
        else:
            transpose_dims = ("y", "x")  # spatial dimensions that need to be transposed
        domain_size = (
            len(front_dataset[transpose_dims[0]]),
            len(front_dataset[transpose_dims[1]]),
        )

        # Reformat the fronts in the current timestep
        if args["front_types"] is not None:
            front_dataset = data_utils.reformat_fronts(
                front_dataset, args["front_types"]
            )
            num_front_types = front_dataset.attrs["num_front_types"] + 1
        else:
            num_front_types = 16

        # Expand the front labels
        if args["front_dilation"] > 0:
            front_dataset = data_utils.expand_fronts(
                front_dataset, iterations=args["front_dilation"]
            )

        # Check for all front types in the dataset
        front_dataset = (
            front_dataset.isel(time=0)
            if "time" in front_dataset.dims
            else front_dataset
        )
        front_dataset = front_dataset.to_array().transpose(*transpose_dims, "variable")

        if args["verbose"]:
            print(
                "%d-%02d Dataset progress (kept/discarded):  (%d/%d timesteps, %d/%d images)"
                % (
                    year,
                    month,
                    timesteps_kept,
                    timesteps_discarded,
                    images_kept,
                    images_discarded,
                ),
                end="\r",
            )

            # open variables dataset
            if extent_crosses_meridian:
                sel_kwargs_1 = {
                    "latitude": slice(
                        args["override_extent"][3], args["override_extent"][2]
                    ),
                    "longitude": slice(args["override_extent"][0], 360),
                }  # extent west of the Prime Meridian
                sel_kwargs_2 = {
                    "latitude": slice(
                        args["override_extent"][3], args["override_extent"][2]
                    ),
                    "longitude": slice(0, args["override_extent"][1] - 360),
                }  # extent east of the Prime Meridian

                variables_dataset_1 = (
                    xr.open_dataset(
                        variables_netcdf_files[timestep_no], engine="netcdf4"
                    )[data_vars]
                    .sel(pressure_level=args["pressure_levels"], **sel_kwargs_1)
                    .isel(**isel_kwargs)
                    .transpose("time", *transpose_dims, "pressure_level")
                    .astype("float16")
                )
                variables_dataset_2 = (
                    xr.open_dataset(
                        variables_netcdf_files[timestep_no], engine="netcdf4"
                    )[data_vars]
                    .sel(pressure_level=args["pressure_levels"], **sel_kwargs_2)
                    .isel(**isel_kwargs)
                    .transpose("time", *transpose_dims, "pressure_level")
                    .astype("float16")
                )
                variables_dataset = xr.merge([variables_dataset_1, variables_dataset_2])

            else:
                variables_dataset = (
                    xr.open_dataset(
                        variables_netcdf_files[timestep_no], engine="netcdf4"
                    )[data_vars]
                    .sel(pressure_level=args["pressure_levels"], **sel_kwargs)
                    .isel(**isel_kwargs)
                    .transpose("time", *transpose_dims, "pressure_level")
                    .astype("float16")
                )
            variables_dataset = (
                variables_dataset.isel(time=0)
                .transpose(*transpose_dims, "pressure_level")
                .astype("float16")
            )

            # create a list of starting indices along the latitude dimension
            start_indices_lat = [
                0,
            ]
            start_indices_lon = [
                0,
            ]

            image_order = list(
                itertools.product(start_indices_lat, start_indices_lon)
            )  # Every possible combination of longitude and latitude starting points

            for i, image_start_indices in enumerate(image_order):
                if args["verbose"]:
                    print(
                        "%d-%02d Dataset progress (kept/discarded):  (%d/%d timesteps, %d/%d images)"
                        % (
                            year,
                            month,
                            timesteps_kept,
                            timesteps_discarded,
                            images_kept,
                            images_discarded,
                        ),
                        end="\r",
                    )

                start_index_lat = image_start_indices[0]
                end_index_lat = start_index_lat + args["image_size"][0]
                start_index_lon = image_start_indices[1]
                end_index_lon = start_index_lon + args["image_size"][1]

                front_image = front_dataset[
                    start_index_lat:end_index_lat, start_index_lon:end_index_lon, :
                ]

                new_variables_dataset = (
                    variables_dataset.copy()
                )  # copy variables dataset to isolate dataset in memory

                # normalize variables and convert dataset to a tensor
                new_variables_dataset = (
                    data_utils.normalize_dataset(
                        new_variables_dataset,
                        args["normalization_method"],
                        dataset_props["normalization_parameters"],
                    )
                    .to_array()
                    .transpose(*transpose_dims, "pressure_level", "variable")
                )
                input_tensor = tf.convert_to_tensor(
                    np.nan_to_num(
                        new_variables_dataset[
                            start_index_lat:end_index_lat,
                            start_index_lon:end_index_lon,
                            :,
                            :,
                        ]
                    ),
                    dtype=tf.float16,
                )

                # combine pressure level and variables dimensions, making the images 2D (excluding the final dimension)
                if args["num_dims"][0] == 2:
                    input_tensor_shape_3d = input_tensor.shape
                    input_tensor = tf.reshape(
                        input_tensor,
                        [
                            input_tensor_shape_3d[0],
                            input_tensor_shape_3d[1],
                            input_tensor_shape_3d[2] * input_tensor_shape_3d[3],
                        ],
                    )

                input_tensor_shapes.append(input_tensor.shape)
                assert len(set(input_tensor_shapes)) == 1, (
                    f"ERROR: Attempted to add {input_tensor_shapes[-1]} to dataset with shape {input_tensor_shapes[0]}. "
                    "Please check your data for inconsistent coordinate systems."
                )

                # add input images to tensorflow dataset
                input_tensor_for_timestep = tf.data.Dataset.from_tensors(input_tensor)
                if "input_tensors_for_month" not in locals():
                    input_tensors_for_month = input_tensor_for_timestep
                else:
                    input_tensors_for_month = input_tensors_for_month.concatenate(
                        input_tensor_for_timestep
                    )

                front_tensor = tf.convert_to_tensor(
                    np.nan_to_num(front_image), dtype=tf.int32
                )
                front_tensor_shapes.append(front_tensor.shape)

                assert len(set(front_tensor_shapes)) == 1, (
                    f"ERROR: Attempted to add {front_tensor_shapes[-1]} to dataset with shape {front_tensor_shapes[0]}. "
                    "Please check your data for inconsistent coordinate systems."
                )

                # if using 3D inputs, turn the fronts dataset into a 3D image
                if args["num_dims"][1] == 3:
                    front_tensor = tf.tile(
                        front_tensor, (1, 1, len(args["pressure_levels"]))
                    )
                else:
                    front_tensor = front_tensor[:, :, 0]

                front_tensor = tf.cast(
                    tf.one_hot(front_tensor, num_front_types), tf.float16
                )  # One-hot encode the labels
                front_tensor_for_timestep = tf.data.Dataset.from_tensors(
                    front_tensor
                )  # convert fronts into a tensorflow dataset
                if "front_tensors_for_month" not in locals():
                    front_tensors_for_month = front_tensor_for_timestep
                else:
                    front_tensors_for_month = front_tensors_for_month.concatenate(
                        front_tensor_for_timestep
                    )

            timesteps_kept += 1
            front_files_kept.append(fronts_netcdf_files[timestep_no])
        else:
            timesteps_discarded += 1

    print(
        "%d-%02d Dataset progress (kept/discarded):  (%d/%d timesteps, %d/%d images)"
        % (
            year,
            month,
            timesteps_kept,
            timesteps_discarded,
            images_kept,
            images_discarded,
        )
    )

    # save the tensorflow datasets
    try:
        tf.data.Dataset.save(input_tensors_for_month, path=tf_dataset_folder_inputs)
        tf.data.Dataset.save(front_tensors_for_month, path=tf_dataset_folder_fronts)
        print(
            "Tensorflow datasets for %d-%02d saved to %s."
            % (year, month, args["tf_outdir"])
        )
    except NameError:
        print("No images could be retained with the provided arguments.")

    with open(
        "%s/front_files_%d%02d.pkl" % (args["tf_outdir"], year, month), "wb"
    ) as f:
        pickle.dump(np.array(front_files_kept), f)
