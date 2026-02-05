"""
Download ERA5 data from the Climate Data Store.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2025.5.3
"""

import os
import cdsapi
import argparse
import numpy as np

VAR_ABBREV = {
    "temperature": "T",
    "geopotential": "Z",
    "specific_humidity": "q",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
}

DATA_FORMATS = {"grib": "grib", "netcdf": "nc"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="reanalysis-era5-pressure-levels"
    )
    parser.add_argument(
        "--product_type", type=str, default="reanalysis", help="CDS product type."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Main output directory for the downloaded files.",
    )
    parser.add_argument(
        "--pressure_level", type=str, required=True, help="Pressure levels of interest."
    )
    parser.add_argument(
        "--variable",
        type=str,
        required=True,
        help="Variable of interest. Must follow CDS nomenclature.",
    )
    parser.add_argument("--year", type=int, required=True, help="Year of interest.")
    parser.add_argument(
        "--month",
        type=int,
        nargs="+",
        default=np.arange(1, 12.1).astype(int),
        help="Months of interest.",
    )
    parser.add_argument(
        "--day",
        type=int,
        nargs="+",
        default=np.arange(0, 31.1).astype(int),
        help="Days of interest.",
    )
    parser.add_argument(
        "--hour",
        type=int,
        nargs="+",
        default=np.arange(0, 21.1, 3).astype(int),
        help="Hours of interest.",
    )
    parser.add_argument("--data_format", type=str, default="grib", help="Data format.")
    parser.add_argument(
        "--download_format", type=str, default="unarchived", help="Download format."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite any existing ERA5 files."
    )
    args = vars(parser.parse_args())

    target = "%s/era5_%s-%shPa_%d.%s" % (
        args["outdir"],
        args["variable"],
        args["pressure_level"],
        args["year"],
        DATA_FORMATS[args["data_format"]],
    )

    if not os.path.isfile(target) or args["overwrite"]:
        request = {
            "product_type": [args["product_type"]],
            "variable": [args["variable"]],
            "year": [args["year"]],
            "month": [f"{mo:02d}" for mo in args["month"]],
            "day": [f"{dy:02d}" for dy in args["day"]],
            "time": [f"{hr:02d}:00" % hr for hr in args["hour"]],
            "pressure_level": [args["pressure_level"]],
            "data_format": args["data_format"],
            "download_format": args["download_format"],
        }

        os.makedirs(args["outdir"], exist_ok=True)

        client = cdsapi.Client()
        client.retrieve(args["dataset"], request).download(target=target)

    else:
        print(
            f"{target} already exists. Pass the --overwrite flag to initiate a new download that overwrites the file."
        )
