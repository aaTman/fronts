"""
Transform ERA5 pressure level GRIB files into netCDF files.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2025.5.3
"""

import datetime as dt
import glob
import os
import numpy as np
import xarray as xr
import argparse
from fronts.utils import variables
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        required=True,
        help="Input directory for the ERA5 GRIB files.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for the ERA5 netCDF files.",
    )
    parser.add_argument(
        "--date", type=str, required=True, help="Date with format YYYY-MM-DD."
    )
    parser.add_argument(
        "--parallel_threads",
        type=int,
        default=4,
        help="Number of threads to use during computations in parallel.",
    )
    args = vars(parser.parse_args())

    yr, mo, dy = args["date"].split("-")

    pl_files = []
    for var in [
        "geopotential",
        "temperature",
        "specific_humidity",
        "u_component_of_wind",
        "v_component_of_wind",
    ]:
        pl_files.append(
            list(sorted(glob.glob("%s/era5_%s*hPa*%s.nc" % (args["indir"], var, yr))))
        )

    chunks = {"time": 1, "pressure_level": 7, "longitude": 720, "latitude": 721}

    print(f"{dt.datetime.utcnow()}: Opening datasets")
    era5_z = xr.open_mfdataset(
        pl_files[0], engine="h5netcdf", chunks="auto", parallel=True
    ).sel(valid_time=args["date"])
    era5_t = xr.open_mfdataset(
        pl_files[1], engine="h5netcdf", chunks="auto", parallel=True
    ).sel(valid_time=args["date"])
    era5_q = xr.open_mfdataset(
        pl_files[2], engine="h5netcdf", chunks="auto", parallel=True
    ).sel(valid_time=args["date"])
    era5_u = xr.open_mfdataset(
        pl_files[3], engine="h5netcdf", chunks="auto", parallel=True
    ).sel(valid_time=args["date"])
    era5_v = xr.open_mfdataset(
        pl_files[4], engine="h5netcdf", chunks="auto", parallel=True
    ).sel(valid_time=args["date"])
    print(f"{dt.datetime.utcnow()}: Merging datasets (this may take a while)")
    era5_pl = xr.merge([era5_z, era5_t, era5_q, era5_u, era5_v]).compute(
        num_workers=args["parallel_threads"], scheduler="processes"
    )

    dims = list(era5_pl.coords.dims.mapping.keys())
    keys = list(era5_pl.keys())

    # variable and coordinate check
    assert "t" in keys, "Temperature (t) is missing from the provided dataset."
    assert "q" in keys, "Specific humidity (q) is missing from the provided dataset."
    assert "z" in keys, "Geopotential (z) is missing from the provided dataset."
    assert "u" in keys, "U-wind (u) is missing from the provided dataset."
    assert "v" in keys, "V-wind (v) is missing from the provided dataset."
    assert "pressure_level" in dims, (
        "Pressure level (pressure_level) is missing from the provided dataset."
    )

    # remove attributes unnecessary coordinates
    era5_pl = era5_pl.drop_attrs()
    era5_pl = era5_pl.drop_vars(["number", "expver"])

    # rename temperature and geopotential
    era5_pl = era5_pl.rename({"t": "T", "z": "sp_z", "valid_time": "time"})

    # convert geopotential to geopotential height (dam)
    era5_pl["sp_z"] /= 98.0665

    coords = ("time", "pressure_level", "latitude", "longitude")

    Ntime, Npl, Nlat, Nlon = era5_pl[
        "T"
    ].shape  # (time, pressure_level, latitude, longitude)
    pressure_levels = era5_pl[
        "pressure_level"
    ].values  # (time, pressure_level, latitude, longitude)

    P = np.full(
        (Ntime, Npl, Nlat, Nlon), pressure_levels[np.newaxis, :, np.newaxis, np.newaxis]
    )

    print(f"{dt.datetime.utcnow()}: Adding variable: Dewpoint temperature")
    era5_pl["Td"] = (
        coords,
        variables.dewpoint_from_specific_humidity(P * 100, era5_pl["q"].data),
    )
    era5_pl["Td"] = era5_pl["Td"].assign_attrs(
        long_name="Dewpoint Temperature", units="K"
    )

    print(f"{dt.datetime.utcnow()}: Adding variable: Virtual temperature")
    era5_pl["Tv"] = (
        coords,
        variables.virtual_temperature_from_dewpoint(
            P * 100, era5_pl["T"].data, era5_pl["Td"].data
        ),
    )
    era5_pl["Tv"] = era5_pl["Tv"].assign_attrs(
        long_name="Virtual Temperature", units="K"
    )

    print(f"{dt.datetime.utcnow()}: Adding variable: Relative humidity")
    era5_pl["RH"] = (
        coords,
        variables.relative_humidity_from_dewpoint(
            era5_pl["T"].data, era5_pl["Td"].data
        ),
    )
    era5_pl["RH"] = era5_pl["RH"].assign_attrs(
        long_name="Relative Humidity", units="dimensionless"
    )

    print(f"{dt.datetime.utcnow()}: Adding variable: Potential temperature")
    era5_pl["theta"] = (
        coords,
        variables.potential_temperature(P * 100, era5_pl["T"].data),
    )
    era5_pl["theta"] = era5_pl["theta"].assign_attrs(
        long_name="Potential Temperature", units="K"
    )

    print(f"{dt.datetime.utcnow()}: Adding variable: Virtual potential temperature")
    era5_pl["theta_v"] = (
        coords,
        variables.virtual_potential_temperature(
            P * 100, era5_pl["T"].data, era5_pl["Td"].data
        ),
    )
    era5_pl["theta_v"] = era5_pl["theta_v"].assign_attrs(
        long_name="Virtual Potential Temperature", units="K"
    )

    print(f"{dt.datetime.utcnow()}: Adding variable: Equivalent potential temperature")
    era5_pl["theta_e"] = (
        coords,
        variables.equivalent_potential_temperature(
            P * 100, era5_pl["T"].data, era5_pl["Td"].data
        ),
    )
    era5_pl["theta_e"] = era5_pl["theta_e"].assign_attrs(
        long_name="Equivalent Potential Temperature", units="K"
    )

    # convert specific humidity from kg/kg to g/kg
    era5_pl["q"] *= 1e3

    # assign attributes to remaining variables and dataset
    era5_pl["T"] = era5_pl["T"].assign_attrs(long_name="Temperature", units="K")
    era5_pl["q"] = era5_pl["q"].assign_attrs(
        long_name="Specific Humidity", units="g/kg"
    )
    era5_pl["u"] = era5_pl["u"].assign_attrs(
        long_name="U-component of wind", units="m/s"
    )
    era5_pl["v"] = era5_pl["v"].assign_attrs(
        long_name="V-component of wind", units="m/s"
    )
    era5_pl["sp_z"] = era5_pl["sp_z"].assign_attrs(
        long_name="Geopotential Height", units="dam"
    )
    era5_pl = era5_pl.assign_attrs(time_created=dt.datetime.utcnow().timestamp())

    # output directory for the ERA5 netCDF files (will be stored by month)
    output_folder = f"{args['outdir']}/{yr}{mo}"
    os.makedirs(output_folder, exist_ok=True)

    for i, timestep in enumerate(era5_pl["time"].values):
        hr = "%02d" % pd.to_datetime(timestep).hour
        print(
            f"{dt.datetime.utcnow()}: Saving {f'{output_folder}/era5_{yr}{mo}{dy}{hr}_global.nc'}"
        )
        era5_pl.isel(time=i).astype("float32").to_netcdf(
            f"{output_folder}/era5_{yr}{mo}{dy}{hr}_global.nc",
            engine="netcdf4",
            mode="w",
        )
