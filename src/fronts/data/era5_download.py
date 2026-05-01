"""Step 1: Download raw ERA5 from ARCO and save as local NetCDF.

Saves the stacked (surface + pressure levels) dataset — raw variables only,
no derivation. Saved as float16 to keep file sizes manageable.

Approximate output sizes per month at float16 (full domain, 6h frequency):
  ~2 GB raw (6 raw variables × 5 levels × ~120 timesteps × 320 × 960)

Run once per month:
    PYTHONPATH=src python src/fronts/data/era5_download.py \\
        --config configs/1702.yaml \\
        --year 2019 --month 1 \\
        --outdir ~/data/era5_raw

Loop over all months (fish shell):
    for m in (seq 1 12)
        PYTHONPATH=src python src/fronts/data/era5_download.py \\
            --config configs/1702.yaml --year 2019 --month $m \\
            --outdir ~/data/era5_raw
    end
"""

import argparse
import os
import calendar

import numpy as np
import pandas as pd
import xarray as xr

from fronts.train import open_config_yaml_as_dataclass, TrainConfig
from fronts.data import era5 as era5_module
from fronts.utils.calc import derived_variable_callable_mapping


# Full domain lon wraps: ARCO 0-360, full domain 130→9.75 (≡ 369.75)
_FULL_LON_PART1 = (130.0, 359.75)
_FULL_LON_PART2 = (0.0,   9.75)
_FULL_LAT       = (80.0,  0.25)   # (max, min) — ARCO stores lat descending


def _open_arco(store: str, consolidated: bool) -> xr.Dataset:
    return xr.open_dataset(
        store,
        chunks={"time": 1},
        engine="zarr",
        backend_kwargs={"storage_options": {"anon": True}},
        consolidated=consolidated,
    )


def _subset_full_domain(ds: xr.Dataset, non_surface_levels: list) -> xr.Dataset:
    """Select the full domain with lon wrap-around, then subset pressure levels."""
    lat_slice = slice(_FULL_LAT[0], _FULL_LAT[1])

    part1 = ds.sel(latitude=lat_slice, longitude=slice(*_FULL_LON_PART1))
    part2 = ds.sel(latitude=lat_slice, longitude=slice(*_FULL_LON_PART2))
    part2 = part2.assign_coords(longitude=part2.longitude + 360.0)
    ds_domain = xr.concat([part1, part2], dim="longitude")

    if non_surface_levels:
        ds_domain = ds_domain.sel(level=non_surface_levels)

    return ds_domain


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download raw ERA5 from ARCO → local NetCDF (float16)."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--year",  type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument(
        "--freq", type=str, default="6h",
        help="Timestep frequency (default: 6h).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-download even if output file already exists.",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    outfile = os.path.join(args.outdir, f"era5_raw_{args.year}{args.month:02d}.nc")
    if os.path.isfile(outfile) and not args.overwrite:
        print(f"Already exists, skipping: {outfile}")
        return

    train_cfg = open_config_yaml_as_dataclass(args.config, TrainConfig)
    era5_cfg = train_cfg.data_config.era5_config

    raw_vars = [v for v in era5_cfg.variables if v not in derived_variable_callable_mapping]
    levels = era5_cfg.levels
    non_surface_levels = [lv for lv in levels if not (lv == 1013 or lv == "surface")]

    print(f"Opening ARCO store: {era5_cfg.store}")
    ds = _open_arco(era5_cfg.store, era5_cfg.consolidated)

    print(f"Subsetting to full domain (lat {_FULL_LAT[0]}→{_FULL_LAT[1]}, "
          f"lon {_FULL_LON_PART1[0]}→{_FULL_LON_PART2[1]+360:.2f}) …")
    ds_domain = _subset_full_domain(ds, non_surface_levels)

    # Time selection for this month
    _, last_day = calendar.monthrange(args.year, args.month)
    start = f"{args.year}-{args.month:02d}-01"
    end   = f"{args.year}-{args.month:02d}-{last_day:02d} 23:59"
    times = pd.date_range(start, end, freq=args.freq)
    avail = pd.DatetimeIndex(ds_domain.time.values)
    times = times[times.isin(avail)]
    print(f"Timesteps: {len(times)}  ({times[0]} … {times[-1]})")

    ds_domain = ds_domain.sel(time=times)

    # Subset to needed raw variable names
    var_names = era5_module.subset_variables(ds_domain, variables=raw_vars, levels=levels)
    ds_raw = ds_domain[var_names]

    print("Stacking surface + pressure levels …")
    ds_stacked = era5_module.maybe_stack_variables(ds_raw, variables=raw_vars, levels=levels)

    # Rechunk for efficient writing
    ds_stacked = ds_stacked.chunk({"time": 10, "latitude": 320, "longitude": 960})

    print(f"Writing → {outfile}")
    print(f"  Variables: {list(ds_stacked.data_vars)}")
    print(f"  Shape: time={len(times)}, lat={len(ds_stacked.latitude)}, "
          f"lon={len(ds_stacked.longitude)}, "
          f"levels per var={[len(ds_stacked[v].level) for v in ds_stacked.data_vars]}")

    # Save as float16 to reduce disk usage by ~2×
    ds_stacked.astype("float16").to_netcdf(outfile, engine="netcdf4")
    size_gb = os.path.getsize(outfile) / 1e9
    print(f"Done. File size: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
