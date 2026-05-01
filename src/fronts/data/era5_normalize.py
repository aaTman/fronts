"""Step 2: Derive variables and normalize raw ERA5 NetCDF → normalized NetCDF.

Loads the raw stacked output from era5_download.py, computes derived variables
(dewpoint, virtual_temperature, etc.), normalizes, and saves as float16 NetCDF.

The raw file can optionally be deleted after normalization to free disk space.

Run once per month:
    PYTHONPATH=src python src/fronts/data/era5_normalize.py \\
        --config configs/1702.yaml \\
        --year 2019 --month 1 \\
        --raw_dir  ~/data/era5_raw \\
        --norm_dir ~/data/era5_norm \\
        --delete_raw

Loop over all months (fish shell):
    for m in (seq 1 12)
        PYTHONPATH=src python src/fronts/data/era5_normalize.py \\
            --config configs/1702.yaml --year 2019 --month $m \\
            --raw_dir ~/data/era5_raw --norm_dir ~/data/era5_norm --delete_raw
    end
"""

import argparse
import os

import xarray as xr

from fronts.train import open_config_yaml_as_dataclass, TrainConfig
from fronts.data import era5 as era5_module
from fronts.utils.calc import derived_variable_callable_mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive + normalize raw ERA5 NetCDF → normalized NetCDF (float16)."
    )
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--year",     type=int, required=True)
    parser.add_argument("--month",    type=int, required=True)
    parser.add_argument("--raw_dir",  type=str, required=True,
                        help="Directory containing era5_raw_YYYYMM.nc files.")
    parser.add_argument("--norm_dir", type=str, required=True,
                        help="Output directory for era5_norm_YYYYMM.nc files.")
    parser.add_argument(
        "--delete_raw", action="store_true",
        help="Delete the raw file after successful normalization.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-normalize even if output file already exists.",
    )
    args = parser.parse_args()

    os.makedirs(args.norm_dir, exist_ok=True)
    raw_file  = os.path.join(args.raw_dir,  f"era5_raw_{args.year}{args.month:02d}.nc")
    norm_file = os.path.join(args.norm_dir, f"era5_norm_{args.year}{args.month:02d}.nc")

    if not os.path.isfile(raw_file):
        raise FileNotFoundError(f"Raw file not found: {raw_file}")
    if os.path.isfile(norm_file) and not args.overwrite:
        print(f"Already exists, skipping: {norm_file}")
        return

    train_cfg = open_config_yaml_as_dataclass(args.config, TrainConfig)
    era5_cfg  = train_cfg.data_config.era5_config
    normalization_method = train_cfg.data_config.normalization_method

    to_derive = [v for v in era5_cfg.variables if v in derived_variable_callable_mapping]

    print(f"Loading raw file: {raw_file}")
    ds = xr.open_dataset(raw_file, chunks={"time": 10}, engine="netcdf4")
    print(f"  Variables: {list(ds.data_vars)}")

    # Compute derived variables (requires float32 for numerical accuracy)
    ds = ds.astype("float32")
    if to_derive:
        print(f"Deriving: {to_derive}")
        ds = era5_module.maybe_derive_variables(ds, to_derive)

    print(f"Normalizing with method={normalization_method!r} …")
    ds_norm = era5_module.normalize_legacy_arco_era5(ds, method=normalization_method)

    ds_norm = ds_norm.chunk({"time": 10})
    print(f"Writing → {norm_file}")
    ds_norm.astype("float16").to_netcdf(norm_file, engine="netcdf4")
    size_gb = os.path.getsize(norm_file) / 1e9
    print(f"Done. File size: {size_gb:.2f} GB")

    if args.delete_raw:
        os.remove(raw_file)
        print(f"Deleted raw file: {raw_file}")


if __name__ == "__main__":
    main()
