"""Throughput benchmark: Google ARCO ERA5 vs Earthmover Arraylake ERA5.

Uses the full global spatial domain (chunks already span the full lat/lon extent)
and times .compute() on both sources. N_TIMESTEPS is tuned so each source reads
~3.5 GB: 5 vars × 6 levels × 721 lat × 1440 lon × float32 ≈ 125 MB/timestep,
so 28 timesteps ≈ 3.5 GB.
"""

import sys
import time

import pandas as pd
import psutil
import xarray as xr
import logging

try:
    from arraylake import Client as ArraylakeClient
except ImportError:
    ArraylakeClient = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

N_TIMESTEPS = 28
TIME_START = "2018-01-01T00:00:00"
TIME_RESOLUTION = "6h"
VARIABLES = ["geopotential", "temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity"]
ARRAYLAKE_VARIABLES = ["z", "t", "u", "v", "q"]  # Earthmover's variable names for the above
PRESSURE_LEVELS = [1000, 925, 850, 700, 500, 300]

ARCO_URI = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Earthmover's Arraylake ERA5 Zarr store, which is publicly accessible with an anonymous token.
ARRAYLAKE_REPO = "earthmover-public/era5"

# The group within the Arraylake repo where the pressure level ERA5 data is stored chunked pancake-style
ARRAYLAKE_GROUP = "pressure/spatial"


def open_arco_subset() -> xr.Dataset:
    """Open and subset the Google ARCO ERA5 Zarr store.

    Open and subset the store to the specified variables, time range, pressure levels, and spatial domain.

    Returns:
        An xarray Dataset containing the subset of ARCO ERA5 data.
    """
    ds = xr.open_zarr(ARCO_URI, chunks=None, storage_options={"token": "anon"})
    return ds[VARIABLES].sel(
        time=pd.date_range(TIME_START, periods=N_TIMESTEPS, freq=TIME_RESOLUTION), level=PRESSURE_LEVELS
    )


def open_arraylake_subset() -> xr.Dataset:
    """Open and subset the Earthmover Arraylake ERA5 Icechunk store.

    Open and subset the store to the specified variables, time range, pressure levels, and spatial domain.

    Returns:
        An xarray Dataset containing the subset of Arraylake ERA5 data.
    """
    if ArraylakeClient is None:
        logger.info("ERROR: arraylake is not installed. Run: pixi add arraylake", file=sys.stderr)
        sys.exit(1)
    client = ArraylakeClient()
    repo = client.get_repo(ARRAYLAKE_REPO)
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, group=ARRAYLAKE_GROUP, chunks=None)
    return ds[ARRAYLAKE_VARIABLES].sel(
        valid_time=pd.date_range(TIME_START, periods=N_TIMESTEPS, freq=TIME_RESOLUTION), pressure_level=PRESSURE_LEVELS
    )


def benchmark(label: str, ds: xr.Dataset) -> None:
    """Benchmark the time and network throughput of computing an xarray Dataset.

    Args:
        label: Label for the dataset being benchmarked (e.g., "ARCO", "Arraylake").
        ds: The xarray Dataset to compute and benchmark.
    """
    logger.info(f"--- {label} ---")
    logger.info(f"  Shape: {dict(ds.sizes)}")
    net_before = psutil.net_io_counters().bytes_recv
    t0 = time.perf_counter()
    ds_computed = ds.compute()
    elapsed = time.perf_counter() - t0
    net_gb = (psutil.net_io_counters().bytes_recv - net_before) / 1e9
    uncompressed_gb = ds_computed.nbytes / 1e9
    logger.info(f"  Wall time        : {elapsed:.2f} s")
    logger.info(f"  Network received : {net_gb:.3f} GB  ({net_gb / elapsed:.3f} GB/s)")
    logger.info(f"  Uncompressed size: {uncompressed_gb:.3f} GB")
    logger.info(f"  Compression ratio: {uncompressed_gb / net_gb:.1f}x\n")


def main() -> None:
    """Main function to run the benchmark."""
    logger.info(f"Benchmarking ERA5 sources — {N_TIMESTEPS} timesteps @ {TIME_RESOLUTION}, full global grid\n")

    logger.info("Opening Earthmover ERA5...")
    arraylake_ds = open_arraylake_subset()
    benchmark("Earthmover ERA5", arraylake_ds)

    logger.info("Opening Google ERA5...")
    arco_ds = open_arco_subset()
    benchmark("Google", arco_ds)


if __name__ == "__main__":
    main()
