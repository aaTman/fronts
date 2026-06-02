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

try:
    from arraylake import Client as ArraylakeClient
except ImportError:
    ArraylakeClient = None

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------
N_TIMESTEPS = 28
TIME_START = "2018-01-01T00:00:00"
TIME_RESOLUTION = "6h"
VARIABLES = ["geopotential", "temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity"]
ARRAYLAKE_VARIABLES = ["z", "t", "u", "v", "q"]  # Earthmover's variable names for the above
PRESSURE_LEVELS = [1000, 925, 850, 700, 500, 300]

ARCO_URI = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ARRAYLAKE_REPO = "earthmover-public/era5"
ARRAYLAKE_GROUP = "pressure/spatial"


def _date_range(n: int) -> pd.DatetimeIndex:
    return pd.date_range(TIME_START, periods=n, freq=TIME_RESOLUTION)


def open_arco_subset() -> xr.Dataset:
    ds = xr.open_zarr(ARCO_URI, chunks=None, storage_options={"token": "anon"})
    return ds[VARIABLES].sel(time=_date_range(N_TIMESTEPS), level=PRESSURE_LEVELS)


def open_arraylake_subset() -> xr.Dataset:
    if ArraylakeClient is None:
        print("ERROR: arraylake is not installed. Run: pixi add arraylake", file=sys.stderr)
        sys.exit(1)
    client = ArraylakeClient()
    repo = client.get_repo(ARRAYLAKE_REPO)
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, group=ARRAYLAKE_GROUP, chunks=None)
    return ds[ARRAYLAKE_VARIABLES].sel(valid_time=_date_range(N_TIMESTEPS), pressure_level=PRESSURE_LEVELS)


def benchmark(label: str, ds: xr.Dataset) -> None:
    print(f"--- {label} ---")
    print(f"  Shape: {dict(ds.sizes)}")
    net_before = psutil.net_io_counters().bytes_recv
    t0 = time.perf_counter()
    ds_computed = ds.compute()
    elapsed = time.perf_counter() - t0
    net_gb = (psutil.net_io_counters().bytes_recv - net_before) / 1e9
    uncompressed_gb = ds_computed.nbytes / 1e9
    print(f"  Wall time        : {elapsed:.2f} s")
    print(f"  Network received : {net_gb:.3f} GB  ({net_gb / elapsed:.3f} GB/s)")
    print(f"  Uncompressed size: {uncompressed_gb:.3f} GB")
    print(f"  Compression ratio: {uncompressed_gb / net_gb:.1f}x\n")


def main() -> None:
    print(f"Benchmarking ERA5 sources — {N_TIMESTEPS} timesteps @ {TIME_RESOLUTION}, full global grid\n")

    print("Opening Arraylake ERA5...")
    arraylake_ds = open_arraylake_subset()
    benchmark("Arraylake", arraylake_ds)

    print("Opening ARCO ERA5...")
    arco_ds = open_arco_subset()
    benchmark("ARCO (GCS)", arco_ds)


if __name__ == "__main__":
    main()
