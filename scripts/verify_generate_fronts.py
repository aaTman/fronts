"""Throwaway verification script for the new front icechunk store. DELETE AFTER USE.

Prints the fronts store's xarray repr and saves a handful of ``identifier`` raster maps as PNGs,
to visually confirm ``fronts.data.generate_fronts``'s conversion + virtual-chunk registration
worked correctly. Run on the HPC system where the store's netcdf files are mounted:

    pixi run -e data python scripts/verify_generate_fronts.py \
        --config configs/generate_fronts.yaml --year 2025 --n-maps 4 --outdir .

``--year`` restricts the plotted timesteps to that calendar year (default 2025, the newly
added data), so the maps actually confirm the new data rather than landing on older years
already present in the store.
"""

import argparse
import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fronts import utils
from fronts.data.generate_fronts import PGEN_TYPE_IDENTIFIERS

_FRONT_NAMES = {code: name for name, code in PGEN_TYPE_IDENTIFIERS.items()}


def main() -> None:
    """Print the fronts store's repr and save a handful of front raster maps as PNGs."""
    parser = argparse.ArgumentParser(description="Verify the fronts icechunk store (temporary script)")
    parser.add_argument("--config", default="configs/generate_fronts.yaml", help="Config with icechunk_storage_config")
    parser.add_argument("--year", type=int, default=2025, help="Only plot timesteps from this calendar year")
    parser.add_argument("--n-maps", type=int, default=4, help="Number of timesteps to plot")
    parser.add_argument("--outdir", default=".", help="Directory to write PNGs into")
    args = parser.parse_args()

    icechunk_config = utils.open_config_yaml_as_dataclass(
        args.config, utils.IcechunkStorageConfig, config_key="icechunk_storage_config"
    )
    ds = utils.open_readonly_icechunk_store(
        icechunk_config.store_path,
        icechunk_config.branch_name,
        group=icechunk_config.group_name,
        zarr_format=icechunk_config.zarr_format,
        virtual_chunk_local_path=icechunk_config.virtual_chunk_local_path,
        chunks=None,
    )
    print(ds)

    times = pd.DatetimeIndex(ds["time"].values)
    year_positions = np.flatnonzero(times.year == args.year)
    if year_positions.size == 0:
        raise ValueError(f"No timesteps found for year {args.year} in this store.")
    print(f"\n{year_positions.size} timesteps found for {args.year}.")

    sample_positions = np.linspace(0, year_positions.size - 1, min(args.n_maps, year_positions.size), dtype=int)
    indices = sorted(set(year_positions[sample_positions]))
    plot_ds = utils.unwrap_longitude(ds)  # monotonic longitude for correct pcolormesh rendering

    os.makedirs(args.outdir, exist_ok=True)
    for i in indices:
        snapshot = plot_ds["identifier"].isel(time=i).compute()
        valid_time = str(snapshot["time"].values)[:16]
        front_pixels = int((snapshot.values > 0).sum())

        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=250))
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        mesh = ax.pcolormesh(
            snapshot["longitude"],
            snapshot["latitude"],
            snapshot.where(snapshot > 0),
            transform=ccrs.PlateCarree(),
            cmap="tab20",
            vmin=0.5,
            vmax=16.5,
        )
        codes = sorted(_FRONT_NAMES)
        cbar = fig.colorbar(mesh, ax=ax, ticks=codes, fraction=0.03)
        cbar.ax.set_yticklabels([_FRONT_NAMES[c] for c in codes])
        ax.set_title(f"identifier @ {valid_time}  ({front_pixels} front pixels)")

        outpath = os.path.join(args.outdir, f"front_check_{i:04d}.png")
        fig.savefig(outpath, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {outpath} (time={valid_time}, front pixels={front_pixels})")


if __name__ == "__main__":
    main()
