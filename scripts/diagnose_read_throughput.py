"""Diagnose ERA5/fronts icechunk store read throughput and training batch patterns.

Run on the machine where the stores are colocated:

    pixi run python scripts/diagnose_read_throughput.py [--config configs/schooner_train.yaml]

Prints per-variable dtype/chunking/compression for both stores, times reads at the
raw store level and through ``inputs_ds_to_dataarray``, then compares contiguous vs
scattered batch reads through a real ``FrontsPyDataset`` and attributes one batch's
cost across raw input I/O, channel stacking, and target remap/one-hot, so the
bottleneck (chunk layout, shuffle access pattern, or per-batch transforms) can be
localized.
"""

import argparse
import time

import numpy as np
import xarray as xr

from fronts import utils
from fronts.data import datasets, inputs, targets


def _fmt_bytes(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


def _open_store(icechunk_config: utils.IcechunkStorageConfig, chunks: str | None = "auto") -> xr.Dataset:
    return utils.open_readonly_icechunk_store(
        store_path=icechunk_config.store_path,
        branch=icechunk_config.branch_name,
        group=icechunk_config.group_name,
        zarr_format=icechunk_config.zarr_format,
        virtual_chunk_local_path=icechunk_config.virtual_chunk_local_path,
        chunks=chunks,
    )


def _print_layout(ds: xr.Dataset, title: str) -> None:
    print(f"=== {title}: per-variable layout ===")
    for name, var in ds.data_vars.items():
        enc = var.encoding
        print(
            f"{name:35s} dtype={var.dtype!s:8s} shape={var.shape} "
            f"chunks={enc.get('chunks')} compressor={enc.get('compressors', enc.get('compressor'))}"
        )


def _time_input_reads(ds: xr.Dataset, data_cfg: datasets.DatasetConfig) -> None:
    n_time = ds.sizes["time"]
    print(f"\nTotal timesteps in store: {n_time}")

    sample_var = next(iter(ds.data_vars))
    print(f"\n=== Raw single-variable read timing ('{sample_var}') ===")
    for k in (1, 10, 50):
        sub = ds[sample_var].isel(time=slice(0, k))
        nbytes = sub.nbytes
        t0 = time.time()
        sub.compute()
        dt = time.time() - t0
        print(f"  {k:3d} steps: {_fmt_bytes(nbytes)} in {dt:6.1f} s -> {nbytes / 1e6 / max(dt, 1e-9):7.1f} MB/s")

    print("\n=== Full-stack read timing (inputs_ds_to_dataarray) ===")
    da = inputs.inputs_ds_to_dataarray(ds, data_cfg.variables)
    print(f"  assembled dtype={da.dtype} shape={da.shape} chunks={da.chunks}")
    for k in (1, 10, 50):
        sub = da.isel(time=slice(0, k))
        nbytes = sub.nbytes
        t0 = time.time()
        sub.compute()
        dt = time.time() - t0
        print(f"  {k:3d} steps: {_fmt_bytes(nbytes)} in {dt:6.1f} s -> {nbytes / 1e6 / max(dt, 1e-9):7.1f} MB/s")


def _build_dataset(data_cfg: datasets.DatasetConfig) -> datasets.FrontsPyDataset:
    """Build a FrontsPyDataset over the time intersection of both stores, no split/filtering.

    Skips the class-balancing subsample and year split from ``train.load_data_into_dataloader``
    (which needs a full targets scan) — index locality behaves the same either way.
    """
    inputs_ds = utils.unwrap_longitude(_open_store(data_cfg.inputs_icechunk_config, chunks=None))
    targets_da = utils.unwrap_longitude(_open_store(data_cfg.targets_icechunk_config, chunks=None))["identifier"]
    common_times = np.intersect1d(targets_da.time.values, inputs_ds.time.values)
    common_times = utils.apply_time_resolution(common_times, data_cfg.time_resolution)
    return datasets.FrontsPyDataset(
        input_ds=inputs_ds.sel(time=common_times),
        target_da=targets_da.sel(time=common_times),
        data_config=data_cfg,
        batch_size=data_cfg.batch_size,
    )


def _time_batch_patterns(dataset: datasets.FrontsPyDataset, reps: int = 3, seed: int = 0) -> None:
    """Compare contiguous vs scattered ``get_at_indices`` batches — the block-shuffle question."""
    rng = np.random.default_rng(seed)
    n = dataset.n_samples
    batch_size = dataset.batch_size
    print(f"\n=== Batch access pattern timing ({reps} reps, batch_size={batch_size}) ===")
    for pattern in ("contiguous", "scattered"):
        times = []
        for _ in range(reps):
            if pattern == "contiguous":
                start = int(rng.integers(0, n - batch_size))
                idxs = np.arange(start, start + batch_size)
            else:
                idxs = np.sort(rng.choice(n, size=batch_size, replace=False))
            t0 = time.time()
            dataset.get_at_indices(idxs)
            times.append(time.time() - t0)
        print(f"  {pattern:10s}: min={min(times):6.1f}s  mean={np.mean(times):6.1f}s  max={max(times):6.1f}s")


def _attribute_batch_cost(dataset: datasets.FrontsPyDataset, seed: int = 0) -> None:
    """Split one scattered batch's cost into raw input I/O, channel stacking, and target transform."""
    rng = np.random.default_rng(seed)
    idxs = np.sort(rng.choice(dataset.n_samples, size=dataset.batch_size, replace=False))
    print(f"\n=== Per-batch cost attribution (scattered batch of {dataset.batch_size}) ===")

    t0 = time.time()
    raw = dataset.input_ds.isel(time=idxs).load()
    t_raw = time.time() - t0
    print(f"  raw input isel().load()          : {t_raw:6.1f}s ({_fmt_bytes(sum(v.nbytes for v in raw.values()))})")

    t0 = time.time()
    stacked = inputs.inputs_ds_to_dataarray(raw, dataset.data_config.variables).values
    t_stack = time.time() - t0
    print(f"  channel stacking on loaded data  : {t_stack:6.1f}s ({_fmt_bytes(stacked.nbytes)})")

    t0 = time.time()
    y_da = targets.one_hot_encode_to_dataarray(targets.remap_fronts(dataset.target_da.isel(time=idxs)))
    if dataset.data_config.front_dilation > 0:
        y_da = targets.dilate_fronts(y_da, dataset.data_config.front_dilation)
    y = y_da.values
    t_targets = time.time() - t0
    print(f"  targets read + remap + one-hot   : {t_targets:6.1f}s ({_fmt_bytes(y.nbytes)})")


def main() -> None:
    """Print store layouts and time read patterns to localize the throughput bottleneck."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/schooner_train.yaml", help="Training config YAML path.")
    args = parser.parse_args()

    yaml_data = utils.load_yaml(args.config)
    data_cfg = utils.parse_config_section(yaml_data, datasets.DatasetConfig, "data_config")

    inputs_ds = _open_store(data_cfg.inputs_icechunk_config)
    _print_layout(inputs_ds, "ERA5 inputs")
    targets_ds = _open_store(data_cfg.targets_icechunk_config)
    _print_layout(targets_ds, "Fronts targets")

    _time_input_reads(inputs_ds, data_cfg)

    dataset = _build_dataset(data_cfg)
    _time_batch_patterns(dataset)
    _attribute_batch_cost(dataset)


if __name__ == "__main__":
    main()
