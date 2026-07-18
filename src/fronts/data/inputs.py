import hashlib
import logging
import os
import pathlib
import tempfile

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

_REDUCTION_DIMS = ("time", "latitude", "longitude")


def _norm_stats_shape(da: xr.DataArray) -> tuple[int, ...]:
    """Shape of the per-channel statistics for ``da``: its dims minus the reduced ones.

    (n_channels,) for the flat 2D-model layout, (n_levels, n_variables) for the 3D
    volume layout.
    """
    return tuple(da.sizes[dim] for dim in da.dims if dim not in _REDUCTION_DIMS)


def inputs_ds_to_dataarray(ds: xr.Dataset, variables: list[str]) -> xr.DataArray:
    """Convert a gridded input Dataset to a lazy 4D DataArray without materializing data.

    Uses xarray's to_array and stack so no data is loaded from disk until
    iteration. Pressure-level variables come first, ordered level-outer,
    variable-inner (all variables at level[0], then all variables at level[1],
    etc.), followed by one channel per single-level variable in the order
    requested. Channel labels are ``{variable}_{level}`` for pressure-level
    variables and the bare variable name for single-level ones. Time-invariant
    variables (e.g. land_sea_mask) are broadcast along the dataset's time
    coordinate.

    Args:
        ds: Dataset containing the requested variables with dims
            (time, level, latitude, longitude) or (time, latitude, longitude);
            time-invariant variables may omit the time dim.
        variables: Variable names to select from ds.

    Returns:
        DataArray of shape (time, latitude, longitude, channel) with dtype float32,
        where channel = n_levels * n_level_variables + n_single_level_variables.
    """
    level_vars = [v for v in variables if "level" in ds[v].dims]
    single_vars = [v for v in variables if "level" not in ds[v].dims]

    pieces = []
    if level_vars:
        stacked = (
            ds[level_vars]
            .to_array(dim="variable")
            .transpose("time", "latitude", "longitude", "level", "variable")
            .stack(channel=("level", "variable"))
        )
        labels = [f"{var}_{int(level)}" for level, var in stacked.indexes["channel"]]
        stacked = stacked.reset_index("channel").drop_vars(["level", "variable"]).assign_coords(channel=labels)
        pieces.append(stacked.reset_coords(drop=True))
    if single_vars:
        broadcast = {}
        for var in single_vars:
            # Strip any self-referential coordinate (e.g. when `var` is itself a Dataset
            # coordinate, not a plain data variable): xr.Dataset() silently demotes a
            # variable carrying a same-named coordinate to Dataset.coords when merged
            # alongside other data variables, which then drops it entirely from to_array().
            # This is for land_sea_mask primarily
            da = ds[var].reset_coords(drop=True)
            if "time" not in da.dims:
                if "time" not in ds.coords:
                    raise ValueError(
                        f"Variable '{var}' has no time dimension and the dataset has no time "
                        "coordinate to broadcast it along."
                    )
                da = da.expand_dims(time=ds["time"])
            broadcast[var] = da
        single = xr.Dataset(broadcast).to_array(dim="channel").transpose("time", "latitude", "longitude", "channel")
        pieces.append(single.reset_coords(drop=True))

    if not pieces:
        raise ValueError("No variables requested.")
    result = pieces[0] if len(pieces) == 1 else xr.concat(pieces, dim="channel", coords="minimal")
    return result.astype(np.float32)


def inputs_ds_to_volume_dataarray(ds: xr.Dataset, variables: list[str]) -> xr.DataArray:
    """Convert a gridded input Dataset to a lazy 5D volume DataArray without materializing data.

    Unlike ``inputs_ds_to_dataarray`` (which flattens level and variable into one
    channel axis for 2D Conv2D models), this keeps the vertical structure intact
    for 3D Conv3D models: pressure-level variables form a trailing (level, variable)
    pair. Single-level variables (e.g. mean_sea_level_pressure) are broadcast along
    the level dimension as constant columns so they participate as ordinary
    variables; time-invariant variables are additionally broadcast along time.

    Args:
        ds: Dataset containing the requested variables with dims
            (time, level, latitude, longitude) or (time, latitude, longitude);
            time-invariant variables may omit the time dim.
        variables: Variable names to select from ds, in the desired variable-axis order.

    Returns:
        DataArray of shape (time, latitude, longitude, level, variable) with dtype float32.

    Raises:
        ValueError: If no variables are requested, or a variable cannot be broadcast
            because the dataset lacks the needed time or level coordinate.
    """
    if not variables:
        raise ValueError("No variables requested.")

    prepared = {}
    for var in variables:
        da = ds[var].reset_coords(drop=True)
        if "time" not in da.dims:
            if "time" not in ds.coords:
                raise ValueError(
                    f"Variable '{var}' has no time dimension and the dataset has no time "
                    "coordinate to broadcast it along."
                )
            da = da.expand_dims(time=ds["time"])
        if "level" not in da.dims:
            if "level" not in ds.coords:
                raise ValueError(
                    f"Variable '{var}' has no level dimension and the dataset has no level "
                    "coordinate to broadcast it along."
                )
            da = da.expand_dims(level=ds["level"])
        prepared[var] = da

    stacked = (
        xr.Dataset(prepared).to_array(dim="variable").transpose("time", "latitude", "longitude", "level", "variable")
    )
    return stacked.reset_coords(drop=True).astype(np.float32)


def compute_norm_stats(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel min and max over the full DataArray in one pass.

    Builds both reductions as a single dask graph so each chunk is read from
    disk once and the full array is never materialized in memory. Suitable for
    building a min-max Rescaling layer.

    Min-max (rather than mean/variance z-score) normalization is used because
    several channels (e.g. specific_humidity, potential_vorticity) have
    near-zero variance in their native units, which blows up z-scored values
    to hundreds of standard deviations for ordinary samples and overflows
    float16 activations. Min-max keeps every channel bounded regardless of
    how small its variance is.

    Args:
        da: DataArray of shape (time, latitude, longitude, channel) or
            (time, latitude, longitude, level, variable); statistics are computed
            separately for every element of the trailing (non-reduced) dims.

    Returns:
        Tuple of (min, max) float32 arrays shaped like the trailing dims of ``da``
        ((n_channels,) or (n_levels, n_variables)).

    Raises:
        ValueError: If the computed statistics contain NaN values.
    """
    reduction_dims = list(_REDUCTION_DIMS)
    stats = xr.Dataset(
        {
            "min": da.min(dim=reduction_dims, skipna=False),
            "max": da.max(dim=reduction_dims, skipna=False),
        }
    ).compute()
    min_val = stats["min"].values.astype(np.float32)
    max_val = stats["max"].values.astype(np.float32)
    if np.isnan(min_val).any() or np.isnan(max_val).any():
        raise ValueError(
            "NaN in normalization statistics — ERA5 data contains missing values. "
            "Check the icechunk store for corrupted or incomplete channels."
        )
    return min_val, max_val


def load_or_compute_norm_stats(
    da: xr.DataArray,
    cache_dir: str | None,
    cache_key_parts: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached normalization statistics or compute and cache them.

    The cache key is a SHA-256 hash of cache_key_parts (e.g. the store snapshot
    ID), so stats are reused only when every part matches a previous run. Cached
    stats whose shape no longer matches ``da``'s trailing dims (channel-count
    changes, or a flat cache read for a volume request), or that predate the
    switch to min-max stats, are ignored and recomputed.

    Args:
        da: DataArray of shape (time, latitude, longitude, channel) or
            (time, latitude, longitude, level, variable).
        cache_dir: Directory for cached stats files. None disables caching.
        cache_key_parts: Strings that uniquely determine the statistics.

    Returns:
        Tuple of (min, max) float32 arrays shaped like the trailing dims of ``da``.
    """
    snapshot_id = cache_key_parts[0]
    if cache_dir is None:
        logger.info("Normalization cache disabled; computing stats for snapshot %s.", snapshot_id)
        return compute_norm_stats(da)

    key = hashlib.sha256("\x1f".join(cache_key_parts).encode()).hexdigest()[:16]
    cache_path = pathlib.Path(cache_dir) / f"norm_stats_{key}.npz"
    expected_shape = _norm_stats_shape(da)
    if cache_path.exists():
        cached = np.load(cache_path)
        if "min" in cached and cached["min"].shape == expected_shape:
            logger.info("Reusing cached normalization stats for snapshot %s from %s.", snapshot_id, cache_path)
            return cached["min"], cached["max"]
        logger.warning(
            "Cached normalization stats for snapshot %s are missing or have a shape mismatch "
            "(data needs stats of shape %s); recomputing.",
            snapshot_id,
            expected_shape,
        )
    else:
        logger.info("No cached normalization stats for snapshot %s (key %s); computing.", snapshot_id, key)

    min_val, max_val = compute_norm_stats(da)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix=".npz.tmp", delete=False) as tmp:
        np.savez(tmp, min=min_val, max=max_val)
        tmp_path = tmp.name
    os.replace(tmp_path, cache_path)
    logger.info("Saved normalization stats for snapshot %s to %s.", snapshot_id, cache_path)
    return min_val, max_val
