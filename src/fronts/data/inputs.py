import hashlib
import logging
import os
import pathlib
import tempfile
from typing import Literal

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

NormalizationMethod = Literal["standardization", "minmax"]
_NORM_STAT_KEYS: dict[NormalizationMethod, tuple[str, str]] = {
    "standardization": ("mean", "variance"),
    "minmax": ("min", "max"),
}


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


def compute_norm_stats(
    da: xr.DataArray, method: NormalizationMethod = "standardization"
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel normalization statistics over the full DataArray in one pass.

    Builds both reductions as a single dask graph so each chunk is read from
    disk once and the full array is never materialized in memory.

    Args:
        da: DataArray of shape (time, latitude, longitude, channel).
        method: "standardization" returns (mean, variance) for a Keras Normalization
            layer; "minmax" returns (min, max) for a Keras Rescaling layer. Min-max
            avoids the huge z-scores that near-zero-variance channels (e.g.
            specific_humidity, potential_vorticity) produce, which can overflow
            float16 activations.

    Returns:
        Tuple of two (n_channels,) float32 arrays: (mean, variance) if
        ``method == "standardization"``, else (min, max).

    Raises:
        ValueError: If ``method`` is unrecognized, or the computed statistics contain NaN.
    """
    if method not in _NORM_STAT_KEYS:
        raise ValueError(f"Unrecognized normalization method {method!r}; expected one of {list(_NORM_STAT_KEYS)}.")
    reduction_dims = ["time", "latitude", "longitude"]
    if method == "standardization":
        stats = xr.Dataset(
            {
                "a": da.mean(dim=reduction_dims, skipna=False),
                "b": da.var(dim=reduction_dims, skipna=False),
            }
        ).compute()
    else:
        stats = xr.Dataset(
            {
                "a": da.min(dim=reduction_dims, skipna=False),
                "b": da.max(dim=reduction_dims, skipna=False),
            }
        ).compute()
    first = stats["a"].values.astype(np.float32)
    second = stats["b"].values.astype(np.float32)
    if np.isnan(first).any() or np.isnan(second).any():
        raise ValueError(
            "NaN in normalization statistics — ERA5 data contains missing values. "
            "Check the icechunk store for corrupted or incomplete channels."
        )
    return first, second


def load_or_compute_norm_stats(
    da: xr.DataArray,
    cache_dir: str | None,
    cache_key_parts: tuple[str, ...],
    method: NormalizationMethod = "standardization",
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached normalization statistics or compute and cache them.

    The cache key is a SHA-256 hash of cache_key_parts (e.g. the store snapshot
    ID) plus ``method``, so stats are reused only when every part matches a
    previous run *and* the cache was written for the same normalization method.
    This keeps a "standardization" run and a "minmax" run sharing the same
    ``cache_dir`` from ever colliding on one file: each method gets its own
    cache filename, so a stale or differently-computed file is simply not found
    (a clean miss, not a crash) rather than being loaded with the wrong keys.

    Args:
        da: DataArray of shape (time, latitude, longitude, channel).
        cache_dir: Directory for cached stats files. None disables caching.
        cache_key_parts: Strings that uniquely determine the statistics.
        method: "standardization" or "minmax" — see ``compute_norm_stats``.

    Returns:
        Tuple of two (n_channels,) float32 arrays: (mean, variance) if
        ``method == "standardization"``, else (min, max).
    """
    snapshot_id = cache_key_parts[0]
    if cache_dir is None:
        logger.info("Normalization cache disabled; computing %s stats for snapshot %s.", method, snapshot_id)
        return compute_norm_stats(da, method=method)

    key_a, key_b = _NORM_STAT_KEYS[method]
    key = hashlib.sha256("\x1f".join((method, *cache_key_parts)).encode()).hexdigest()[:16]
    cache_path = pathlib.Path(cache_dir) / f"norm_stats_{method}_{key}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        if key_a in cached and cached[key_a].shape[0] == da.sizes["channel"]:
            logger.info(
                "Reusing cached %s normalization stats for snapshot %s from %s.", method, snapshot_id, cache_path
            )
            return cached[key_a], cached[key_b]
        logger.warning(
            "Cached %s normalization stats for snapshot %s are missing or have a channel-count mismatch "
            "(data has %d channels); recomputing.",
            method,
            snapshot_id,
            da.sizes["channel"],
        )
    else:
        logger.info("No cached %s normalization stats for snapshot %s (key %s); computing.", method, snapshot_id, key)

    first, second = compute_norm_stats(da, method=method)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix=".npz.tmp", delete=False) as tmp:
        np.savez(tmp, **{key_a: first, key_b: second})
        tmp_path = tmp.name
    os.replace(tmp_path, cache_path)
    logger.info("Saved %s normalization stats for snapshot %s to %s.", method, snapshot_id, cache_path)
    return first, second
