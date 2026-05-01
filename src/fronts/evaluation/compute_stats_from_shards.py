"""Compute performance statistics directly from TF dataset shards.

Pairs model predictions with y (truth labels embedded in each shard)
for guaranteed sample-level alignment — no external fronts file or
timestamp mapping needed.

Each shard element is an (x, y) pair produced by the same data pipeline
call, so prediction[i] and truth[i] are always aligned.

y is expected to have shape (lat, lon, num_classes) where class 0 is
no-front and classes 1..N are the front types in --front_types order.

Outputs TP/FP/TN/FN as NetCDF over 100 probability thresholds and 5
neighborhood radii (50–250 km), with optional land/ocean masking.

Usage:
    PYTHONPATH=src python src/fronts/evaluation/compute_stats_from_shards.py \\
        --model_path ~/models/fronts/1702_retrain.keras \\
        --tf_indir ~/data/tf_datasets \\
        --front_types CF WF SF OF DL \\
        --outdir ~/models/fronts/stats

    # Land-only:
    PYTHONPATH=src python src/fronts/evaluation/compute_stats_from_shards.py \\
        --model_path ~/models/fronts/1702_retrain.keras \\
        --tf_indir ~/data/tf_datasets \\
        --front_types CF WF SF OF DL \\
        --outdir ~/models/fronts/stats \\
        --mask land
"""

import argparse
import os

import numpy as np
import xarray as xr
import tensorflow as tf
from scipy.ndimage import maximum_filter
from tqdm import tqdm

from fronts.evaluation.predict_from_tf_shards import (
    SqueezeAxes,
    load_model_patched,
    load_shards,
    _numeric_shard_key,
)
from fronts.utils import constants


NEIGHBORHOODS_KM = np.array([50, 100, 150, 200, 250])
# Each 50km step = 2 pixel expansions at 0.25deg (~28km/pixel at mid-lats).
EXPAND_ITERS_PER_STEP = 2
N_THRESHOLDS = 100
THRESHOLDS = np.linspace(0.01, 1.0, N_THRESHOLDS, dtype=np.float32)


def _expand_binary(arr: np.ndarray, total_iters: int) -> np.ndarray:
    """Binary dilation via maximum_filter — equivalent to expand_fronts."""
    size = 2 * total_iters + 1
    return maximum_filter(arr, size=size).astype(arr.dtype)


def _build_spatial_mask(lats: np.ndarray, lons: np.ndarray, mask: str) -> np.ndarray:
    """Return a (Nlat, Nlon) bool array — True where points should be included."""
    import regionmask
    land_regions = regionmask.defined_regions.natural_earth_v5_1_2.land_110
    raw = land_regions.mask(lons, lats)
    is_land = ~np.isnan(raw.values)
    if mask == "land":
        spatial_mask = is_land
    else:
        spatial_mask = ~is_land
    print(
        "Spatial mask (%s): %d / %d grid points included."
        % (mask, spatial_mask.sum(), spatial_mask.size)
    )
    return spatial_mask


def compute_stats(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    front_types: list[str],
    lats: np.ndarray,
    lons: np.ndarray,
    spatial_mask: np.ndarray | None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Iterate over (x, y) pairs, run predictions, accumulate TP/FP/TN/FN.

    Returns (spatial_ds, aggregate_ds).

    spatial_ds  — TP/FP/TN/FN accumulated per grid point:
                  variables tp_spatial_CF etc., shape (lat, lon, neighborhood, threshold)
    aggregate_ds — TP/FP/TN/FN summed over all grid points:
                  variables tp_CF etc., shape (neighborhood, threshold)
    """
    n_fronts = len(front_types)
    n_nbhd = len(NEIGHBORHOODS_KM)
    Nlat, Nlon = len(lats), len(lons)

    lat_weights = np.cos(np.deg2rad(lats))[:, np.newaxis] * np.ones((1, Nlon))
    if spatial_mask is not None:
        lat_weights = lat_weights * spatial_mask.astype(float)

    tp_sp = np.zeros((n_fronts, Nlat, Nlon, n_nbhd, N_THRESHOLDS), dtype=np.float32)
    fp_sp = np.zeros_like(tp_sp)
    tn_sp = np.zeros_like(tp_sp)
    fn_sp = np.zeros_like(tp_sp)
    tp_ag = np.zeros((n_fronts, n_nbhd, N_THRESHOLDS), dtype=np.float32)
    fp_ag = np.zeros_like(tp_ag)
    tn_ag = np.zeros_like(tp_ag)
    fn_ag = np.zeros_like(tp_ag)

    card = dataset.cardinality().numpy()
    n_samples = card if card > 0 else None

    for x, y in tqdm(dataset.batch(1), total=n_samples, unit="sample"):
        pred = model(x, training=False)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        # pred: (1, lat, lon, n_classes), y: (1, lat, lon, n_classes)
        pred_np = pred.numpy()[0].astype(np.float32)  # (lat, lon, n_classes)
        y_np = y.numpy()[0].astype(np.float32)        # (lat, lon, n_classes)

        # Drop no-front class 0 → (lat, lon, n_fronts)
        pred_fronts = pred_np[:, :, 1:]
        truth_fronts = (y_np[:, :, 1:] > 0.5)  # binary bool (lat, lon, n_fronts)

        # Expand axis for threshold broadcast: (lat, lon, n_fronts, 1)
        pf4 = pred_fronts[:, :, :, np.newaxis]   # (lat, lon, n_fronts, 1)
        tf4 = truth_fronts[:, :, :, np.newaxis]  # (lat, lon, n_fronts, 1)
        wt = lat_weights[:, :, np.newaxis, np.newaxis]  # (lat, lon, 1, 1)

        # TN/FN — no neighbourhood expansion
        below = pf4 < THRESHOLDS  # (lat, lon, n_fronts, N_THRESHOLDS)
        tn_contrib = (below & ~tf4).astype(np.float32) * wt
        fn_contrib = (below & tf4).astype(np.float32) * wt
        for fi in range(n_fronts):
            # spatial: tile across neighbourhoods (same for all nbhd)
            tn_sp[fi, :, :, :, :] += tn_contrib[:, :, fi, np.newaxis, :]
            fn_sp[fi, :, :, :, :] += fn_contrib[:, :, fi, np.newaxis, :]
            tn_ag[fi, :, :] += tn_contrib[:, :, fi, np.newaxis, :].sum(axis=(0, 1))
            fn_ag[fi, :, :] += fn_contrib[:, :, fi, np.newaxis, :].sum(axis=(0, 1))

        # TP/FP — cumulative neighbourhood expansion
        expanded = truth_fronts.copy()  # (lat, lon, n_fronts)
        for ni in range(n_nbhd):
            # Expand by 2 more iterations (cumulative) to reach (ni+1)*50km
            for fi in range(n_fronts):
                expanded[:, :, fi] = _expand_binary(
                    expanded[:, :, fi].astype(np.uint8), EXPAND_ITERS_PER_STEP
                ).astype(bool)

            exp4 = expanded[:, :, :, np.newaxis]  # (lat, lon, n_fronts, 1)
            above = pf4 >= THRESHOLDS              # (lat, lon, n_fronts, N_THRESHOLDS)
            tp_contrib = (above & exp4).astype(np.float32) * wt
            fp_contrib = (above & ~exp4).astype(np.float32) * wt
            for fi in range(n_fronts):
                tp_sp[fi, :, :, ni, :] += tp_contrib[:, :, fi, :]
                fp_sp[fi, :, :, ni, :] += fp_contrib[:, :, fi, :]
                tp_ag[fi, ni, :] += tp_contrib[:, :, fi, :].sum(axis=(0, 1))
                fp_ag[fi, ni, :] += fp_contrib[:, :, fi, :].sum(axis=(0, 1))

    # Build output datasets
    spatial_ds = xr.Dataset(coords={
        "latitude": lats,
        "longitude": lons,
        "neighborhood": NEIGHBORHOODS_KM,
        "threshold": THRESHOLDS,
    })
    aggregate_ds = xr.Dataset(coords={
        "neighborhood": NEIGHBORHOODS_KM,
        "threshold": THRESHOLDS,
    })

    for fi, ft in enumerate(front_types):
        dims_sp = ("latitude", "longitude", "neighborhood", "threshold")
        dims_ag = ("neighborhood", "threshold")
        spatial_ds[f"tp_spatial_{ft}"] = (dims_sp, tp_sp[fi])
        spatial_ds[f"fp_spatial_{ft}"] = (dims_sp, fp_sp[fi])
        spatial_ds[f"tn_spatial_{ft}"] = (dims_sp, tn_sp[fi])
        spatial_ds[f"fn_spatial_{ft}"] = (dims_sp, fn_sp[fi])
        aggregate_ds[f"tp_{ft}"] = (dims_ag, tp_ag[fi])
        aggregate_ds[f"fp_{ft}"] = (dims_ag, fp_ag[fi])
        aggregate_ds[f"tn_{ft}"] = (dims_ag, tn_ag[fi])
        aggregate_ds[f"fn_{ft}"] = (dims_ag, fn_ag[fi])

    return spatial_ds, aggregate_ds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute performance stats from TF dataset shards (y-aligned)."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tf_indir", type=str, required=True)
    parser.add_argument(
        "--front_types", type=str, nargs="+", required=True,
        help="Front type names in class order (excluding no-front class 0).",
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument(
        "--mask", type=str, default=None, choices=["land", "ocean"],
        help="Restrict stats to land or ocean grid points only.",
    )
    parser.add_argument("--gpu_device", type=int, default=None)
    parser.add_argument("--lon_min", type=float, default=constants.DOMAIN_EXTENTS["conus"][0])
    parser.add_argument("--lon_max", type=float, default=constants.DOMAIN_EXTENTS["conus"][1])
    parser.add_argument("--lat_min", type=float, default=constants.DOMAIN_EXTENTS["conus"][2])
    parser.add_argument("--lat_max", type=float, default=constants.DOMAIN_EXTENTS["conus"][3])
    parser.add_argument("--resolution", type=float, default=0.25)
    args = parser.parse_args()

    if args.gpu_device is not None:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_visible_devices(gpus[args.gpu_device], "GPU")
            tf.config.experimental.set_memory_growth(gpus[args.gpu_device], True)
            print(f"Using GPU {args.gpu_device}: {gpus[args.gpu_device].name}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("Running on CPU.")

    print(f"Loading model from {args.model_path} …")
    model = load_model_patched(args.model_path)
    print(f"Model loaded. {len(model.outputs)} output(s).")

    dataset = load_shards(args.tf_indir)
    print(f"Dataset element spec: {dataset.element_spec}")

    # Build spatial mask once (lat/lon derived from first batch inside compute_stats)
    lats = np.arange(args.lat_min, args.lat_max+0.01, args.resolution)
    lons = np.arange(args.lon_min, args.lon_max+0.01, args.resolution)
    spatial_mask = _build_spatial_mask(lats, lons, args.mask) if args.mask else None

    spatial_ds, aggregate_ds = compute_stats(
        model, dataset, args.front_types,
        lats=lats, lons=lons,
        spatial_mask=spatial_mask,
    )

    os.makedirs(args.outdir, exist_ok=True)
    mask_suffix = f"_{args.mask}" if args.mask else ""
    spatial_path = os.path.join(args.outdir, f"stats_spatial{mask_suffix}.nc")
    aggregate_path = os.path.join(args.outdir, f"stats_aggregate{mask_suffix}.nc")

    spatial_ds.astype("float32").to_netcdf(spatial_path)
    aggregate_ds.astype("float32").to_netcdf(aggregate_path)
    print(f"Spatial stats   → {spatial_path}")
    print(f"Aggregate stats → {aggregate_path}")


if __name__ == "__main__":
    main()
