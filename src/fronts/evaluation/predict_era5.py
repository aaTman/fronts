"""Step 3: Generate front probability predictions from local normalized ERA5 NetCDFs.

Loads monthly normalized files produced by era5_normalize.py, iterates one
timestep at a time through model inference, and writes a single prediction NetCDF.

Output has dimensions (time, latitude, longitude) with one variable per front type,
suitable for generate_performance_stats.py.

Usage:
    PYTHONPATH=src python src/fronts/evaluation/predict_era5.py \\
        --model_path ~/models/fronts/1702_retrain.keras \\
        --config configs/1702.yaml \\
        --norm_dir ~/data/era5_norm \\
        --year 2019 \\
        --outfile ~/models/fronts/predictions_full_2019.nc \\
        --gpu_device 0
"""

import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from tqdm import tqdm

from fronts.train import open_config_yaml_as_dataclass, TrainConfig
from fronts.evaluation.predict_from_tf_shards import load_model_patched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict from local normalized ERA5 NetCDFs, one timestep at a time."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to training YAML (e.g. configs/1702.yaml).",
    )
    parser.add_argument(
        "--norm_dir", type=str, required=True,
        help="Directory containing era5_norm_YYYYMM.nc files.",
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--outfile", type=str, required=True)
    parser.add_argument("--gpu_device", type=int, default=None)
    parser.add_argument(
        "--front_types", type=str, nargs="+", default=None,
        help="Front type names. Defaults to target_config.front_types from config.",
    )
    parser.add_argument(
        "--variable_order", type=str, nargs="+", default=None,
        help=(
            "Explicit variable ordering for the last axis of the model input. "
            "Defaults to the variable list from the config (raw + derived, in order). "
            "Override if the model was trained with a different variable ordering."
        ),
    )
    args = parser.parse_args()

    # GPU / CPU
    if args.gpu_device is not None:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_visible_devices(gpus[args.gpu_device], "GPU")
            tf.config.experimental.set_memory_growth(gpus[args.gpu_device], True)
            print(f"Using GPU {args.gpu_device}: {gpus[args.gpu_device].name}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("Running on CPU.")

    train_cfg = open_config_yaml_as_dataclass(args.config, TrainConfig)
    era5_cfg  = train_cfg.data_config.era5_config
    front_types = args.front_types or train_cfg.data_config.target_config.front_types
    variable_order = args.variable_order or era5_cfg.variables
    print(f"Front types: {front_types}")
    print(f"Variable order for model input: {variable_order}")

    # Load model
    print(f"\nLoading model from {args.model_path} …")
    model = load_model_patched(args.model_path)
    print(f"Model loaded. {len(model.outputs)} output(s).")

    # Collect normalized monthly files, sorted by month
    pattern = os.path.join(args.norm_dir, f"era5_norm_{args.year}*.nc")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No normalized files found matching: {pattern}")
    print(f"\nFound {len(files)} monthly file(s): {[os.path.basename(f) for f in files]}")

    # Open all months as one lazily-concatenated dataset (one timestep loaded per iter)
    ds = xr.open_mfdataset(files, combine="by_coords", engine="netcdf4", chunks={"time": 1})
    lats  = ds.latitude.values
    lons  = ds.longitude.values
    times = pd.DatetimeIndex(ds.time.values)
    print(f"  lat: {len(lats)}, lon: {len(lons)}, timesteps: {len(times)}")
    print(f"  Variables in file: {list(ds.data_vars)}")

    # Run inference one timestep at a time
    all_preds = []
    for t in tqdm(times, unit="step"):
        ds_t = ds.sel(time=t)

        # Build (1, lat, lon, level, variable) float32 array.
        # variable_order controls channel ordering — must match what the model was
        # trained on.  Variables not present in the file raise a KeyError here,
        # which is intentional: fix the config rather than silently skip.
        da = (
            ds_t[variable_order]
            .to_array(dim="variable")
            .transpose("latitude", "longitude", "level", "variable")
            .values.astype(np.float32)
        )
        arr = da[np.newaxis, ...]   # (1, lat, lon, level, variable)

        pred = model(arr, training=False)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        pred_np = pred.numpy()[0]   # (lat, lon, n_classes)
        all_preds.append(pred_np[:, :, 1:].astype(np.float32))  # drop no-front class

    preds = np.stack(all_preds, axis=0)   # (time, lat, lon, n_fronts)

    if preds.shape[-1] != len(front_types):
        raise ValueError(
            f"Model output has {preds.shape[-1]} front classes but "
            f"{len(front_types)} --front_types specified: {front_types}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.outfile)), exist_ok=True)
    ds_out = xr.Dataset(
        data_vars={
            ft: (["time", "latitude", "longitude"], preds[:, :, :, i])
            for i, ft in enumerate(front_types)
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    ds_out.to_netcdf(args.outfile)
    print(f"\nPredictions written to {args.outfile}")


if __name__ == "__main__":
    main()
