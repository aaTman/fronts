"""Generate front probability predictions from a .keras model and TF dataset shards.

Designed for the new training pipeline where test data is saved as numbered shards
(e.g. 2019-1_tf, 2019-2_tf, ..., 2019-100_tf) rather than monthly files.

Shards are loaded in numeric order, predictions are run in batches, and the result
is written to a single NetCDF with one variable per front type.

Lat/lon/time coordinates are taken from --fronts_file (the truth data NetCDF),
which is expected to have (time, latitude, longitude) dimensions.

Usage:
    PYTHONPATH=src python src/fronts/evaluation/predict_from_tf_shards.py \\
        --model_path ~/models/fronts/1702.keras \\
        --tf_indir ~/data/tf_datasets \\
        --fronts_file ~/data/fronts_subset.nc \\
        --front_types CF WF SF OF DL \\
        --outfile ~/models/fronts/predictions_2019.nc
"""

import argparse
import io
import json
import os
import re
import tempfile
import zipfile

import numpy as np
import xarray as xr
import tensorflow as tf


class SqueezeAxes(tf.keras.layers.Layer):
    """Replaces Lambda squeeze layers that were saved with Python 3.11 bytecode.

    The deep_supervision_side_output function creates Lambda layers that do
    tf.squeeze(x, axis=squeeze_axes). These are patched out at load time
    because Python's marshal bytecode format is not cross-version compatible
    (model was saved on Python 3.11, loaded on Python 3.12).
    """

    def __init__(self, squeeze_axes: int, **kwargs):
        super().__init__(**kwargs)
        self.squeeze_axes = squeeze_axes

    def call(self, x, mask=None):
        return tf.squeeze(x, axis=self.squeeze_axes)

    def get_config(self):
        cfg = super().get_config()
        cfg["squeeze_axes"] = self.squeeze_axes
        return cfg


def _patch_lambda_layers(config: dict) -> dict:
    """Recursively replace Lambda squeeze layers in a Keras model config dict."""
    if isinstance(config, dict):
        if config.get("class_name") == "Lambda":
            fn_cfg = config.get("config", {}).get("function", {}).get("config", {})
            closure = fn_cfg.get("closure")
            if closure and len(closure) == 1:
                axes = closure[0][0] if isinstance(closure[0], list) else closure[0]
                layer_cfg = config.get("config", {})
                return {
                    "module": "predict_from_tf_shards",
                    "class_name": "SqueezeAxes",
                    "registered_name": "SqueezeAxes",
                    "config": {
                        "name": layer_cfg.get("name", "squeeze"),
                        "trainable": layer_cfg.get("trainable", True),
                        "dtype": layer_cfg.get("dtype"),
                        "squeeze_axes": axes,
                    },
                    "build_config": config.get("build_config"),
                    "name": config.get("name"),
                    "inbound_nodes": config.get("inbound_nodes"),
                }
        return {k: _patch_lambda_layers(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_patch_lambda_layers(v) for v in config]
    return config


def load_model_patched(model_path: str) -> tf.keras.Model:
    """Load a .keras model, patching Lambda squeeze layers for cross-Python compat."""
    with zipfile.ZipFile(model_path, "r") as zin:
        original_config = json.loads(zin.read("config.json"))
        patched_config = _patch_lambda_layers(original_config)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
            for name in zin.namelist():
                if name == "config.json":
                    zout.writestr(name, json.dumps(patched_config))
                else:
                    zout.writestr(name, zin.read(name))

    buf.seek(0)
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp.write(buf.read())
        tmp_path = tmp.name

    try:
        model = tf.keras.models.load_model(
            tmp_path,
            custom_objects={"SqueezeAxes": SqueezeAxes},
            safe_mode=False,
            compile=False,
        )
    finally:
        os.unlink(tmp_path)

    return model


def _numeric_shard_key(path: str) -> int:
    """Extract the numeric suffix from a shard directory name like 2019-42_tf."""
    m = re.search(r"-(\d+)_tf$", os.path.basename(path))
    return int(m.group(1)) if m else 0


def load_shards(tf_indir: str) -> tf.data.Dataset:
    """Concatenate all *_tf shards in tf_indir, sorted by numeric suffix."""
    shard_dirs = sorted(
        [
            os.path.join(tf_indir, d)
            for d in os.listdir(tf_indir)
            if d.endswith("_tf") and os.path.isdir(os.path.join(tf_indir, d))
        ],
        key=_numeric_shard_key,
    )
    if not shard_dirs:
        raise FileNotFoundError(f"No *_tf shard directories found in {tf_indir}")
    print(f"Found {len(shard_dirs)} shards ({os.path.basename(shard_dirs[0])} … "
          f"{os.path.basename(shard_dirs[-1])})")

    full_ds = tf.data.Dataset.load(shard_dirs[0])
    for shard_dir in shard_dirs[1:]:
        full_ds = full_ds.concatenate(tf.data.Dataset.load(shard_dir))
    return full_ds


def run_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    batch_size: int,
) -> np.ndarray:
    """Run model predictions over the full dataset and return a numpy array.

    Handles deep supervision (multi-output models) by using only the first output.
    Squeezes a leading size-1 batch dimension if the raw TF dataset records have
    shape (1, lat, lon, level, var) from the xbatcher pipeline.

    Returns array of shape (N, lat, lon, num_classes).
    """
    import math
    from tqdm import tqdm

    card = dataset.cardinality().numpy()
    total_batches = math.ceil(card / batch_size) if card > 0 else None
    predictions = []
    for x, _ in tqdm(dataset.batch(batch_size), total=total_batches, unit="batch"):
        pred = model(x, training=False)

        # Deep supervision: the model returns a list of outputs; use the first.
        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        predictions.append(pred.numpy())

    return np.concatenate(predictions, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate predictions from a .keras model + TF dataset shards."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the .keras model file or SavedModel directory.",
    )
    parser.add_argument(
        "--tf_indir",
        type=str,
        required=True,
        help="Directory containing the *_tf shard subdirectories.",
    )
    parser.add_argument(
        "--fronts_file",
        type=str,
        required=True,
        help=(
            "Path to the truth fronts NetCDF (e.g. fronts_subset.nc). "
            "Used to extract lat/lon/time coordinates for the output dataset."
        ),
    )
    parser.add_argument(
        "--front_types",
        type=str,
        nargs="+",
        required=True,
        help="Front type names, one per output class (excluding the no-front class 0).",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Output NetCDF path for the probability predictions.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for model.predict() (default: 32).",
    )
    parser.add_argument(
        "--gpu_device",
        type=int,
        default=None,
        help="GPU device index. Omit to run on CPU.",
    )
    parser.add_argument(
        "--lat_max",
        type=float,
        default=60.0,
        help="Latitude of the northernmost ERA5 grid point (default: 60.0).",
    )
    parser.add_argument(
        "--lon_min",
        type=float,
        default=-140.0,
        help="Longitude of the westernmost ERA5 grid point (default: -140.0).",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.25,
        help="ERA5 grid resolution in degrees (default: 0.25).",
    )
    args = parser.parse_args()

    # GPU / CPU setup
    if args.gpu_device is not None:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_visible_devices(gpus[args.gpu_device], "GPU")
            tf.config.experimental.set_memory_growth(gpus[args.gpu_device], True)
            print(f"Using GPU {args.gpu_device}: {gpus[args.gpu_device].name}")
        else:
            print("WARNING: No GPUs found; running on CPU.")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("Running on CPU.")

    # Load model, patching Lambda squeeze layers that have Python-version-specific
    # bytecode (model saved on Python 3.11, loaded on Python 3.12).
    print(f"Loading model from {args.model_path} …")
    model = load_model_patched(args.model_path)
    n_outputs = len(model.outputs)
    print(f"Model loaded. {n_outputs} output(s).")

    # Load shards
    dataset = load_shards(args.tf_indir)

    # Infer per-record element spec to report shape
    elem_spec = dataset.element_spec
    print(f"Dataset element spec: inputs={elem_spec[0]}, targets={elem_spec[1]}")

    # Load time from truth fronts file (lat/lon come from the prediction shape)
    print(f"Loading time coordinates from {args.fronts_file} …")
    fronts_ds = xr.open_dataset(args.fronts_file)
    times = fronts_ds.time.values
    fronts_ds.close()
    print(f"  time: {len(times)} steps")

    # Run predictions
    print(f"Running predictions (batch_size={args.batch_size}) …")
    raw_preds = run_predictions(model, dataset, args.batch_size)
    print(f"Raw prediction shape: {raw_preds.shape}")

    # Drop the no-front class (index 0) — keep only the front-type classes
    preds = raw_preds[..., 1:]  # (N, lat, lon, num_front_types)

    n_preds, n_times = preds.shape[0], len(times)
    if n_preds != n_times:
        n = min(n_preds, n_times)
        print(
            f"WARNING: {n_preds} prediction timesteps vs {n_times} fronts timesteps "
            f"— using first {n} of each."
        )
        preds = preds[:n]
        times = times[:n]

    # Validate front_types length matches model output classes
    if preds.shape[-1] != len(args.front_types):
        raise ValueError(
            f"Model has {preds.shape[-1]} front-type outputs but "
            f"{len(args.front_types)} --front_types were specified: {args.front_types}"
        )

    # Derive lat/lon from prediction spatial shape and ERA5 grid parameters.
    # The fronts file may be on a different grid, so we reconstruct coordinates
    # from the known ERA5 domain extent and resolution.
    pred_nlat, pred_nlon = preds.shape[1], preds.shape[2]
    lats = np.arange(args.lat_max, args.lat_max - pred_nlat * args.resolution, -args.resolution)[:pred_nlat]
    lons = np.arange(args.lon_min, args.lon_min + pred_nlon * args.resolution, args.resolution)[:pred_nlon]
    print(f"  lat: {len(lats)} points ({lats[0]:.2f} … {lats[-1]:.2f}), "
          f"lon: {len(lons)} points ({lons[0]:.2f} … {lons[-1]:.2f})")

    # Write output NetCDF
    os.makedirs(os.path.dirname(os.path.abspath(args.outfile)), exist_ok=True)
    ds_out = xr.Dataset(
        data_vars={
            ft: (["time", "latitude", "longitude"], preds[:, :, :, i].astype("float32"))
            for i, ft in enumerate(args.front_types)
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    ds_out.to_netcdf(args.outfile)
    print(f"Predictions written to {args.outfile}")


if __name__ == "__main__":
    main()
