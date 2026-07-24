r"""Run near-real-time fronts model inference against Earthmover's Arraylake ERA5 store.

Reads the latest (or a given) ERA5 timestep directly from Arraylake, lazily,
subset to the model's variables/levels/domain, and runs it through a trained
model. Normalization is baked into the model checkpoint, so no external
normalization stats file is needed.

Usage:
    pixi run -e schooner python -m fronts.infer \
        --config_path configs/schooner_realtime.yaml

    pixi run -e schooner python -m fronts.infer \
        --config_path configs/schooner_realtime.yaml \
        --init-time 2026-07-24T12:00:00 --plot
"""

import argparse
import dataclasses
import logging
import os

import numpy as np
import tensorflow as tf
import xarray as xr

from fronts import utils
from fronts.data import generate, inputs
from fronts.model import SharedTargetModel

log = logging.getLogger(__name__)

FRONT_TYPE_CLASS_INDEX: dict[str, int] = {"CF": 1, "WF": 2, "SF": 3, "OF": 4, "DL": 5}
_ZARR_ASYNC_CONCURRENCY = 16


@dataclasses.dataclass
class RealtimeInferenceConfig:
    """Configuration for near-real-time inference against Arraylake ERA5.

    Attributes:
        model_path: Path to the saved .keras model checkpoint.
        era5_uri: URI to the ERA5 data, e.g. ``arraylake://earthmover-public/era5``.
        variables: ERA5 variable names (Google ARCO naming), in the same order
            used when the model was trained.
        pressure_levels: Pressure levels to load for each pressure-level variable.
        coordinates: Spatial bounding box matching the model's training domain.
        volume_inputs: If True, build a (level, variable) volume input for a 3D
            Conv3D model instead of a flattened 2D channel input.
        front_types: Front type labels in class order (excluding background class 0).
        time_resolution: Temporal resolution of the source data, e.g. "6h".
        outdir: Directory to write the output NetCDF file.
        gpu_device: GPU index to use. None runs on CPU.
    """

    model_path: str
    era5_uri: str
    variables: list[str]
    pressure_levels: list[int]
    coordinates: utils.BoundingBox
    volume_inputs: bool
    front_types: list[str]
    time_resolution: str
    outdir: str
    gpu_device: int | None


def load_era5_dataset(cfg: RealtimeInferenceConfig, init_time: np.datetime64) -> xr.Dataset:
    """Lazily open a single ERA5 timestep from Arraylake, subset to the model's inputs.

    Args:
        cfg: Real-time inference configuration.
        init_time: Timestep to load.

    Returns:
        Lazy dataset containing exactly ``cfg.variables`` (direct and derived),
        subset to ``cfg.pressure_levels`` and ``cfg.coordinates``.
    """
    era5_config = generate.ERA5DataLoaderConfig(
        era5_uri=cfg.era5_uri,
        variables=cfg.variables,
        pressure_levels=cfg.pressure_levels,
        time_start=init_time.astype("datetime64[us]").item(),
        time_end=init_time.astype("datetime64[us]").item(),
        time_resolution=cfg.time_resolution,
        coordinates=cfg.coordinates,
        storage_options=None,
        chunks=None,
        zarr_async_concurrency=_ZARR_ASYNC_CONCURRENCY,
    )
    return generate.generate_era5_data(era5_config)


def predict_probabilities(model: tf.keras.Model, era5_ds: xr.Dataset, cfg: RealtimeInferenceConfig) -> xr.Dataset:
    """Run model inference on a single-timestep ERA5 dataset.

    Args:
        model: Loaded Keras model with baked-in normalization.
        era5_ds: Single-timestep ERA5 dataset with exactly ``cfg.variables``.
        cfg: Real-time inference configuration.

    Returns:
        Dataset with one probability variable per front type, dims (latitude, longitude).
    """
    if cfg.volume_inputs:
        x_np = inputs.inputs_ds_to_volume_dataarray(era5_ds, cfg.variables).values[0]
    else:
        x_np = inputs.inputs_ds_to_dataarray(era5_ds, cfg.variables).values[0]

    pred = model(x_np[np.newaxis], training=False)
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    pred_np = pred.numpy()[0].astype(np.float32)  # (lat, lon, n_classes)

    lats = era5_ds["latitude"].values
    lons = era5_ds["longitude"].values
    probs_ds = xr.Dataset(coords={"latitude": lats, "longitude": lons})
    for ft in cfg.front_types:
        probs_ds[ft] = (["latitude", "longitude"], pred_np[:, :, FRONT_TYPE_CLASS_INDEX[ft]])
    return probs_ds


def run_inference(cfg: RealtimeInferenceConfig, init_time: np.datetime64) -> xr.Dataset:
    """Load a model and ERA5 data, then run inference for a single timestep.

    Args:
        cfg: Real-time inference configuration.
        init_time: Timestep to run inference for.

    Returns:
        Dataset with one probability variable per front type, dims (latitude, longitude).
    """
    utils.configure_gpu(cfg.gpu_device)
    log.info("Loading model from %s …", cfg.model_path)
    model = tf.keras.models.load_model(
        cfg.model_path, compile=False, custom_objects={"SharedTargetModel": SharedTargetModel}
    )

    log.info("Opening ERA5 data from %s for %s …", cfg.era5_uri, init_time)
    era5_ds = load_era5_dataset(cfg, init_time)

    return predict_probabilities(model, era5_ds, cfg)


def write_output(probs_ds: xr.Dataset, outdir: str, init_time: np.datetime64) -> str:
    """Write a probability Dataset to NetCDF, named by its init time.

    Args:
        probs_ds: Per-front-type probability Dataset, dims (latitude, longitude).
        outdir: Directory to write into; created if missing.
        init_time: Timestep the prediction was made for.

    Returns:
        Path to the written NetCDF file.
    """
    os.makedirs(outdir, exist_ok=True)
    timestamp = np.datetime_as_string(init_time, unit="h")
    path = os.path.join(outdir, f"fronts_{timestamp}.nc")
    probs_ds.to_netcdf(path)
    log.info("Wrote %s", path)
    return path


def main() -> None:
    """Entry point for near-real-time fronts inference from config."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run near-real-time fronts model inference against Arraylake ERA5.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to real-time inference config YAML.")
    parser.add_argument(
        "--init-time",
        type=str,
        default="latest",
        help="ISO8601 timestep to run inference for, or 'latest' to use the most recent available.",
    )
    parser.add_argument("--plot", action="store_true", help="Also render a probability map PNG to outdir.")
    args = parser.parse_args()

    cfg = utils.open_config_yaml_as_dataclass(
        args.config_path, RealtimeInferenceConfig, config_key="realtime_config", type_hooks=utils.YAML_TYPE_HOOKS
    )

    if args.init_time == "latest":
        from fronts.data import sources

        init_time = sources.latest_arraylake_time(cfg.era5_uri)
    else:
        init_time = np.datetime64(args.init_time)
    log.info("Running inference for init time %s", init_time)

    probs_ds = run_inference(cfg, init_time)
    write_output(probs_ds, cfg.outdir, init_time)

    if args.plot:
        from fronts.plot.plot import plot_test_prediction

        fig = plot_test_prediction(
            probs_ds["latitude"].values,
            probs_ds["longitude"].values,
            probs_ds,
            cfg.front_types,
            filled_contours=True,
            open_contours=True,
            title=str(init_time),
        )
        os.makedirs(cfg.outdir, exist_ok=True)
        timestamp = np.datetime_as_string(init_time, unit="h")
        fig_path = os.path.join(cfg.outdir, f"fronts_{timestamp}.png")
        fig.savefig(fig_path, bbox_inches="tight", dpi=200)
        log.info("Wrote %s", fig_path)


if __name__ == "__main__":
    main()
