import dataclasses
import pathlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

try:
    import tensorflow as tf

    from fronts import infer
    from fronts import model as fronts_model
    from fronts.utils import BoundingBox

    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _TF_AVAILABLE, reason="TensorFlow not available")

_VARS = ["temperature", "specific_humidity"]
_LEVELS = [1000, 500]
_LAT = np.linspace(45.0, 25.0, 4)
_LON = np.linspace(200.0, 220.0, 4)
_TIME = pd.date_range("2019-01-01", periods=2, freq="6h")
_FRONT_TYPES = ["CF", "WF"]
_N_CHANNELS = len(_LEVELS) * len(_VARS)


@pytest.fixture
def minimal_ds() -> xr.Dataset:
    rng = np.random.default_rng(0)
    shape = (len(_TIME), len(_LEVELS), len(_LAT), len(_LON))
    coords = {"time": _TIME, "level": _LEVELS, "latitude": _LAT, "longitude": _LON}
    dims = ["time", "level", "latitude", "longitude"]
    return xr.Dataset(
        {var: xr.DataArray(rng.standard_normal(shape).astype(np.float32), dims=dims, coords=coords) for var in _VARS}
    )


@pytest.fixture
def era5_zarr(tmp_path: pathlib.Path, minimal_ds: xr.Dataset) -> str:
    path = tmp_path / "era5.zarr"
    minimal_ds.to_zarr(path)
    return str(path)


@pytest.fixture
def cfg(era5_zarr: str) -> "infer.RealtimeInferenceConfig":
    return infer.RealtimeInferenceConfig(
        model_path="unused",
        era5_uri=era5_zarr,
        variables=_VARS,
        pressure_levels=_LEVELS,
        coordinates=BoundingBox(lat_min=25.0, lat_max=45.0, lon_min=200.0, lon_max=220.0),
        volume_inputs=False,
        front_types=_FRONT_TYPES,
        time_resolution="6h",
        outdir="unused",
        gpu_device=None,
    )


def _build_tiny_model() -> "tf.keras.Model":
    return fronts_model.UNet3Plus(
        input_shape=(None, None, _N_CHANNELS),
        num_classes=6,
        pool_size=(2, 2),
        upsample_size=(2, 2),
        levels=3,
        filter_num=[4, 8, 16],
        deep_supervision=False,
        output_activation="softmax",
        normalization_method="minmax",
        normalization_stat_a=np.zeros(_N_CHANNELS, dtype=np.float32),
        normalization_stat_b=np.ones(_N_CHANNELS, dtype=np.float32),
    ).build()


class TestLoadEra5Dataset:
    def test_returns_requested_variables(self, cfg: "infer.RealtimeInferenceConfig"):
        ds = infer.load_era5_dataset(cfg, np.datetime64(_TIME[0]))
        assert set(ds.data_vars) == set(_VARS)

    def test_subsets_to_single_timestep(self, cfg: "infer.RealtimeInferenceConfig"):
        ds = infer.load_era5_dataset(cfg, np.datetime64(_TIME[0]))
        assert ds.sizes["time"] == 1

    def test_subsets_to_requested_levels(self, cfg: "infer.RealtimeInferenceConfig"):
        ds = infer.load_era5_dataset(cfg, np.datetime64(_TIME[0]))
        assert list(map(int, ds.level.values)) == _LEVELS

    def test_subsets_to_bounding_box(self, cfg: "infer.RealtimeInferenceConfig"):
        ds = infer.load_era5_dataset(cfg, np.datetime64(_TIME[0]))
        assert float(ds.latitude.min()) >= 25.0
        assert float(ds.latitude.max()) <= 45.0


class TestPredictProbabilities:
    def test_output_has_one_variable_per_front_type(self, cfg: "infer.RealtimeInferenceConfig"):
        model = _build_tiny_model()
        era5_t = infer.load_era5_dataset(cfg, np.datetime64(_TIME[0]))
        probs_ds = infer.predict_probabilities(model, era5_t, cfg)
        assert set(probs_ds.data_vars) == set(_FRONT_TYPES)

    def test_output_dims_match_domain(self, cfg: "infer.RealtimeInferenceConfig"):
        model = _build_tiny_model()
        era5_t = infer.load_era5_dataset(cfg, np.datetime64(_TIME[0]))
        probs_ds = infer.predict_probabilities(model, era5_t, cfg)
        assert probs_ds.sizes["latitude"] == len(_LAT)
        assert probs_ds.sizes["longitude"] == len(_LON)

    def test_output_probabilities_are_valid(self, cfg: "infer.RealtimeInferenceConfig"):
        model = _build_tiny_model()
        era5_t = infer.load_era5_dataset(cfg, np.datetime64(_TIME[0]))
        probs_ds = infer.predict_probabilities(model, era5_t, cfg)
        for ft in _FRONT_TYPES:
            values = probs_ds[ft].values
            assert np.isfinite(values).all()
            assert (values >= 0.0).all()
            assert (values <= 1.0).all()


class TestWriteOutput:
    def test_writes_readable_netcdf(self, tmp_path: pathlib.Path):
        probs_ds = xr.Dataset(
            {"CF": (["latitude", "longitude"], np.zeros((2, 2), dtype=np.float32))},
            coords={"latitude": [10.0, 20.0], "longitude": [30.0, 40.0]},
        )
        path = infer.write_output(probs_ds, str(tmp_path), np.datetime64("2019-01-01T00"))
        assert pathlib.Path(path).exists()
        reopened = xr.open_dataset(path)
        np.testing.assert_array_equal(reopened["CF"].values, probs_ds["CF"].values)

    def test_filename_includes_init_time(self, tmp_path: pathlib.Path):
        probs_ds = xr.Dataset(
            {"CF": (["latitude", "longitude"], np.zeros((1, 1), dtype=np.float32))},
            coords={"latitude": [10.0], "longitude": [30.0]},
        )
        path = infer.write_output(probs_ds, str(tmp_path), np.datetime64("2019-03-15T06"))
        assert "2019-03-15T06" in path


class TestRunInference:
    def test_end_to_end_with_saved_checkpoint(self, cfg: "infer.RealtimeInferenceConfig", tmp_path: pathlib.Path):
        model = _build_tiny_model()
        model_path = tmp_path / "model.keras"
        model.save(model_path)

        real_cfg = dataclasses.replace(cfg, model_path=str(model_path))
        probs_ds = infer.run_inference(real_cfg, np.datetime64(_TIME[0]))

        assert set(probs_ds.data_vars) == set(_FRONT_TYPES)
        assert probs_ds.sizes["latitude"] == len(_LAT)
        assert probs_ds.sizes["longitude"] == len(_LON)
