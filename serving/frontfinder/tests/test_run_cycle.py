import numpy as np
import pytest
import xarray as xr

from frontfinder.config.manifests import ALL_CLASSES, BEST_LOSS_MANIFEST, MODEL_1702_MANIFEST
from frontfinder.ingest.ecmwf_ifs import FakeIFSFieldSource, IFSCycle
from frontfinder.scheduler.run_cycle import ModelRunConfig, run_cycle, run_one_model


class FakePredictor:
    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        n, h, w, _ = patches.shape
        out = np.random.default_rng(1).random((n, h, w, len(ALL_CLASSES))).astype(np.float32)
        return out / out.sum(axis=-1, keepdims=True)  # looks like a softmax output


@pytest.fixture
def tiny_source():
    lat = np.linspace(25.0, 56.75, 64)
    lon = np.linspace(228.0, 299.75, 64)
    return FakeIFSFieldSource(lat, lon, seed=7)


def test_run_one_model_writes_a_readable_zarr_pyramid(tiny_source, tmp_path):
    cycle = IFSCycle(date="2026-08-19", run_hour=0)
    run_config = ModelRunConfig(
        manifest=MODEL_1702_MANIFEST,
        predictor=FakePredictor(),
        patch_size=64,
        overlap=16,
        n_pyramid_levels=2,
    )
    store_path = run_one_model(run_config, tiny_source, cycle, str(tmp_path))

    reopened = xr.open_datatree(store_path, engine="zarr", consolidated=True)
    lvl0 = reopened["0"].to_dataset()
    assert set(lvl0.data_vars) == set(MODEL_1702_MANIFEST.served_classes)
    assert lvl0.attrs["model"] == "model_1702"
    assert lvl0.attrs["cycle_time"] == "2026-08-19T00:00:00"


def test_run_one_model_writes_latest_pointer_json(tiny_source, tmp_path):
    import json

    cycle = IFSCycle(date="2026-08-19", run_hour=0)
    run_config = ModelRunConfig(
        manifest=MODEL_1702_MANIFEST,
        predictor=FakePredictor(),
        patch_size=64,
        overlap=16,
        n_pyramid_levels=2,
    )
    run_one_model(run_config, tiny_source, cycle, str(tmp_path))

    pointer_path = tmp_path / "model_1702" / "latest.json"
    assert pointer_path.exists()
    pointer = json.loads(pointer_path.read_text())
    assert pointer["store"] == "2026-08-19T00Z.zarr"
    assert pointer["cycle_time"] == "2026-08-19T00:00:00"


def test_run_cycle_runs_both_models_and_returns_both_paths(tiny_source, tmp_path):
    cycle = IFSCycle(date="2026-08-19", run_hour=6)
    configs = [
        ModelRunConfig(manifest=BEST_LOSS_MANIFEST, predictor=FakePredictor(), patch_size=64, overlap=16, n_pyramid_levels=2),
        ModelRunConfig(manifest=MODEL_1702_MANIFEST, predictor=FakePredictor(), patch_size=64, overlap=16, n_pyramid_levels=2),
    ]
    results = run_cycle(configs, tiny_source, cycle, str(tmp_path))
    assert set(results.keys()) == {"best_loss", "model_1702"}
    for path in results.values():
        assert path.endswith(".zarr")


class AlwaysFailsPredictor:
    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        raise RuntimeError("boom")


def test_run_cycle_continues_after_one_model_fails(tiny_source, tmp_path):
    cycle = IFSCycle(date="2026-08-19", run_hour=6)
    configs = [
        ModelRunConfig(manifest=BEST_LOSS_MANIFEST, predictor=AlwaysFailsPredictor(), patch_size=64, overlap=16, n_pyramid_levels=2),
        ModelRunConfig(manifest=MODEL_1702_MANIFEST, predictor=FakePredictor(), patch_size=64, overlap=16, n_pyramid_levels=2),
    ]
    results = run_cycle(configs, tiny_source, cycle, str(tmp_path))
    assert "best_loss" not in results
    assert "model_1702" in results
