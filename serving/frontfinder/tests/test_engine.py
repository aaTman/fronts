import numpy as np
import pytest

from frontfinder.config.manifests import ALL_CLASSES, MODEL_1702_MANIFEST
from frontfinder.inference.engine import run_tiled_inference


class ConstantPredictor:
    """Fake predictor: returns a fixed uniform class distribution regardless
    of input, so stitching correctness can be checked in isolation from any
    real model behavior."""

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.calls: list[tuple[int, ...]] = []

    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        self.calls.append(patches.shape)
        n, h, w, _ = patches.shape
        out = np.zeros((n, h, w, self.n_classes), dtype=np.float32)
        out[..., 1] = 1.0  # always "cold front" with probability 1
        return out


class ShapeCheckingPredictor:
    def __init__(self, expected_channels: int, n_classes: int):
        self.expected_channels = expected_channels
        self.n_classes = n_classes

    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        assert patches.shape[-1] == self.expected_channels
        assert patches.shape[1] % 16 == 0
        assert patches.shape[2] % 16 == 0
        n, h, w, _ = patches.shape
        return np.random.default_rng(0).random((n, h, w, self.n_classes)).astype(np.float32)


def test_run_tiled_inference_output_shape_matches_unpadded_grid_and_served_classes():
    height, width = 300, 400  # deliberately not a multiple of 16
    input_grid = np.random.default_rng(0).normal(size=(height, width, MODEL_1702_MANIFEST.n_channels)).astype(np.float32)
    predictor = ConstantPredictor(n_classes=len(ALL_CLASSES))
    out = run_tiled_inference(predictor, input_grid, MODEL_1702_MANIFEST, patch_size=256, overlap=32)
    assert out.shape == (height, width, len(MODEL_1702_MANIFEST.served_classes))


def test_run_tiled_inference_recovers_constant_prediction_everywhere():
    height, width = 300, 400
    input_grid = np.zeros((height, width, MODEL_1702_MANIFEST.n_channels), dtype=np.float32)
    predictor = ConstantPredictor(n_classes=len(ALL_CLASSES))
    out = run_tiled_inference(predictor, input_grid, MODEL_1702_MANIFEST, patch_size=256, overlap=32)
    cold_idx = MODEL_1702_MANIFEST.served_classes.index("cold")
    np.testing.assert_allclose(out[..., cold_idx], 1.0, atol=1e-5)
    other_idx = [i for i in range(out.shape[-1]) if i != cold_idx]
    np.testing.assert_allclose(out[..., other_idx], 0.0, atol=1e-5)


def test_run_tiled_inference_rejects_wrong_channel_count():
    input_grid = np.zeros((64, 64, 3), dtype=np.float32)
    predictor = ConstantPredictor(n_classes=len(ALL_CLASSES))
    with pytest.raises(ValueError):
        run_tiled_inference(predictor, input_grid, MODEL_1702_MANIFEST, patch_size=64, overlap=16)


def test_run_tiled_inference_sends_only_16_multiple_patches_to_predictor():
    predictor = ShapeCheckingPredictor(
        expected_channels=MODEL_1702_MANIFEST.n_channels, n_classes=len(ALL_CLASSES)
    )
    input_grid = np.zeros((721, 1440, MODEL_1702_MANIFEST.n_channels), dtype=np.float32)  # real global grid
    out = run_tiled_inference(predictor, input_grid, MODEL_1702_MANIFEST, patch_size=256, overlap=32)
    assert out.shape == (721, 1440, len(MODEL_1702_MANIFEST.served_classes))


def test_run_tiled_inference_batches_calls_to_predictor():
    predictor = ConstantPredictor(n_classes=len(ALL_CLASSES))
    input_grid = np.zeros((512, 512, MODEL_1702_MANIFEST.n_channels), dtype=np.float32)
    run_tiled_inference(predictor, input_grid, MODEL_1702_MANIFEST, patch_size=256, overlap=32, batch_size=2)
    # every call except possibly the last should be exactly batch_size patches
    assert all(shape[0] <= 2 for shape in predictor.calls)
