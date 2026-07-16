"""Tests for UNet3Plus's min-max input normalization layer."""

import numpy as np
import pytest

try:
    import tensorflow as tf

    from fronts import model as fronts_model

    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _TF_AVAILABLE, reason="TensorFlow not available")

N_CHANNELS = 4
_INPUT_SHAPE = (None, None, N_CHANNELS)


def _build_model(normalization_min=None, normalization_max=None) -> "tf.keras.Model":
    return fronts_model.UNet3Plus(
        input_shape=_INPUT_SHAPE,
        num_classes=6,
        pool_size=(2, 2),
        upsample_size=(2, 2),
        levels=3,
        filter_num=[8, 16, 32],
        deep_supervision=False,
        output_activation="softmax",
        normalization_min=normalization_min,
        normalization_max=normalization_max,
    ).build()


class TestMinMaxNormalization:
    def test_no_stats_skips_normalization_layer(self):
        built = _build_model()
        assert not any(layer.name == "input_normalization" for layer in built.layers)

    def test_min_maps_to_zero_and_max_to_one(self):
        min_val = np.array([0.0, -10.0, 100.0, 1e-6], dtype=np.float32)
        max_val = np.array([1.0, 10.0, 200.0, 2e-6], dtype=np.float32)
        built = _build_model(min_val, max_val)

        norm_layer = built.get_layer("input_normalization")
        at_min = norm_layer(min_val.reshape(1, 1, 1, N_CHANNELS)).numpy().reshape(-1)
        at_max = norm_layer(max_val.reshape(1, 1, 1, N_CHANNELS)).numpy().reshape(-1)
        np.testing.assert_allclose(at_min, 0.0, atol=1e-5)
        np.testing.assert_allclose(at_max, 1.0, atol=1e-5)

    def test_midpoint_maps_to_half(self):
        min_val = np.array([0.0, -10.0, 100.0, 1e-6], dtype=np.float32)
        max_val = np.array([1.0, 10.0, 200.0, 2e-6], dtype=np.float32)
        built = _build_model(min_val, max_val)

        norm_layer = built.get_layer("input_normalization")
        midpoint = (min_val + max_val) / 2
        out = norm_layer(midpoint.reshape(1, 1, 1, N_CHANNELS)).numpy().reshape(-1)
        np.testing.assert_allclose(out, 0.5, atol=1e-5)

    def test_tiny_variance_channel_does_not_overflow_or_explode(self):
        """Test tiny variance to make sure overflow.

        A near-zero-variance channel (like specific_humidity/potential_vorticity in
        native units) must not produce huge normalized values the way z-score would.
        """
        min_val = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        max_val = np.array([1.0, 1.0, 1.0, 1e-12], dtype=np.float32)
        built = _build_model(min_val, max_val)

        norm_layer = built.get_layer("input_normalization")
        sample = np.array([0.5, 0.5, 0.5, 5e-13], dtype=np.float32)
        out = norm_layer(sample.reshape(1, 1, 1, N_CHANNELS)).numpy().reshape(-1)
        assert np.isfinite(out).all()
        assert np.abs(out).max() < 10.0

    def test_zero_range_channel_does_not_produce_nan(self):
        """A perfectly constant channel (max == min) must not divide by zero."""
        min_val = np.array([0.0, 5.0, -3.0, 0.0], dtype=np.float32)
        max_val = np.array([1.0, 5.0, -3.0, 1.0], dtype=np.float32)
        built = _build_model(min_val, max_val)

        norm_layer = built.get_layer("input_normalization")
        sample = np.array([0.5, 5.0, -3.0, 0.5], dtype=np.float32)
        out = norm_layer(sample.reshape(1, 1, 1, N_CHANNELS)).numpy().reshape(-1)
        assert np.isfinite(out).all()

    def test_out_of_range_sample_extrapolates_linearly_without_clipping(self):
        """Matches the AIES reference's (value - min) / (max - min), which does not clip."""
        min_val = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        max_val = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        built = _build_model(min_val, max_val)

        norm_layer = built.get_layer("input_normalization")
        sample = np.array([2.0, -1.0, 0.5, 1.5], dtype=np.float32)
        out = norm_layer(sample.reshape(1, 1, 1, N_CHANNELS)).numpy().reshape(-1)
        np.testing.assert_allclose(out, sample, atol=1e-5)

    def test_forward_pass_end_to_end_with_extreme_channel_range(self):
        """Regression check for the fp16 overflow this replaces.

        A physically tiny-scale channel (like potential_vorticity, range ~1e-12) must
        not blow up the full model's forward pass, unlike the old z-score Normalization layer.
        """
        min_val = np.array([0.0, -50.0, 900.0, -1e-12], dtype=np.float32)
        max_val = np.array([1.0, 50.0, 1100.0, 1e-12], dtype=np.float32)
        built = _build_model(min_val, max_val)

        rng = np.random.default_rng(0)
        x = rng.uniform(low=min_val, high=max_val, size=(2, 16, 16, N_CHANNELS)).astype(np.float32)
        out = built(x, training=False)
        out_np = out.numpy() if not isinstance(out, (list, tuple)) else out[0].numpy()
        assert np.isfinite(out_np).all()
