"""Tests for the UNet3Plus builder, focused on the 3D (Conv3D volume-input) instantiation."""

import numpy as np
import pytest

try:
    import tensorflow as tf

    from fronts.model import UNet3Plus

    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _TF_AVAILABLE, reason="TensorFlow not available")

N_LEVELS = 4
N_VARS = 3
N_CLASSES = 6
_LEVELS_3D = 3


def _build_3d_model(normalization_mean=None, normalization_variance=None) -> "tf.keras.Model":
    return UNet3Plus(
        input_shape=(None, None, N_LEVELS, N_VARS),
        num_classes=N_CLASSES,
        pool_size=(2, 2, 1),
        upsample_size=(2, 2, 1),
        levels=_LEVELS_3D,
        filter_num=[4, 8, 16],
        kernel_size=3,
        squeeze_axes=3,
        first_encoder_connections=True,
        deep_supervision=True,
        batch_normalization=True,
        activation="gelu",
        output_activation="softmax",
        modules_per_node=1,
        normalization_mean=normalization_mean,
        normalization_variance=normalization_variance,
    ).build()


@pytest.fixture(scope="module")
def built_3d_model() -> "tf.keras.Model":
    return _build_3d_model()


class Test3DBuild:
    def test_model_name_marks_3d(self, built_3d_model):
        assert built_3d_model.name == "unet_3plus_3D"

    def test_all_convolutions_are_3d(self, built_3d_model):
        conv_classes = {type(layer).__name__ for layer in built_3d_model.layers if "Conv" in type(layer).__name__}
        assert "Conv3D" in conv_classes
        assert "Conv2D" not in conv_classes

    def test_one_output_per_supervised_level(self, built_3d_model):
        assert len(built_3d_model.outputs) == _LEVELS_3D

    def test_outputs_are_2d_class_maps(self, built_3d_model):
        """The squeeze collapse must drop the level axis: (batch, lat, lon, classes)."""
        for out in built_3d_model.outputs:
            assert len(out.shape) == 4
            assert out.shape[-1] == N_CLASSES


class Test3DForwardPass:
    def test_forward_pass_shapes_and_softmax(self, built_3d_model):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((2, 16, 16, N_LEVELS, N_VARS)).astype(np.float32)
        outs = built_3d_model(x, training=False)
        outs = outs if isinstance(outs, (list, tuple)) else [outs]
        for out in outs:
            values = out.numpy()
            assert values.shape == (2, 16, 16, N_CLASSES)
            assert np.isfinite(values).all()
            np.testing.assert_allclose(values.sum(axis=-1), 1.0, atol=1e-4)

    def test_dynamic_spatial_dims(self, built_3d_model):
        """The squeeze path must handle spatial sizes unknown at build time."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal((1, 24, 32, N_LEVELS, N_VARS)).astype(np.float32)
        outs = built_3d_model(x, training=False)
        outs = outs if isinstance(outs, (list, tuple)) else [outs]
        assert outs[0].shape == (1, 24, 32, N_CLASSES)


class Test3DNormalization:
    def test_level_variable_stats_build_and_run(self):
        rng = np.random.default_rng(2)
        mean = rng.standard_normal((N_LEVELS, N_VARS)).astype(np.float32)
        variance = (np.abs(rng.standard_normal((N_LEVELS, N_VARS))) + 0.5).astype(np.float32)
        built = _build_3d_model(normalization_mean=mean, normalization_variance=variance)
        assert any(layer.name == "input_normalization" for layer in built.layers)

        x = rng.standard_normal((1, 16, 16, N_LEVELS, N_VARS)).astype(np.float32)
        outs = built(x, training=False)
        outs = outs if isinstance(outs, (list, tuple)) else [outs]
        assert np.isfinite(outs[0].numpy()).all()

    def test_normalization_layer_standardizes_per_level_and_variable(self):
        rng = np.random.default_rng(3)
        mean = rng.standard_normal((N_LEVELS, N_VARS)).astype(np.float32)
        variance = (np.abs(rng.standard_normal((N_LEVELS, N_VARS))) + 0.5).astype(np.float32)
        built = _build_3d_model(normalization_mean=mean, normalization_variance=variance)
        norm_layer = built.get_layer("input_normalization")

        sample = mean.reshape(1, 1, 1, N_LEVELS, N_VARS)
        out = norm_layer(sample).numpy().reshape(N_LEVELS, N_VARS)
        np.testing.assert_allclose(out, 0.0, atol=1e-5)


class Test3DSerialization:
    def test_save_load_round_trip(self, built_3d_model, tmp_path):
        """The ops.squeeze collapse must survive .keras serialization — checkpoints get reloaded."""
        rng = np.random.default_rng(4)
        x = rng.standard_normal((1, 16, 16, N_LEVELS, N_VARS)).astype(np.float32)
        original = built_3d_model(x, training=False)
        original = original if isinstance(original, (list, tuple)) else [original]

        path = str(tmp_path / "model_3d.keras")
        built_3d_model.save(path)
        reloaded = tf.keras.models.load_model(path, compile=False)
        again = reloaded(x, training=False)
        again = again if isinstance(again, (list, tuple)) else [again]

        for a, b in zip(original, again, strict=True):
            np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-5)


class Test2DStillWorks:
    def test_2d_build_and_forward_unchanged(self):
        built = UNet3Plus(
            input_shape=(None, None, 8),
            num_classes=N_CLASSES,
            pool_size=(2, 2),
            upsample_size=(2, 2),
            levels=3,
            filter_num=[4, 8, 16],
            kernel_size=3,
            deep_supervision=False,
            output_activation="softmax",
        ).build()
        assert built.name == "unet_3plus_2D"
        rng = np.random.default_rng(5)
        x = rng.standard_normal((1, 16, 16, 8)).astype(np.float32)
        out = built(x, training=False)
        out = out[0] if isinstance(out, (list, tuple)) else out
        assert out.shape == (1, 16, 16, N_CLASSES)
        assert np.isfinite(out.numpy()).all()
